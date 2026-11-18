#include "Run.h"
#include "DecoderLayer.h"
#include "../scheduler/LanguageScheduler.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace arago {
namespace {

json load_json(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Cannot open " + path);
  return json::parse(in);
}

uint64_t unsigned_field(const json& j, const char* key) {
  const auto& value = j.at(key);
  if (!value.is_number_unsigned()) throw std::invalid_argument(std::string(key) + " must be an unsigned integer");
  return value.get<uint64_t>();
}

HardwareConfig hardware_config(const json& j) {
  if (j.at("target") != "arago-dual-array" || unsigned_field(j, "clock_hz") != 1000000000ULL)
    throw std::invalid_argument("This target uses one 1 GHz clock domain");
  HardwareConfig h;
  h.scratchpad_bytes = unsigned_field(j, "scratchpad_bytes");
  h.dma_bytes_per_cycle = unsigned_field(j, "dma_bytes_per_cycle");
  h.dma_setup = unsigned_field(j, "dma_setup");
  h.dram_read_start = unsigned_field(j, "dram_read_start");
  h.dram_write_start = unsigned_field(j, "dram_write_start");
  h.array_input_bytes_per_cycle = {unsigned_field(j,"array16_input_bytes_per_cycle"),
                                   unsigned_field(j,"array32_input_bytes_per_cycle")};
  h.array_launch = unsigned_field(j, "array_launch");
  h.array_output_bytes_per_cycle = unsigned_field(j, "array_output_bytes_per_cycle");
  h.vector_launch = unsigned_field(j, "vector_launch");
  h.vector_bytes_per_cycle = unsigned_field(j, "vector_bytes_per_cycle");
  return h;
}

// Retain ONNXim's actual FIFO admission, scheduled LanguageModel generation,
// cache progression and retirement. Jump over an idle interval or timed block
// instead of calling cycle() once per modeled nanosecond.
class TraceScheduler final : public LangScheduler {
public:
  using LangScheduler::LangScheduler;
  std::unique_ptr<Model> issue() {
    if (_active_requests.empty() && !_request_queue.empty()) {
      const auto arrival = _request_queue.front()->request_time;
      if (arrival > _cycle) _cycle = arrival - 1;
    }
    cycle();
    if (!can_schedule_model()) throw std::logic_error("ONNXim failed to produce a scheduled batch");
    return pop_model();
  }
  void retire(uint32_t id, Cycle elapsed) {
    if (elapsed > std::numeric_limits<Cycle>::max() - _cycle) throw std::overflow_error("simulation clock overflow");
    _cycle += elapsed;
    finish_model(id);
  }
  Cycle now() const { return _cycle; }
};

uint64_t visible(const LangInput& r) {
  // Upstream records the next generated position during retirement. In a
  // decode iteration current_length therefore already includes that position.
  return uint64_t(r.context_length) + (r.cached_decode ? 0 : r.seq_length);
}

json phase(const PhaseMetrics& p) {
  return {{"cycles",p.cycles}, {"dram_read_bytes",p.read_bytes}, {"dram_write_bytes",p.write_bytes},
          {"peak_sram_bytes",p.peak_sram}, {"array16_tiles",p.small_tiles}, {"array32_tiles",p.large_tiles},
          {"array16_compute_cycles",p.small_compute_cycles}, {"array32_compute_cycles",p.large_compute_cycles},
          {"dma_busy_cycles",p.dma_busy_cycles}};
}

}  // namespace

void run_trace(const std::string& model_path, const std::string& hardware_path,
               const std::string& trace_path, const std::string& attention,
               unsigned batch_limit, std::ostream& result) {
  AttentionSchedule schedule;
  if (attention == "naive") schedule = AttentionSchedule::Naive;
  else if (attention == "flashattention" || attention == "streaming") schedule = AttentionSchedule::Streaming;
  else throw std::invalid_argument("attention must be naive, flashattention, or streaming");
  if (batch_limit == 0 || batch_limit > 65535) throw std::invalid_argument("batch limit must be 1..65535");
  const auto hardware_json = load_json(hardware_path);
  const auto hw = hardware_config(hardware_json);
  const auto dram_capacity = unsigned_field(hardware_json, "dram_bytes");
  const auto m = load_json(model_path);
  if (m.at("hidden_size") != 4096 || m.at("num_hidden_layers") != 32 ||
      m.at("num_attention_heads") != 32 || m.at("num_kv_heads") != 8 ||
      m.at("intermediate_size") != 14336 || m.at("vocab_size") != 128256 ||
      m.at("ffn_type") != "llama" || !m.at("run_single_layer").get<bool>() ||
      m.at("tensor_parallel_size") != 1 || m.at("pipeline_parallel_size") != 1)
    throw std::invalid_argument("Target supports the single-layer Llama 3 8B preset on one device");
  SimulationConfig config{};
  config.precision = 2;  // The outer scheduler's KV tensor storage is 16 bit.
  config.core_freq = 1000;
  config.dram_req_size = 64;
  const auto max_sequence = unsigned_field(m, "max_seq_length");
  if (!max_sequence || max_sequence > 131072) throw std::invalid_argument("max_seq_length must be 1..131072");
  if (!std::ifstream(trace_path)) throw std::runtime_error("Cannot open " + trace_path);
  auto model = std::make_unique<LanguageModel>(m,config,"llama3-8b");
  TraceScheduler scheduler("llama3-8b",trace_path,std::move(model),config,
                           {{"max_batch_size",batch_limit}});
  if (!scheduler.busy()) throw std::invalid_argument("Trace contains no requests");
  while (scheduler.busy()) {
    auto instance = scheduler.issue();
    const auto& requests = dynamic_cast<const LanguageModel&>(*instance).requests();
    const auto& first = requests.at(0);
    const auto context = visible(first);
    if (context > max_sequence || context == 0) throw std::invalid_argument("Context exceeds model limits");
    for (const auto& r : requests) {
      if (r.seq_length != first.seq_length || visible(r) != context || r.cached_decode != first.cached_decode)
        throw std::invalid_argument("Layer probe requires homogeneous synchronous batches");
    }
    LayerShape shape{static_cast<unsigned>(requests.size()), first.seq_length, static_cast<unsigned>(context)};
    // Resident W8 matrices including untied embedding/head, 32-layer A16 KV,
    // and a conservative full-layer workspace allowance (FFN, scores, hidden).
    const uint64_t tokens = uint64_t(shape.batch)*shape.queries;
    const uint64_t matrices = 2ULL*128256*4096 + 32ULL*(2ULL*4096*4096 + 2ULL*4096*1024 + 3ULL*4096*14336);
    const uint64_t cache = uint64_t(shape.batch)*shape.context*32*2*1024*2;
    const uint64_t workspace = tokens*(6*14336 + 10*4096) + uint64_t(32)*4*shape.context*4;
    if (matrices+cache+workspace > dram_capacity) throw std::invalid_argument("target DRAM capacity exceeded for this resident layout");
    const auto metrics = run_decoder_layer(shape,schedule,hw);
    result << json{{"target","arago-dual-array"},{"scope","one_decoder_layer"},{"schedule",attention},
                   {"phase",first.cached_decode ? "decode":"prefill"},{"batch",shape.batch},
                   {"queries",shape.queries},{"visible_context",shape.context},{"start_cycle",scheduler.now()},
                   {"layer",phase(metrics.layer)},{"attention",phase(metrics.attention)}}.dump() << '\n';
    scheduler.retire(instance->get_id(),metrics.layer.cycles);
  }
}
}  // namespace arago
