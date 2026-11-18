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
  if (j.contains("sram_dma_rate")) h.sram_dma_rate = unsigned_field(j,"sram_dma_rate");
  h.queued_delivery = j.value("queued_delivery", false);
  h.duplex_dma = j.value("duplex_dma", false);
  h.overlap_drain = j.value("overlap_drain", true);
  h.wavefront_fill = j.value("wavefront_fill", false);
  h.transpose_decode = j.value("transpose_decode", false);
  h.double_buffer = j.value("double_buffer", true);
  h.trace_panels = j.value("trace_panels",false);
  h.share_kv = j.value("share_kv", true);
  h.model_endpoints = j.value("model_endpoints", false);
  const auto policy = j.value("precision_policy", std::string("scaled"));
  if (policy != "scaled" && policy != "fixedshift" && policy != "exact4" && policy != "a16exact" && policy != "bounded" && policy != "saturate")
    throw std::invalid_argument("unknown precision_policy");
  h.exact_a16 = policy == "a16exact";
  h.exact_digits = policy == "exact4" || h.exact_a16;
  for (const char* key : {"pipeline_stages","tensor_shards","query_tile","key_tile","mac_percent"}) {
    if (!j.contains(key)) continue;
    auto value = unsigned_field(j,key);
    if (value == 0 || value > 256) throw std::invalid_argument(key);
    if (std::string(key) == "tensor_shards") h.tensor_shards = value;
    if (std::string(key) == "pipeline_stages") h.pipeline_stages = value;
    if (std::string(key) == "mac_percent") h.mac_percent = value;
    if (std::string(key) == "query_tile") h.query_tile = value;
    if (std::string(key) == "key_tile") h.key_tile = value;
  }
  if (h.tensor_shards != 1 && h.tensor_shards != 2 && h.tensor_shards != 4)
    throw std::invalid_argument("tensor_shards must be 1, 2 or 4");
  if (h.pipeline_stages != 1 && h.pipeline_stages != 2 && h.pipeline_stages != 4)
    throw std::invalid_argument("pipeline_stages must be 1, 2 or 4");
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
  else if (attention == "rowblocked") schedule = AttentionSchedule::RowBlocked;
  else if (attention == "flashattention" || attention == "streaming") schedule = AttentionSchedule::Streaming;
  else throw std::invalid_argument("attention must be naive, rowblocked, flashattention, or streaming");
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
    const uint64_t local_layers = 32/hw.pipeline_stages;
    const uint64_t endpoint_weights = (hw.pipeline_stages == 1 ? 2ULL : 1ULL)*128256*4096;
    const uint64_t projection_weights = 2ULL*4096*4096 + 2ULL*4096*1024 + 3ULL*4096*14336;
    const uint64_t matrices = endpoint_weights + local_layers*
        (projection_weights/hw.tensor_shards + 2ULL*4096*2) + 4096*2;
    const uint64_t cache = uint64_t(shape.batch)*shape.context*local_layers*2*1024*2/hw.tensor_shards;
    const uint64_t workspace = tokens*(6*14336 + 10*4096) + uint64_t(32)*4*shape.context*4;
    if (matrices+cache+workspace > dram_capacity) throw std::invalid_argument("target DRAM capacity exceeded for this resident layout");
    const auto metrics = run_decoder_layer(shape,schedule,hw);
    json record = {{"target","arago-dual-array"},{"scope","one_decoder_layer"},{"schedule",attention},
                   {"phase",first.cached_decode ? "decode":"prefill"},{"batch",shape.batch},
                   {"queries",shape.queries},{"visible_context",shape.context},{"start_cycle",scheduler.now()},
                   {"layer",phase(metrics.layer)},{"attention",phase(metrics.attention)}};
    if (hw.trace_panels) {
      if (schedule != AttentionSchedule::Streaming) throw std::invalid_argument("tile plan requires streaming");
      record["tile_plan"] = json::array();
      for (const auto& e : metrics.tile_plan)
        record["tile_plan"].push_back({{"q0",e.query_offset},{"q",e.queries},{"k0",e.key_offset},
          {"k",e.keys},{"slot",e.slot},{"slot_ready",e.load_ready},{"loaded",e.loaded},
          {"score",e.scores_ready},{"pv",e.values_ready},{"released",e.released}});
    }
    if (hw.model_endpoints) {
      record["resident_estimate_B"] = matrices+cache+workspace;
      Cycle embedding;
      record["endpoints"] = phase(run_endpoints(shape,hw,embedding));
      record["embedding_cycles"] = embedding;
    }
    result << record.dump() << '\n';
    scheduler.retire(instance->get_id(),metrics.layer.cycles);
  }
}
}  // namespace arago
