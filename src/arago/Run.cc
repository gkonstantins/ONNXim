#include "Run.h"
#include "Region.h"
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

}  // namespace

void run_trace(const std::string& model_path, const std::string& hardware_path,
               const std::string& trace_path, const std::string& attention,
               unsigned batch_limit, std::ostream& result) {
  if (batch_limit == 0 || batch_limit > 65535) throw std::invalid_argument("batch limit must be 1..65535");
  const auto hardware = load_json(hardware_path);
  const Region region(hardware, attention);
  const auto m = load_json(model_path);
  if (m.at("hidden_size") != 4096 || m.at("num_hidden_layers") != 32 ||
      m.at("num_attention_heads") != 32 || m.at("num_kv_heads") != 8 ||
      m.at("intermediate_size") != 14336 || m.at("vocab_size") != 128256 ||
      m.at("ffn_type") != "llama" || !m.at("run_single_layer").get<bool>() ||
      m.at("tensor_parallel_size") != 1 || m.at("pipeline_parallel_size") != 1)
    throw std::invalid_argument("Target supports the single-layer Llama 3 8B preset on one device");
  SimulationConfig config{};
  config.precision = 2;  // The outer scheduler's KV tensor storage is 16 bit.
  config.core_freq = std::max<uint64_t>(1, unsigned_field(hardware,"clock_hz") / 1000000);
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
    auto record = region.run(shape);
    record["phase"] = first.cached_decode ? "decode" : "prefill";
    record["start_cycle"] = scheduler.now();
    result << record.dump() << '\n';
    scheduler.retire(instance->get_id(),record.at("layer").at("cycles").get<Cycle>());
  }
}
}  // namespace arago
