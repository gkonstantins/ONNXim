#include "Region.h"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
using json = nlohmann::json;

namespace arago {
namespace {
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
  h.trace_engines = j.contains("engine_window");
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

json phase(const PhaseMetrics& p) {
  return {{"cycles",p.cycles}, {"dram_read_bytes",p.read_bytes}, {"dram_write_bytes",p.write_bytes},
          {"peak_sram_bytes",p.peak_sram}, {"array16_tiles",p.small_tiles}, {"array32_tiles",p.large_tiles},
          {"array16_compute_cycles",p.small_compute_cycles}, {"array32_compute_cycles",p.large_compute_cycles},
          {"dma_busy_cycles",p.dma_busy_cycles}};
}

}  // namespace


Region::Region(const json& config, const std::string& algorithm)
    : config_(config), algorithm_(algorithm), hardware_(hardware_config(config)),
      dram_capacity_(unsigned_field(config,"dram_bytes")) {
  if (algorithm == "naive") schedule_ = AttentionSchedule::Naive;
  else if (algorithm == "rowblocked") schedule_ = AttentionSchedule::RowBlocked;
  else if (algorithm == "flashattention" || algorithm == "streaming") schedule_ = AttentionSchedule::Streaming;
  else throw std::invalid_argument("attention must be naive, rowblocked, flashattention, or streaming");
}

json Region::run(LayerShape shape) const {
  auto hw = hardware_;
  TraceProgram program;
  if (config_.contains("program_primitive")) {
    program.selected = config_.at("program_primitive").get<int>();
    hw.program = &program;
  }
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
  if (matrices+cache+workspace > dram_capacity_) throw std::invalid_argument("target DRAM capacity exceeded for this resident layout");
  OperatorProfile operations;
  auto* profile = config_.value("profile_operators",false) ? &operations : nullptr;
  const auto metrics = run_decoder_layer(shape,schedule_,hw,profile);
  json record = {{"target","arago-dual-array"},{"scope","one_decoder_layer"},{"schedule",algorithm_},
                 {"phase","unspecified"},{"batch",shape.batch},
                 {"queries",shape.queries},{"visible_context",shape.context},
                 {"layer",phase(metrics.layer)},{"attention",phase(metrics.attention)}};
  if (hw.trace_panels) {
    if (schedule_ != AttentionSchedule::Streaming) throw std::invalid_argument("tile plan requires streaming");
    record["tile_plan"] = json::array();
    for (const auto& e : metrics.tile_plan)
      record["tile_plan"].push_back({{"q0",e.query_offset},{"q",e.queries},{"k0",e.key_offset},
        {"k",e.keys},{"slot",e.slot},{"slot_ready",e.load_ready},{"loaded",e.loaded},
        {"score",e.scores_ready},{"pv",e.values_ready},{"released",e.released}});
  }
  if (hw.model_endpoints) {
    record["resident_estimate_B"] = matrices+cache+workspace;
    Cycle embedding;
    record["endpoints"] = phase(run_endpoints(shape,hw,embedding,profile));
    record["embedding_cycles"] = embedding;
  }

  if (profile) {
    for (const auto& [name, counters] : operations) {
      auto entry = phase(counters);
      if (hw.program) {
        entry["program"] = json::array();
        for (const auto& run : counters.program)
          entry["program"].push_back({run.id,run.cycles,run.repeats});
      }
      if (counters.sample && (hw.program || (hw.trace_engines && config_.at("engine_window")==name))) {
        const auto& window = *counters.sample;
        json events=json::array();
        const char* units[]={"DMA","16x16","32x16","SIMD"};
        const char* actions[]={"Read","Write","Launch","Compute","Drain","Vector"};
        for (const auto& e : window.events) {
          events.push_back({{"unit",units[static_cast<unsigned>(e.unit)]},
                            {"action",actions[static_cast<unsigned>(e.action)]},{"ready",e.ready},
                            {"work",e.operation ? e.operation : ""},{"kv_block",e.kv_block},
                            {"start",e.start},{"end",e.end},{"bytes",e.bytes},
                            {"posted",e.posted},{"delivered",e.delivered},{"service",e.service}});
        }
        record["engine_window"]={{"op",name},{"offset",counters.sample_offset},
                                 {"cycles",window.cycles},{"total_events",window.event_count},
                                 {"events",std::move(events)}};
      }
      entry["op"] = name;
      entry["simd_port_bytes"] = counters.vector_bytes;
      record["operations"].push_back(std::move(entry));
    }
  }
  return record;
}
}  // namespace arago
