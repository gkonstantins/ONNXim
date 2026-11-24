#include "arago/DecoderLayer.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
using namespace arago;
using Count = std::uint64_t;

void check(bool pass, const char* message) {
  if (!pass) throw std::runtime_error(message);
}

void print(const char* workload, const char* algorithm, const char* phase,
           const PhaseMetrics& m) {
  std::cout << workload << ',' << algorithm << ',' << phase << ',' << m.cycles
            << ',' << m.read_bytes << ',' << m.write_bytes << ',' << m.peak_sram
            << ',' << m.small_tiles << ',' << m.large_tiles
            << ',' << m.small_compute_cycles << ',' << m.large_compute_cycles << ',' << m.dma_busy_cycles << '\n';
}

void compare(const char* workload, const LayerShape& shape, unsigned query_tile = 32) {
  HardwareConfig config;
  config.query_tile = query_tile;
  const auto naive = run_decoder_layer(shape, AttentionSchedule::Naive, config);
  const auto stream = run_decoder_layer(shape, AttentionSchedule::Streaming, config);

  // Independent traffic oracle: one Q read / output write per query head.
  // Each GQA group loads K and V once for the prefix visible to that row block.
  // The naive schedule also writes and reads every 16-bit score and probability.
  Count groups = 8ULL * shape.batch;
  Count q_io = 2ULL * shape.batch * 32 * shape.queries * 128;
  Count kv_reads = 0, extra_each_direction = 0;
  for (unsigned first = 0; first < shape.queries; first += query_tile) {
    const Count nr = std::min<Count>(query_tile, shape.queries-first);
    const Count keys = shape.context-shape.queries+first+nr;
    kv_reads += groups * 2 * keys * 128 * 2;
    extra_each_direction += groups * 4 * nr * keys * 2 * 2;
  }
  check(stream.attention.read_bytes == q_io + kv_reads, "streamed Q/K/V reads");
  check(stream.attention.write_bytes == q_io, "streamed final outputs");
  check(naive.attention.read_bytes == q_io + kv_reads + extra_each_direction, "naive reads");
  check(naive.attention.write_bytes == q_io + extra_each_direction, "naive writes");
  check(naive.attention.small_tiles == stream.attention.small_tiles &&
        naive.attention.large_tiles == stream.attention.large_tiles, "unequal native work");
  check(naive.layer.cycles - naive.attention.cycles ==
        stream.layer.cycles - stream.attention.cycles, "comparison changed other operations");
  check(naive.layer.read_bytes - naive.attention.read_bytes ==
        stream.layer.read_bytes - stream.attention.read_bytes, "change outside attention");
  check(naive.layer.peak_sram > 0 && naive.layer.peak_sram <= (1u << 20) &&
        stream.layer.peak_sram > 0 && stream.layer.peak_sram <= (1u << 20), "SRAM budget");
  for (const auto* m : {&naive.layer, &stream.layer, &naive.attention, &stream.attention}) {
    check(m->small_compute_cycles <= m->cycles && m->large_compute_cycles <= m->cycles &&
          m->dma_busy_cycles <= m->cycles && m->dma_busy_cycles > 0, "resource occupancy bounds");
  }
  check(naive.layer.small_compute_cycles == stream.layer.small_compute_cycles &&
        naive.layer.large_compute_cycles == stream.layer.large_compute_cycles,
        "schedules must issue equal array service");
  print(workload,"naive","attention",naive.attention);
  print(workload,"naive","layer",naive.layer);
  print(workload,"streaming","attention",stream.attention);
  print(workload,"streaming","layer",stream.layer);
}
}  // namespace

int main() {
  std::cout << "workload,schedule,phase,cycles,read_B,write_B,peak_SRAM_B,16x16_tiles,32x16_tiles,16x16_service,32x16_service,DMA_busy\n";
  for (bool pingpong : {false,true}) {
    HardwareConfig h;
    h.trace_panels = h.optimized_decode = h.overlap_decode = true;
    h.double_buffer = pingpong;
    h.key_tile = 256;
    const auto result = run_decoder_layer({16,1,521}, AttentionSchedule::Streaming,h);
    check(result.tile_plan.size()==3,"full panels and ragged tail");
    Cycle free[2]{};
    for (const auto& b : result.tile_plan) {
      check(b.load_ready>=free[b.slot] && b.load_ready<=b.k_loaded &&
            b.k_loaded<=b.loaded && b.k_loaded<=b.k_packed &&
            b.k_packed<=b.scores_ready && b.loaded<=b.v_packed &&
            b.values_ready>=std::max(b.scores_ready,b.v_packed) &&
            b.released>=b.values_ready,"split K/V consumer or slot hazard");
      free[b.slot]=b.released;
    }
    const auto& first = result.tile_plan[0];
    const auto& next = result.tile_plan[1];
    check(pingpong ? next.k_packed < first.released : next.k_packed >= first.released,
          "K lookahead must respect the number of physical slots");
    check(result.tile_plan.front().k_packed<result.tile_plan.front().loaded,
          "first K panel can be packed before V arrives");
  }
  compare("ragged", {1, 35, 163});
  compare("prefill", {16, 2048, 2048});
  compare("prefill-48", {16, 2048, 2048}, 48);
  compare("decode", {16, 1, 2176});
  // A complete 192-row output slab and an entire GQA request are repeated
  // with no live buffers across repetitions. One versus two copies must have
  // exactly twice the bytes and cycles, even with nondefault port rates.
  arago::HardwareConfig slow;
  slow.dma_bytes_per_cycle = 23;
  slow.array_output_bytes_per_cycle = 5;
  for (auto schedule : {arago::AttentionSchedule::Naive, arago::AttentionSchedule::Streaming}) {
    const auto one = arago::run_decoder_layer({1, 192, 192}, schedule, slow);
    const auto two = arago::run_decoder_layer({2, 192, 192}, schedule, slow);
    check(two.attention.cycles == 2*one.attention.cycles &&
          two.attention.read_bytes == 2*one.attention.read_bytes, "attention replay");
    check(two.layer.cycles == 2*one.layer.cycles &&
          two.layer.read_bytes == 2*one.layer.read_bytes &&
          two.layer.write_bytes == 2*one.layer.write_bytes &&
          two.layer.dma_busy_cycles == 2*one.layer.dma_busy_cycles &&
          two.layer.small_compute_cycles == 2*one.layer.small_compute_cycles &&
          two.layer.large_compute_cycles == 2*one.layer.large_compute_cycles &&
          two.layer.peak_sram == one.layer.peak_sram, "layer slab replay");
  }
  bool rejected = false;
  try { arago::run_decoder_layer({1, 2, 1}, arago::AttentionSchedule::Naive); }
  catch (const std::invalid_argument&) { rejected = true; }
  check(rejected, "invalid cache length accepted");
  arago::HardwareConfig tiny;
  tiny.scratchpad_bytes = 1024;
  rejected = false;
  try { arago::run_decoder_layer({1,1,1}, arago::AttentionSchedule::Streaming, tiny); }
  catch (const std::length_error&) { rejected = true; }
  check(rejected, "insufficient SRAM accepted");
}
