#pragma once

#include "Hardware.h"
#include <utility>
#include <memory>

namespace arago {

enum class AttentionSchedule { Naive, Streaming, RowBlocked };

// Queries are the last 'queries' positions of a context that includes them.
struct LayerShape {
  unsigned batch;
  unsigned queries;
  unsigned context;
};

struct EngineWindow {
  std::vector<Event> events;
  Cycle cycles;
  std::uint64_t event_count;
};

struct PrimitiveRun {
  int id;
  Cycle cycles;
  std::uint64_t repeats;
};

struct PhaseMetrics {
  Cycle cycles = 0;
  std::uint64_t read_bytes = 0;
  std::uint64_t write_bytes = 0;
  std::uint64_t peak_sram = 0;
  std::uint64_t small_tiles = 0;
  std::uint64_t large_tiles = 0;
  Cycle small_compute_cycles = 0; // native service, including input-port stalls
  Cycle large_compute_cycles = 0;
  std::uint64_t vector_bytes = 0; // total scratchpad traffic at the SIMD port
  Cycle dma_busy_cycles = 0; // descriptor setup, first byte and payload
  std::shared_ptr<const EngineWindow> sample;
  std::vector<PrimitiveRun> program;
  Cycle sample_offset = 0; // first instance of the selected primitive in this phase
};

using OperatorProfile = std::vector<std::pair<const char*, PhaseMetrics>>;

struct BlockEvent {
  unsigned query_offset, queries, key_offset, keys, slot;
  Cycle load_ready, loaded, scores_ready, values_ready, released;
  Cycle k_loaded = 0, k_packed = 0, v_packed = 0;
};

struct LayerMetrics {
  std::vector<BlockEvent> tile_plan; // one representative user/KV group per query panel
  PhaseMetrics layer;
  PhaseMetrics attention;
};

// One Llama 3.1 8B decoder block, from incoming to outgoing hidden states.
// The schedule has explicit barriers between independent output slabs. A
// completed slab can therefore be replayed by multiplying its counters.
LayerMetrics run_decoder_layer(const LayerShape& shape, AttentionSchedule schedule,
                               const HardwareConfig& config = {}, OperatorProfile* profile = nullptr);

PhaseMetrics run_endpoints(const LayerShape& shape, const HardwareConfig& config, Cycle& embedding,
                           OperatorProfile* profile = nullptr);

}  // namespace arago
