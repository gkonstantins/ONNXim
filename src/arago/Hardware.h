#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace arago {

using Cycle = std::uint64_t;
enum class Array { Small, Large };

// Rates are payload bytes per target cycle. Read and write DMA
// descriptors use the same resource. The two DMA links pipeline a payload.
// Per invocation. Number each independently simulated primitive in schedule order.
struct TraceProgram {
  int selected = -1;
  int next = 0;
};

struct HardwareConfig {
  std::uint64_t clock_hz = 1000000000;
  unsigned array_input_bits = 8, array_output_bits = 16;
  unsigned input_bytes() const { return array_input_bits / 8; }
  unsigned output_bytes() const { return array_output_bits / 8; }
  TraceProgram* program = nullptr;
  std::uint64_t scratchpad_bytes = 1ULL << 20;
  std::uint64_t dma_bytes_per_cycle = 64;
  std::uint64_t sram_dma_rate = UINT64_MAX;
  bool queued_delivery = false, duplex_dma = false, overlap_drain = true;
  bool wavefront_fill = false, transpose_decode = false, double_buffer = true;
  bool share_kv = true;
  unsigned mac_percent = 100;
  unsigned pipeline_stages = 1, tensor_shards = 1, query_tile = 32, key_tile = 256;
  bool mlp_pingpong = false;
  bool projection_pingpong = false;
  bool optimized_decode = false;
  bool hybrid_kv = false;  // Fetch 256 keys, reduce in 128-key panels.
  bool trace_engines = false;
  bool trace_panels = false, exact_digits = false, exact_a16 = false, model_endpoints = false;
  Cycle dma_setup = 200;
  Cycle dram_read_start = 200;
  Cycle dram_write_start = 150;
  std::array<std::uint64_t, 2> array_input_bytes_per_cycle = {64, 96};
  std::uint64_t array_output_bytes_per_cycle = 8;
  Cycle array_launch = 100;
  std::uint64_t vector_bytes_per_cycle = 128;
  Cycle vector_launch = 300;
};

enum class Unit { Dma, Array16, Array32, Vector };
enum class Action { Read, Write, Launch, Compute, Drain, Vector };
struct Event {
  Unit unit;
  Action action;
  Cycle ready;
  Cycle start;
  Cycle end;
  std::uint64_t bytes;
  const char* operation = nullptr;
  int kv_block = -1;
  Cycle posted = 0, delivered = 0, service = 0; // DMA or SIMD phase boundaries
};

// The caller issues a FIFO per resource and supplies cross-resource
// dependencies via ready. An earlier reservation on a resource retains its
// place even when a later command is ready sooner. Timestamps are half-open.
class Hardware {
 public:
  explicit Hardware(HardwareConfig config = {}, std::size_t trace_limit = 0);
  Cycle read(std::uint64_t bytes, Cycle ready = 0);
  Cycle write(std::uint64_t bytes, Cycle ready = 0);
  Cycle gemm(Array engine, unsigned k, Cycle ready = 0);
  // Independent equal-depth tiles with all operands ready at the indicated
  // cycle. Load/pack/accumulate/write commands and buffers remain explicit.
  Cycle gemm_repeat(Array engine, unsigned k, std::uint64_t count, Cycle ready = 0,
                    Cycle posted = UINT64_MAX);
  // Sum of bytes transferred in the two directions on the shared SIMD port.
  Cycle vector(std::uint64_t port_bytes, Cycle ready = 0);

  // Reserve storage for the current schedule. Release only after its last
  // reader, or propagate the last-reader cycle to the next use of that slot.
  void reserve(std::uint64_t bytes);
  void release(std::uint64_t bytes);
  Cycle finish() const;
  std::uint64_t read_bytes() const { return read_bytes_; }
  std::uint64_t write_bytes() const { return write_bytes_; }
  void trace_label(const char* operation, int kv_block = -1) {
    trace_operation_ = operation;
    trace_kv_block_ = kv_block;
  }
  Cycle dma_busy_cycles() const;
  std::uint64_t read_descriptors() const { return reads_; }
  std::uint64_t write_descriptors() const { return writes_; }
  std::uint64_t vector_commands() const { return vector_commands_; }
  std::uint64_t vector_bytes() const { return vector_bytes_; }
  std::uint64_t peak_sram() const { return peak_sram_; }
  std::uint64_t array_tiles(Array engine) const;
  Cycle array_compute_cycles(Array engine) const;
  int primitive_id() const { return primitive_id_; }
  const std::vector<Event>& events() const { return events_; }

 private:
  struct Pipeline {
    Cycle delivery_free = 0;
    Cycle core_free = 0;
    Cycle drain_free = 0;
    Cycle active = 0;
    std::uint64_t tiles = 0;
  };
  void record(Unit unit, Action action, Cycle ready, Cycle start, Cycle end,
              std::uint64_t bytes, Cycle posted = 0, Cycle delivered = 0, Cycle service = 0);
  Cycle dma(std::uint64_t bytes, Cycle ready, bool is_write);
  HardwareConfig config_;
  std::size_t trace_limit_;
  int primitive_id_ = -1;
  const char* trace_operation_ = nullptr;
  int trace_kv_block_ = -1;
  std::vector<Event> events_;
  std::array<Pipeline, 2> pipelines_;
  Cycle dma_free_ = 0;
  Cycle dma_busy_ = 0;
  Cycle write_free_ = 0, read_delivery_ = 0, write_delivery_ = 0;
  std::vector<std::array<Cycle, 2>> dma_intervals_;
  Cycle vector_delivery_ = 0;
  Cycle vector_free_ = 0;
  std::uint64_t read_bytes_ = 0;
  std::uint64_t write_bytes_ = 0;
  std::uint64_t reads_ = 0;
  std::uint64_t writes_ = 0;
  std::uint64_t vector_commands_ = 0;
  std::uint64_t vector_bytes_ = 0;
  std::uint64_t live_sram_ = 0;
  std::uint64_t peak_sram_ = 0;
};

}  // namespace arago
