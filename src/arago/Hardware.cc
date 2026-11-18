#include "Hardware.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace arago {
namespace {

std::uint64_t ceil_div(std::uint64_t bytes, std::uint64_t rate) {
  return bytes / rate + (bytes % rate != 0);
}

std::uint64_t add(std::uint64_t lhs, std::uint64_t rhs) {
  if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs)
    throw std::overflow_error("timing or activity counter exceeded uint64");
  return lhs + rhs;
}

std::uint64_t times(std::uint64_t lhs, std::uint64_t rhs) {
  if (rhs && lhs > std::numeric_limits<std::uint64_t>::max() / rhs)
    throw std::overflow_error("timing or activity counter exceeded uint64");
  return lhs * rhs;
}

std::size_t index(Array engine) {
  switch (engine) {
    case Array::Small: return 0;
    case Array::Large: return 1;
  }
  throw std::invalid_argument("unknown systolic array");
}

}  // namespace

Hardware::Hardware(HardwareConfig config, std::size_t trace_limit)
    : config_(config), trace_limit_(trace_limit) {
  if (!config_.mac_percent || !config_.scratchpad_bytes || !config_.dma_bytes_per_cycle || !config_.sram_dma_rate ||
      !config_.array_input_bytes_per_cycle[0] || !config_.array_input_bytes_per_cycle[1] ||
      !config_.array_output_bytes_per_cycle || !config_.vector_bytes_per_cycle)
    throw std::invalid_argument("scratchpad capacity and link rates must be positive");
}

void Hardware::record(Unit unit, Action action, Cycle ready, Cycle start,
                      Cycle end, std::uint64_t bytes) {
  if (events_.size() < trace_limit_)
    events_.push_back({unit, action, ready, start, end, bytes});
}

Cycle Hardware::dma(std::uint64_t bytes, Cycle ready, bool is_write) {
  if (bytes == 0) return ready;
  auto& free = config_.duplex_dma && is_write ? write_free_ : dma_free_;
  auto& delivery = config_.duplex_dma && is_write ? write_delivery_ : read_delivery_;
  const Cycle posted = std::max(ready, config_.queued_delivery ? delivery : free);
  delivery = add(posted, config_.dma_setup);
  const Cycle start = std::max(free, posted);
  const Cycle first = add(std::max(free, delivery),
                          is_write ? config_.dram_write_start : config_.dram_read_start);
  const auto rate = std::min(config_.dma_bytes_per_cycle, config_.sram_dma_rate);
  const Cycle end = add(first, ceil_div(bytes, rate));
  if (is_write) {
    write_bytes_ = add(write_bytes_, bytes);
    ++writes_;
  } else {
    read_bytes_ = add(read_bytes_, bytes);
    ++reads_;
  }
  dma_busy_ = add(dma_busy_, end - start);
  free = end;
  if (config_.duplex_dma) dma_intervals_.push_back({start, end});
  record(Unit::Dma, is_write ? Action::Write : Action::Read, ready, start, end, bytes);
  return end;
}

Cycle Hardware::dma_busy_cycles() const {
  if (!config_.duplex_dma) return dma_busy_;
  auto intervals = dma_intervals_;
  std::sort(intervals.begin(), intervals.end());
  Cycle last = 0, occupied = 0;
  for (auto interval : intervals) {
    if (interval[1] > last) {
      occupied += interval[1] - std::max(last, interval[0]);
      last = interval[1];
    }
  }
  return occupied;
}

Cycle Hardware::read(std::uint64_t bytes, Cycle ready) {
  return dma(bytes, ready, false);
}

Cycle Hardware::write(std::uint64_t bytes, Cycle ready) {
  return dma(bytes, ready, true);
}

Cycle Hardware::vector(std::uint64_t port_bytes, Cycle ready) {
  if (port_bytes == 0) return ready;
  const Cycle start = std::max(config_.queued_delivery ? vector_delivery_ : vector_free_, ready);
  vector_delivery_ = add(start, config_.vector_launch);
  const Cycle service = ceil_div(port_bytes, config_.vector_bytes_per_cycle);
  const Cycle end = add(std::max(vector_free_, vector_delivery_), service);
  vector_bytes_ = add(vector_bytes_, port_bytes);
  vector_free_ = end;
  record(Unit::Vector, Action::Vector, ready, start, end, port_bytes);
  return end;
}

Cycle Hardware::gemm(Array engine, unsigned k, Cycle ready) {
  return gemm_repeat(engine, k, 1, ready);
}

Cycle Hardware::gemm_repeat(Array engine, unsigned k, std::uint64_t count, Cycle ready, Cycle posted) {
  if (!k || k > 256) throw std::invalid_argument("native GEMM requires 1 <= K <= 256");
  if (count == 0) return ready;
  const std::size_t id = index(engine);
  auto& p = pipelines_[id];
  const Unit unit = engine == Array::Small ? Unit::Array16 : Unit::Array32;
  const std::uint64_t rows = engine == Array::Small ? 16 : 32;
  constexpr unsigned cols = 16;
  const std::uint64_t input_bytes = (rows + cols) * k;       // native INT8
  const std::uint64_t output_bytes = rows * cols * 2;        // INT16 partial
  const Cycle service = std::max<Cycle>(ceil_div(k*100,config_.mac_percent) + (config_.wavefront_fill ? rows + cols - 2 : 0), ceil_div(input_bytes, config_.array_input_bytes_per_cycle[id]));
  const Cycle issue = add(config_.array_launch, service);
  const Cycle drain = ceil_div(output_bytes, config_.array_output_bytes_per_cycle);

  if (config_.queued_delivery) {
    // Arrival is independent of operand and engine readiness. A blocked
    // readout port holds the previous result at the core output.
    for (std::uint64_t tile = 0; tile < count; ++tile) {
      const auto post = std::max(posted == UINT64_MAX ? ready : posted, p.delivery_free);
      p.delivery_free = add(post, config_.array_launch);
      const auto compute = std::max({ready, p.delivery_free,
                        config_.overlap_drain ? p.core_free : p.drain_free});
      const auto produced = add(compute, service);
      p.core_free = std::max(produced, p.drain_free);
      p.drain_free = add(p.core_free, drain);
      record(unit, Action::Launch, post, post, p.delivery_free, 0);
      record(unit, Action::Compute, ready, compute, produced, input_bytes);
      record(unit, Action::Drain, produced, p.core_free, p.drain_free, output_bytes);
    }
    p.active = add(p.active,times(count,service));
    p.tiles = add(p.tiles,count);
    return p.drain_free;
  }
  const Cycle start = std::max(ready,config_.overlap_drain ? p.core_free : p.drain_free);
  const Cycle first_drain = std::max(add(start,issue),p.drain_free);
  const Cycle stride = config_.overlap_drain ? std::max(issue,drain) : add(issue,drain);
  const Cycle last_drain = add(first_drain, times(count - 1, stride));
  const Cycle end = add(last_drain, drain);

  if (events_.size() < trace_limit_) {
    Cycle launch = start;
    Cycle unload = first_drain;
    for (std::uint64_t tile = 0; tile < count && events_.size() < trace_limit_; ++tile) {
      const Cycle compute = add(launch, config_.array_launch);
      const Cycle produced = add(compute, service);
      record(unit, Action::Launch, ready, launch, compute, 0);
      record(unit, Action::Compute, compute, compute, produced, input_bytes);
      record(unit, Action::Drain, produced, unload, add(unload, drain), output_bytes);
      launch = config_.overlap_drain ? unload : add(unload, drain);
      if (tile + 1 < count) unload = add(unload, stride);
    }
  }
  p.active = add(p.active, times(count, service));
  p.tiles = add(p.tiles, count);
  p.core_free = last_drain;
  p.drain_free = end;
  return end;
}

void Hardware::reserve(std::uint64_t bytes) {
  if (bytes > config_.scratchpad_bytes - live_sram_)
    throw std::length_error("scratchpad capacity exceeded");
  live_sram_ += bytes;
  peak_sram_ = std::max(peak_sram_, live_sram_);
}

void Hardware::release(std::uint64_t bytes) {
  if (bytes > live_sram_) throw std::logic_error("scratchpad release exceeds reservation");
  live_sram_ -= bytes;
}

Cycle Hardware::finish() const {
  return std::max({dma_free_, write_free_, vector_free_, pipelines_[0].drain_free, pipelines_[1].drain_free});
}

std::uint64_t Hardware::array_tiles(Array engine) const { return pipelines_[index(engine)].tiles; }
Cycle Hardware::array_compute_cycles(Array engine) const { return pipelines_[index(engine)].active; }

}  // namespace arago
