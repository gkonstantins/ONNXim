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
  if (!config_.scratchpad_bytes || !config_.dma_bytes_per_cycle ||
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
  const Cycle start = std::max(dma_free_, ready);
  const Cycle setup = add(config_.dma_setup,
                          is_write ? config_.dram_write_start : config_.dram_read_start);
  const Cycle end = add(add(start, setup), ceil_div(bytes, config_.dma_bytes_per_cycle));
  if (is_write) {
    write_bytes_ = add(write_bytes_, bytes);
    ++writes_;
  } else {
    read_bytes_ = add(read_bytes_, bytes);
    ++reads_;
  }
  dma_busy_ = add(dma_busy_, end - start);
  dma_free_ = end;
  record(Unit::Dma, is_write ? Action::Write : Action::Read, ready, start, end, bytes);
  return end;
}

Cycle Hardware::read(std::uint64_t bytes, Cycle ready) {
  return dma(bytes, ready, false);
}

Cycle Hardware::write(std::uint64_t bytes, Cycle ready) {
  return dma(bytes, ready, true);
}

Cycle Hardware::vector(std::uint64_t port_bytes, Cycle ready) {
  if (port_bytes == 0) return ready;
  const Cycle start = std::max(vector_free_, ready);
  const Cycle service = ceil_div(port_bytes, config_.vector_bytes_per_cycle);
  const Cycle end = add(add(start, config_.vector_launch), service);
  vector_bytes_ = add(vector_bytes_, port_bytes);
  vector_free_ = end;
  record(Unit::Vector, Action::Vector, ready, start, end, port_bytes);
  return end;
}

Cycle Hardware::gemm(Array engine, unsigned k, Cycle ready) {
  return gemm_repeat(engine, k, 1, ready);
}

Cycle Hardware::gemm_repeat(Array engine, unsigned k, std::uint64_t count, Cycle ready) {
  if (!k || k > 256) throw std::invalid_argument("native GEMM requires 1 <= K <= 256");
  if (count == 0) return ready;
  const std::size_t id = index(engine);
  auto& p = pipelines_[id];
  const Unit unit = engine == Array::Small ? Unit::Array16 : Unit::Array32;
  const std::uint64_t rows = engine == Array::Small ? 16 : 32;
  constexpr unsigned cols = 16;
  const std::uint64_t input_bytes = (rows + cols) * k;       // native INT8
  const std::uint64_t output_bytes = rows * cols * 2;        // INT16 partial
  const Cycle service = std::max<Cycle>(k, ceil_div(input_bytes, config_.array_input_bytes_per_cycle[id]));
  const Cycle issue = add(config_.array_launch, service);
  const Cycle drain = ceil_div(output_bytes, config_.array_output_bytes_per_cycle);

  const Cycle start = std::max(ready, p.core_free);
  const Cycle first_drain = std::max(add(start, issue), p.drain_free);
  // The core releases a tile as soon as readout can accept it. Thus a full
  // readout port backpressures the core; the two stages may otherwise overlap.
  const Cycle stride = std::max(issue, drain);
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
      launch = unload;
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
  return std::max({dma_free_, vector_free_, pipelines_[0].drain_free, pipelines_[1].drain_free});
}

std::uint64_t Hardware::array_tiles(Array engine) const { return pipelines_[index(engine)].tiles; }
Cycle Hardware::array_compute_cycles(Array engine) const { return pipelines_[index(engine)].active; }

}  // namespace arago
