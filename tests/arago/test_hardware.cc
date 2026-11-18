#include "Hardware.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace arago;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void expect_event(const Event& event, Unit unit, Action action, Cycle start,
                  Cycle end, std::uint64_t bytes) {
  require(event.unit == unit && event.action == action && event.start == start &&
          event.end == end && event.bytes == bytes, "incorrect timing event");
}

template<class E, class Fn> void expect_error(Fn fn) {
  try {
    fn();
  } catch (const E&) {
    return;
  }
  throw std::runtime_error("expected model validation failure");
}

void cold_score() {
  Hardware hw({}, 16);
  // 32 queries, 16 keys, head width 128, native INT8 panels in DRAM.
  hw.reserve(4096 + 2048 + 1024);
  const auto q = hw.read(4096);
  const auto k = hw.read(2048);
  const auto done = hw.gemm(Array::Large, 128, std::max(q, k));
  const auto& e = hw.events();
  require(e.size() == 5, "wrong number of cold tile events");
  expect_event(e[0], Unit::Dma, Action::Read, 0, 464, 4096);
  expect_event(e[1], Unit::Dma, Action::Read, 464, 896, 2048);
  expect_event(e[2], Unit::Array32, Action::Launch, 896, 996, 0);
  expect_event(e[3], Unit::Array32, Action::Compute, 996, 1124, 6144);
  expect_event(e[4], Unit::Array32, Action::Drain, 1124, 1252, 1024);
  require(done == 1252 && hw.finish() == done && hw.read_bytes() == 6144 &&
          hw.read_descriptors() == 2 && hw.write_bytes() == 0 &&
          hw.peak_sram() == 7168, "unexpected cold-tile counters");
  require(hw.dma_busy_cycles() == 896 && hw.array_compute_cycles(Array::Large) == 128 &&
          hw.array_compute_cycles(Array::Small) == 0, "cold tile resource occupancy");
  hw.release(7168);
  std::cout << "cold tile: Q=4096 B, K=2048 B; DMA ready 896, output drain "
               "[1124,1252), scores ready cycle " << done << "\n";
}

void dual_array() {
  Hardware hw({}, 24);
  // 48 query positions share 16 keys: assign 16 rows to one array and 32
  // rows to the other. Both Q descriptors are posted at cycle zero. The same
  // final K read is a true data dependency of both tiles.
  hw.reserve(2048 + 4096 + 2048 + 512 + 1024);
  const auto q16 = hw.read(2048);
  const auto q32 = hw.read(4096);
  const auto keys = hw.read(2048);
  const auto small = hw.gemm(Array::Small, 128, std::max(q16, keys));
  const auto large = hw.gemm(Array::Large, 128, std::max(q32, keys));
  const auto& e = hw.events();
  require(e.size() == 9, "wrong number of dual array events");
  require(e[0].ready == 0 && e[1].ready == 0 && e[2].ready == 0,
          "input DMA requests must contend at the same ready cycle");
  expect_event(e[0], Unit::Dma, Action::Read, 0, 432, 2048);
  expect_event(e[1], Unit::Dma, Action::Read, 432, 896, 4096);
  expect_event(e[2], Unit::Dma, Action::Read, 896, 1328, 2048);
  expect_event(e[4], Unit::Array16, Action::Compute, 1428, 1556, 4096);
  expect_event(e[7], Unit::Array32, Action::Compute, 1428, 1556, 6144);
  expect_event(e[5], Unit::Array16, Action::Drain, 1556, 1620, 512);
  expect_event(e[8], Unit::Array32, Action::Drain, 1556, 1684, 1024);
  require(small == 1620 && large == 1684 && hw.finish() == large &&
          hw.read_bytes() == 8192 && hw.read_descriptors() == 3,
          "incorrect resource sharing");
  require(hw.dma_busy_cycles() == 1328 && hw.array_compute_cycles(Array::Small) == 128 &&
          hw.array_compute_cycles(Array::Large) == 128, "dual array overlap occupancy");
  // A delayed transfer must leave the preceding idle gap out of the counter.
  hw.write(1, 2000);
  require(hw.dma_busy_cycles() == 1328 + 351, "DMA idle interval counted as busy");
  std::cout << "shared DMA: Q16 [0,432), Q32 [432,896), shared K [896,1328); "
               "both arrays compute [1428,1556), outputs ready " << small
            << " and " << large << "\n";
}

void same_array_readout_overlap() {
  // For a short K the large array produces new tiles faster than its 8 B/cycle
  // readout can drain them. A waiting result blocks the core from accepting a
  // third command until the second result can start draining.
  Hardware hw({}, 12);
  require(hw.gemm_repeat(Array::Large, 1, 3) == 485, "incorrect drain bottleneck");
  const auto& e = hw.events();
  expect_event(e[2], Unit::Array32, Action::Drain, 101, 229, 1024);
  expect_event(e[4], Unit::Array32, Action::Compute, 201, 202, 48);
  expect_event(e[5], Unit::Array32, Action::Drain, 229, 357, 1024);
  expect_event(e[6], Unit::Array32, Action::Launch, 229, 329, 0);
  expect_event(e[8], Unit::Array32, Action::Drain, 357, 485, 1024);
}

void repeated_pipeline() {
  // Compare batched and individual tiles across pipeline fill, a change of
  // depth, output bottlenecks, incoming stalls, and a full trace buffer.
  for (Array array : {Array::Small, Array::Large}) {
    for (unsigned k : {1u, 128u, 256u}) {
      HardwareConfig cfg;
      cfg.array_output_bytes_per_cycle = 4;
      Hardware compact(cfg, 8), individual(cfg, 8);
      compact.gemm(array, 1, 17);
      individual.gemm(array, 1, 17);
      Cycle last = 0;
      for (int i = 0; i < 25; ++i) last = individual.gemm(array, k, 18);
      require(compact.gemm_repeat(array, k, 25, 18) == last, "batched GEMM timing mismatch");
      require(compact.array_compute_cycles(array) == individual.array_compute_cycles(array) &&
              compact.array_tiles(array) == 26 && compact.events().size() == 8,
              "batched GEMM counters or trace mismatch");
      for (unsigned i = 0; i < compact.events().size(); ++i) {
        const auto& e = individual.events()[i];
        expect_event(compact.events()[i], e.unit, e.action, e.start, e.end, e.bytes);
      }
      require(compact.gemm(array, k, last + 50) == individual.gemm(array, k, last + 50),
              "pipeline state diverged after repeated tiles");
    }
  }
}

void miscellaneous() {
  Hardware hw;
  require(hw.read(1) == 401 && hw.write(65) == 753, "incorrect shared DMA direction or rounding");
  require(hw.vector(129, 700) == 1002 && hw.finish() == 1002 && hw.events().empty(),
          "SIMD port or default trace behavior incorrect");
  hw.reserve(1u << 20);
  expect_error<std::length_error>([&] { hw.reserve(1); });
  hw.release(1u << 20);
  expect_error<std::logic_error>([&] { hw.release(1); });
  expect_error<std::invalid_argument>([&] { hw.gemm(Array::Small, 257); });
  expect_error<std::invalid_argument>([&] { hw.gemm(Array::Small, 0); });
  HardwareConfig narrow;
  narrow.array_input_bytes_per_cycle[1] = 1;
  Hardware limited(narrow);
  require(limited.gemm(Array::Large, 128) == 100 + 6144 + 128, "array ingress bandwidth ignored");
}

void transport_variants() {
  HardwareConfig cfg;
  cfg.queued_delivery = true;
  Hardware device(cfg);
  require(device.read(4096)==464 && device.read(2048)==696,"queued shared DMA");
  require(device.gemm_repeat(Array::Large,128,1,696,0)==952,"prefetched command must wait for K");
  for (bool queued : {false,true}) {
    for (bool overlap : {false,true}) {
      for (unsigned depth : {1u,128u,256u}) {
        cfg.queued_delivery=queued; cfg.overlap_drain=overlap;
        Hardware batched(cfg), single(cfg);
        Cycle last=0;
        for (int i=0;i<9;++i) last=single.gemm(Array::Large,depth,42);
        require(batched.gemm_repeat(Array::Large,depth,9,42)==last,"queue/drain repeated replay");
      }
    }
  }
  cfg = HardwareConfig{};
  cfg.duplex_dma=true;
  Hardware duplex(cfg);
  require(duplex.read(1024)==416 && duplex.write(1024)==366,"duplex dependency-free service");
  require(duplex.finish()==416 && duplex.dma_busy_cycles()==416,"duplex utilization union");
  for (unsigned slower : {32u,64u,128u}) {
    cfg.sram_dma_rate=slower;
    Hardware link(cfg);
    require(link.read(4096)==400+4096/std::min(slower,64u),"series link bottleneck");
  }
}
}  // namespace

int main() {
  try {
    cold_score();
    dual_array();
    same_array_readout_overlap();
    repeated_pipeline();
    miscellaneous();
    transport_variants();
    std::cout << "all hardware checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
