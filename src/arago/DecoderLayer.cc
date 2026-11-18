#include "DecoderLayer.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace arago {
namespace {

using Count = std::uint64_t;
constexpr unsigned hidden = 4096, head_dim = 128, kv_heads = 8, group = 4;
constexpr unsigned kv_width = kv_heads * head_dim, ffn = 14336;
constexpr unsigned q_block = 32, kv_block = 128;
constexpr Count max_context = 131072;

// Phase boundaries drain all pipes; independent batches/heads/slabs use the
// same workspace consecutively. Peak SRAM is a maximum, never a sum here.
Count add_repeated(Count base, Count increment, Count times) {
  if (increment && times > (std::numeric_limits<Count>::max() - base) / increment) {
    throw std::overflow_error("repeated layer activity exceeds uint64");
  }
  return base + increment * times;
}

void append(PhaseMetrics& total, const PhaseMetrics& p, Count repetitions = 1) {
  total.cycles = add_repeated(total.cycles, p.cycles, repetitions);
  total.read_bytes = add_repeated(total.read_bytes, p.read_bytes, repetitions);
  total.write_bytes = add_repeated(total.write_bytes, p.write_bytes, repetitions);
  total.peak_sram = std::max(total.peak_sram, p.peak_sram);
  total.small_tiles = add_repeated(total.small_tiles, p.small_tiles, repetitions);
  total.large_tiles = add_repeated(total.large_tiles, p.large_tiles, repetitions);
  total.small_compute_cycles = add_repeated(total.small_compute_cycles, p.small_compute_cycles, repetitions);
  total.large_compute_cycles = add_repeated(total.large_compute_cycles, p.large_compute_cycles, repetitions);
  total.dma_busy_cycles = add_repeated(total.dma_busy_cycles, p.dma_busy_cycles, repetitions);
}

PhaseMetrics metrics(const Hardware& h) {
  return {h.finish(), h.read_bytes(), h.write_bytes(), h.peak_sram(),
          h.array_tiles(Array::Small), h.array_tiles(Array::Large),
          h.array_compute_cycles(Array::Small), h.array_compute_cycles(Array::Large), h.dma_busy_cycles()};
}

Count ceil_div(Count n, Count d) { return (n + d - 1) / d; }

// For full 48-row groups use 32 rows on the large and 16 on the small
// engine. Tiny tails use a 16-row tile. Both arrays share all input buffers.
struct RowTiles {
  Count small, large;
  Count padded() const { return 16 * small + 32 * large; }
};

RowTiles rows_for(Count rows) {
  const auto groups = rows / 48;
  const auto tail = rows % 48;
  return {groups + (tail > 0 && (tail <= 16 || tail > 32)),
          groups + (tail > 16)};
}

Cycle panel_gemm(Hardware& h, Count rows, Count columns, unsigned depth,
                 Cycle operands_ready) {
  const auto rt = rows_for(rows);
  const auto col_tiles = ceil_div(columns, 16);
  Cycle end = operands_ready;
  if (rt.small) {
    end = h.gemm_repeat(Array::Small, depth, rt.small * col_tiles, operands_ready);
  }
  if (rt.large) {
    end = std::max(end, h.gemm_repeat(Array::Large, depth, rt.large * col_tiles,
                                      operands_ready));
  }
  return end;
}

// Signed native INT8 panels, INT16 calibrated partial return, widening/scale
// restoration in SIMD. A16 is the stored model input, W8 the stored weights.
// DMA counts compact edge payloads; SIMD zero-fills padded activation lanes.
// DRAM panels are assumed packed/blocked for the selected output slab.
PhaseMetrics dense_slab(Count m, unsigned n, unsigned k, const HardwareConfig& cfg) {
  Hardware h(cfg);
  const auto padded = rows_for(m).padded();
  const Count outputs = padded * n;
  const Count depth_max = std::min(k, 256u);
  h.reserve(2*m*depth_max + padded*depth_max + n*depth_max + 6*outputs);
  Cycle ready = 0;
  for (unsigned offset = 0; offset < k; offset += 256) {
    const auto depth = std::min(256u, k-offset);
    const auto a = h.read(2*m*depth, ready);
    const auto w = h.read(Count(n)*depth, ready);
    const auto packed = h.vector((2*m+padded)*depth, a);
    const auto done = panel_gemm(h, m, n, depth, std::max(w, packed));
    // Read native partial, and previous wide sum on subsequent passes.
    ready = h.vector(outputs * (offset == 0 ? 6 : 10), done);
  }
  ready = h.vector((4*padded + 2*m)*n, ready);
  h.write(2*m*n, ready);
  return metrics(h);
}

void dense(PhaseMetrics& out, Count m, unsigned n, unsigned k, const HardwareConfig& cfg) {
  // Each output slab completes before reusing its allocation. All model N
  // dimensions are multiples of 128, including the two fused projections.
  constexpr Count mblock = 192;
  constexpr unsigned nblock = 128;
  if (n % nblock) throw std::logic_error("unexpected projection width");
  if (m >= mblock) {
    append(out, dense_slab(mblock, nblock, k, cfg), (m/mblock) * (n/nblock));
  }
  if (m % mblock) append(out, dense_slab(m%mblock, nblock, k, cfg), n/nblock);
}

enum class VectorOp { Norm, Rotary, CopyKV, Residual, SwiGLU };

// Streaming row operations on AT MOST 16k elements per visit. A Norm visit
// includes only complete 4096-feature rows and loads one 16-bit gamma vector.
// Rotary tables hold a sine/cosine pair for each pair of head features; reuse
// the table across all heads at each position. KV append goes through SRAM.
PhaseMetrics vector_rows(Count rows, unsigned width, VectorOp op, const HardwareConfig& cfg) {
  auto slab = [&](Count r) {
    Hardware h(cfg);
    const Count n = r * width;
    Count read = 2*n, second_input = 0, write = 2*n, second_output = 0;
    Count port_bytes = 0, auxiliary = 0;
    switch (op) {
      case VectorOp::Norm:
        auxiliary = 2*width;
        port_bytes = 8*n;
        break;
      case VectorOp::Rotary:
        // Q and K occupy distinct ranges. V bypasses RoPE.
        read = write = 2*r*hidden;
        second_input = second_output = 2*r*kv_width;
        auxiliary = r*head_dim*2;
        port_bytes = 6*n;
        break;
      case VectorOp::CopyKV:
        read = write = second_input = second_output = 2*r*kv_width;
        break;
      case VectorOp::Residual:
      case VectorOp::SwiGLU:
        second_input = 2*n;
        port_bytes = 6*n;
        break;
    }
    h.reserve(read + second_input + write + second_output + auxiliary);
    Cycle t = h.read(read);
    if (second_input) t = h.read(second_input);
    if (auxiliary) t = h.read(auxiliary);
    if (port_bytes) t = h.vector(port_bytes, t);
    t = h.write(write, t);
    if (second_output) h.write(second_output, t);
    return metrics(h);
  };
  const Count block = std::max<Count>(1, 16384/width);
  PhaseMetrics result;
  if (rows >= block) append(result, slab(block), rows/block);
  if (rows % block) append(result, slab(rows%block), 1);
  return result;
}

// Score and probability panels have packed [KV block][query head/row][key]
// layout in DRAM. Each transfer below is one contiguous payload descriptor.
// Q/K/V packing uses three SRAM bytes/actual lane (A16 read, INT8 write).
// K/V are shared by the four GQA heads, flattened along the query row axis.
PhaseMetrics attention_group(unsigned q, unsigned prefix, AttentionSchedule algorithm,
                             const HardwareConfig& cfg) {
  Hardware h(cfg);
  const Count rows = group * q;
  const Count padded = rows_for(rows).padded();
  const Count qb = rows * head_dim;
  const unsigned max_keys = std::min(kv_block, prefix);
  const Count padded_keys = 16 * ceil_div(max_keys, 16);
  const Count score_scratch = padded * padded_keys;
  const Count value_scratch = padded * head_dim;
  const Count kv = Count(max_keys)*head_dim;
  const Count blocks = ceil_div(prefix, kv_block);
  auto keys_at = [prefix](Count b) {
    return std::min<Count>(kv_block, prefix - b*kv_block);
  };

  Cycle done = 0;
  if (algorithm == AttentionSchedule::Naive) {
    // Pass 1: QK. Scores are 16-bit; compact spills omit padding lanes.
    const Count qk_space = 2*qb + padded*head_dim + 3*kv + 2*score_scratch;
    h.reserve(qk_space);
    done = h.vector(2*qb+padded*head_dim, h.read(2*qb));
    for (Count b = 0; b < blocks; ++b) {
      const auto n = keys_at(b);
      auto packed = h.vector(3*n*head_dim, h.read(2*n*head_dim, done));
      done = panel_gemm(h, rows, n, head_dim, packed);
      done = h.write(2*rows*n, done);
    }
    h.release(qk_space);

    // Pass 2: rowwise stable softmax using max, exp sum, then normalized
    // output, with register intermediates and three reads of each score. The
    // causal mask uses absolute query positions inside the final block(s).
    // Select a row chunk that fits even with a long cached context.
    const Count row_bytes = 2*Count(prefix)+8;
    const Count soft_rows = std::min({rows, Count(32), cfg.scratchpad_bytes / row_bytes});
    if (!soft_rows) throw std::invalid_argument("one softmax row exceeds SRAM");
    h.reserve(soft_rows*row_bytes);
    for (Count r = 0; r < rows; r += soft_rows) {
      const auto nr = std::min(soft_rows, rows-r);
      for (Count b = 0; b < blocks; ++b) done = h.read(2*nr*keys_at(b), done);
      done = h.vector(8*nr*prefix, done);
      for (Count b = 0; b < blocks; ++b) done = h.write(2*nr*keys_at(b), done);
    }
    h.release(soft_rows*row_bytes);

    // Pass 3: normalized P16 -> P8; V16 -> V8. Native PV partials are
    // widened before summation in the running FP32 output.
    const Count pv_space = 3*kv + 3*score_scratch + 6*value_scratch;
    h.reserve(pv_space);
    for (Count b = 0; b < blocks; ++b) {
      const auto n = keys_at(b);
      const auto v = h.read(2*n*head_dim, done);
      const auto p = h.read(2*rows*n, done);
      const auto packed = h.vector((2*rows+padded)*n + 3*n*head_dim, std::max(p,v));
      done = panel_gemm(h, rows, head_dim, n, packed);
      done = h.vector(value_scratch*(b == 0 ? 6 : 10), done);
    }
    done = h.vector(4*value_scratch+2*qb, done);
    h.write(2*qb, done);
    h.release(pv_space);
  } else {
    // Double-buffered K/V: prefetch b+1 while processing b. The slot for
    // b+2 is only available after b's PV has drained and its update is done.
    const Count allocation = 2*qb + padded*head_dim + 2*6*kv +
                             7*score_scratch + 6*value_scratch + 8*rows;
    h.reserve(allocation);
    const auto query_ready = h.vector(2*qb+padded*head_dim, h.read(2*qb));
    auto load = [&](Count b, Cycle slot_free) {
      const auto bytes = 2*keys_at(b)*head_dim;
      h.read(bytes, slot_free);
      return h.read(bytes, slot_free);
    };
    Cycle ready = load(0, 0);
    for (Count b = 0; b < blocks; ++b) {
      const auto n = keys_at(b);
      Cycle next = b+1 < blocks ? load(b+1, done) : 0;
      const auto packed = h.vector(6*n*head_dim, ready);
      const auto scores = panel_gemm(h, rows, n, head_dim, std::max({packed, query_ready, done}));
      // SIMD reads logits and builds normalized-to-local-max INT8 weights;
      // FP32 temporaries allow two passes over exp values. m/l are FP32.
      const auto weights = h.vector((2*rows + 8*rows + padded)*n + 16*rows, scores);
      const auto values = panel_gemm(h, rows, head_dim, n, weights);
      // Fused rescale of old numerator and addition of new widened partial.
      done = h.vector(value_scratch*(b == 0 ? 6 : 10), values);
      ready = next;
    }
    done = h.vector(4*value_scratch+4*rows+2*qb, done);
    h.write(2*qb, done);
    h.release(allocation);
  }
  return metrics(h);
}

}  // namespace

LayerMetrics run_decoder_layer(const LayerShape& shape, AttentionSchedule schedule,
                               const HardwareConfig& config) {
  if (!shape.batch || !shape.queries || shape.context < shape.queries ||
      shape.context > max_context || shape.batch > 65535) {
    throw std::invalid_argument("invalid Llama batch, query length or visible context");
  }
  const Count m = Count(shape.batch)*shape.queries;
  LayerMetrics out;
  append(out.layer, vector_rows(m, hidden, VectorOp::Norm, config));
  dense(out.layer, m, hidden+2*kv_width, hidden, config); // fused Q/K/V
  append(out.layer, vector_rows(m, hidden+kv_width, VectorOp::Rotary, config));
  append(out.layer, vector_rows(m, 2*kv_width, VectorOp::CopyKV, config));

  const unsigned past = shape.context - shape.queries;
  for (unsigned first = 0; first < shape.queries; first += q_block) {
    const auto q = std::min(q_block, shape.queries-first);
    // No command for fully future KV blocks; each issued rectangle includes
    // masked elements for early query rows. Identical completed GQA groups
    // are replayed exactly; distinct requests own distinct caches.
    append(out.attention, attention_group(q, past+first+q, schedule, config),
           Count(shape.batch)*kv_heads);
  }
  append(out.layer, out.attention);
  dense(out.layer, m, hidden, hidden, config); // attention output projection
  append(out.layer, vector_rows(m, hidden, VectorOp::Residual, config));
  append(out.layer, vector_rows(m, hidden, VectorOp::Norm, config));
  dense(out.layer, m, 2*ffn, hidden, config); // fused gate and up
  append(out.layer, vector_rows(m, ffn, VectorOp::SwiGLU, config));
  dense(out.layer, m, hidden, ffn, config); // down
  append(out.layer, vector_rows(m, hidden, VectorOp::Residual, config));
  return out;
}

}  // namespace arago
