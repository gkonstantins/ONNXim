#include "DecoderLayer.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace arago {
namespace {

using Count = std::uint64_t;
constexpr unsigned hidden = 4096, head_dim = 128, kv_heads = 8, group = 4;
constexpr unsigned kv_width = kv_heads * head_dim, ffn = 14336;

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
  if (!total.sample && p.sample && repetitions) {
    total.sample = p.sample;
    total.sample_offset = total.cycles + p.sample_offset;
  }
  if (p.program.size() == 1 && repetitions) {
    const auto& run = p.program.front();
    total.program.push_back({run.id, run.cycles, add_repeated(0, run.repeats, repetitions)});
  } else if (!p.program.empty()) {
    for (Count i = 0; i < repetitions; ++i)
      total.program.insert(total.program.end(), p.program.begin(), p.program.end());
  }
  total.cycles = add_repeated(total.cycles, p.cycles, repetitions);
  total.read_bytes = add_repeated(total.read_bytes, p.read_bytes, repetitions);
  total.write_bytes = add_repeated(total.write_bytes, p.write_bytes, repetitions);
  total.peak_sram = std::max(total.peak_sram, p.peak_sram);
  total.small_tiles = add_repeated(total.small_tiles, p.small_tiles, repetitions);
  total.large_tiles = add_repeated(total.large_tiles, p.large_tiles, repetitions);
  total.small_compute_cycles = add_repeated(total.small_compute_cycles, p.small_compute_cycles, repetitions);
  total.large_compute_cycles = add_repeated(total.large_compute_cycles, p.large_compute_cycles, repetitions);
  total.vector_bytes = add_repeated(total.vector_bytes, p.vector_bytes, repetitions);
  total.dma_busy_cycles = add_repeated(total.dma_busy_cycles, p.dma_busy_cycles, repetitions);
}

PhaseMetrics metrics(const Hardware& h) {
  PhaseMetrics result{h.finish(), h.read_bytes(), h.write_bytes(), h.peak_sram(),
          h.array_tiles(Array::Small), h.array_tiles(Array::Large),
          h.array_compute_cycles(Array::Small), h.array_compute_cycles(Array::Large), h.vector_bytes(), h.dma_busy_cycles()};
  if (h.primitive_id() >= 0) result.program.push_back({h.primitive_id(),result.cycles,1});
  if (!h.events().empty()) {
    const Count events = h.read_descriptors()+h.write_descriptors()+h.vector_commands()
                       +3*(result.small_tiles+result.large_tiles);
    result.sample = std::make_shared<EngineWindow>(EngineWindow{h.events(),result.cycles,events});
  }
  return result;
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

Cycle native_panel(Hardware& h, Count rows, Count columns, unsigned depth,
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

// In exact4 mode hold two digits per input and one additional INT32
// reconstruction buffer. The native partial slot is reused four times.
Cycle panel_gemm(Hardware& h, Count rows, Count columns, unsigned depth,
                 Cycle ready, const HardwareConfig& cfg, bool dense_weight = false) {
  if (!cfg.exact_digits) return native_panel(h, rows, columns, depth, ready);
  const Count padded = rows_for(rows).padded();
  const Count cols = 16*ceil_div(columns,16);
  const Count inputs = (padded+cols)*depth, outputs = padded*cols;
  const Count left_digits = cfg.exact_a16 ? 4 : 2;
  const Count right_digits = cfg.exact_a16 && !dense_weight ? 4 : 2;
  // The transpose path exchanges operand digit counts for A16/W8.
  const Count digit_space = (left_digits*padded + right_digits*cols)*depth;
  const Count wide = cfg.exact_a16 ? 8 : 4;
  const Count storage = dense_weight && cfg.transpose_decode && cfg.exact_a16
      ? std::max(digit_space,(right_digits*padded+left_digits*cols)*depth) : digit_space;
  const Count extra = storage + wide*outputs;
  h.reserve(extra);
  ready = h.vector(inputs + storage + (cfg.exact_a16 ? inputs : 0), ready);
  for (unsigned part=0; part<left_digits*right_digits; ++part) {
    ready = native_panel(h,rows,columns,depth,ready);
    ready = h.vector((2 + wide + (part ? wide : 0))*outputs,ready);
  }
  h.release(extra);
  return ready;
}

// Native signed panels and partial returns use the configured interface widths.
// A16 is the stored model input, W8 the stored weights, independently of the ISA.
// DMA counts compact edge payloads; SIMD zero-fills padded activation lanes.
// DRAM panels are assumed packed/blocked for the selected output slab.
PhaseMetrics dense_slab(Count m, unsigned n, unsigned k, const HardwareConfig& cfg) {
  Hardware h(cfg);
  const auto padded = rows_for(m).padded();
  const Count outputs = padded * n;
  const Count depth_max = std::min(k, 256u);
  const Count ib = cfg.input_bytes(), ob = cfg.output_bytes();
  h.reserve(2*m*depth_max + ib*(padded+n)*depth_max + (ob+4)*outputs +
            (ib == 1 ? 0 : n*depth_max));
  Cycle ready = 0;
  for (unsigned offset = 0; offset < k; offset += 256) {
    const auto depth = std::min(256u, k-offset);
    const auto a = h.read(2*m*depth, ready);
    const auto w = h.read(Count(n)*depth, ready);
    auto packed = h.vector((2*m+ib*padded)*depth, a);
    if (ib != 1) packed = h.vector((1+ib)*n*depth, std::max(w,packed));
    auto ready_inputs = std::max(w,packed);
    Cycle done;
    if (cfg.transpose_decode && m <= 16) {
      // Transpose the two packed panels, compute Y^T, transpose partials back.
      const Count transposed = ib*(m+n)*depth;
      h.reserve(transposed + ob*outputs);
      ready_inputs = h.vector(2*transposed,ready_inputs);
      done = panel_gemm(h,n,m,depth,ready_inputs,cfg,true);
      done = h.vector(2*ob*outputs,done);
      h.release(transposed + ob*outputs);
    } else {
      done = panel_gemm(h,m,n,depth,ready_inputs,cfg,true);
    }
    // Read native partial, and previous wide sum on subsequent passes.
    ready = h.vector(outputs * (ob + (offset == 0 ? 4 : 8)), done);
  }
  ready = h.vector((4*padded + 2*m)*n, ready);
  h.write(2*m*n, ready);
  return metrics(h);
}

PhaseMetrics dense(Count m, unsigned n, unsigned k, const HardwareConfig& cfg) {
  // Each output slab completes before reusing its allocation. All model N
  // dimensions are multiples of 128, including the two fused projections.
  PhaseMetrics out;
  constexpr Count mblock = 192;
  constexpr unsigned nblock = 128;
  if (n % nblock) throw std::logic_error("unexpected projection width");
  if (m >= mblock) {
    append(out, dense_slab(mblock, nblock, k, cfg), (m/mblock) * (n/nblock));
  }
  if (m % mblock) append(out, dense_slab(m%mblock, nblock, k, cfg), n/nblock);
  return out;
}

// Fixedshift MLP. Weights are blocked in DRAM in array-consumption order.
// Each slab drains before the next slab reuses its workspace.
PhaseMetrics mlp_slab(unsigned rows, unsigned channels, unsigned depth, bool decode,
                      const HardwareConfig& cfg) {
  Hardware h(cfg);
  const Count padded = rows_for(rows).padded();
  const Count outputs = Count(rows)*channels;
  const Count returns = decode ? rows_for(channels).padded()*16 : padded*channels;
  const Count sums = decode ? outputs : returns;
  // Decode packs each user into a column of one persistent [16,depth] panel.
  // Prefill streams raw A16 in two slots into one reusable packed panel.
  const Count inputs = decode ? 18ULL*depth : (4*rows+padded)*256;
  const Count workspace = inputs + 2ULL*channels*256 + 2*returns + 4*sums + 2*outputs;
  if (decode && workspace > 676*1024) throw std::logic_error("decode MLP slab exceeds budget");
  h.reserve(workspace);
  Cycle input_ready = 0;
  if (decode) {
    for (unsigned user=0; user<rows; ++user) {
      const auto raw = h.read(2*depth,input_ready); // reuse one raw row
      const auto written_rows = user+1 == rows ? 16-rows+1 : 1;
      input_ready = h.vector((2+written_rows)*depth,raw);
    }
  }
  Cycle slot_free[2]{}, merged = 0;
  for (unsigned offset=0; offset<depth; offset+=256) {
    const unsigned slot = (offset/256)%2;
    auto a = decode ? input_ready : h.read(2*rows*256,slot_free[slot]);
    auto w = h.read(channels*256,slot_free[slot]);
    if (!decode) a = h.vector((2*rows+padded)*256,std::max(a,merged));
    const auto ready = std::max({a,w,merged});
    slot_free[slot] = decode ? native_panel(h,channels,rows,256,ready)
                             : native_panel(h,rows,channels,256,ready);
    merged = h.vector(2*returns+(offset ? 8 : 4)*sums,slot_free[slot]);
  }
  h.write(2*outputs,h.vector(4*sums+2*outputs,merged));
  return metrics(h);
}

PhaseMetrics mlp_projection(const LayerShape& shape, unsigned channels, unsigned depth,
                            bool gate_up, const HardwareConfig& cfg) {
  const Count rows = Count(shape.batch)*shape.queries;
  if (!cfg.mlp_pingpong) return dense(rows,channels,depth,cfg);
  const bool decode = shape.queries == 1;
  const unsigned row_block = decode ? 16 : 96;
  PhaseMetrics result;
  for (Count i=0; i<rows;) {
    const unsigned m = std::min<Count>(row_block,rows-i);
    const Count row_repeats = (rows-i)/m;
    unsigned width = 640;
    if (decode) {
      // A full 48-channel group gives 32 rows to one array and 16 to the other.
      const unsigned groups = (676*1024-18*depth)/(48*(512+32+6*m));
      width = std::min(gate_up ? 1104u : 720u,48*groups);
    }
    PhaseMetrics row_group;
    for (unsigned j=0; j<channels;) {
      const unsigned n = std::min(width,channels-j);
      const unsigned repeats = (channels-j)/n;
      append(row_group,mlp_slab(m,n,depth,decode,cfg),repeats);
      j += n*repeats;
    }
    append(result,row_group,row_repeats);
    i += m*row_repeats;
  }
  return result;
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
        read = write = 2*r*hidden/cfg.tensor_shards;
        second_input = second_output = 2*r*kv_width/cfg.tensor_shards;
        auxiliary = r*head_dim*2;
        port_bytes = 6*n;
        break;
      case VectorOp::CopyKV:
        read = write = second_input = second_output = 2*r*kv_width/cfg.tensor_shards;
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
// Q/K/V packing reads A16 and writes the configured native operand width.
// K/V are shared by the four GQA heads, flattened along the query row axis.
PhaseMetrics attention_group(unsigned q, unsigned prefix, AttentionSchedule algorithm,
                             const HardwareConfig& cfg, unsigned first, std::vector<BlockEvent>* plan) {
  Hardware h(cfg);
  const Count ib = cfg.input_bytes(), ob = cfg.output_bytes();
  const auto kv_block = cfg.key_tile;
  const Count rows = (cfg.share_kv ? group : 1) * q;
  const Count padded = rows_for(rows).padded();
  const Count qb = rows * head_dim;
  const unsigned max_keys = std::min(kv_block, prefix);
  const unsigned compute_keys = cfg.hybrid_kv ? 128 : kv_block;
  const Count padded_keys = 16 * ceil_div(std::min(max_keys,compute_keys), 16);
  const Count score_scratch = padded * padded_keys;
  const Count value_scratch = padded * head_dim;
  const Count kv = Count(max_keys)*head_dim;
  const Count blocks = ceil_div(prefix, kv_block);
  auto keys_at = [prefix,kv_block](Count b) {
    return std::min<Count>(kv_block, prefix - b*kv_block);
  };

  Cycle done = 0;
  h.trace_label("Query load and pack");
  if (algorithm == AttentionSchedule::Naive) {
    // Pass 1: QK. Scores are 16-bit; compact spills omit padding lanes.
    const Count qk_space = 2*qb + ib*padded*head_dim + (2+ib)*kv + ob*score_scratch;
    h.reserve(qk_space);
    done = h.vector(2*qb+ib*padded*head_dim, h.read(2*qb));
    for (Count b = 0; b < blocks; ++b) {
      const auto n = keys_at(b);
      h.trace_label("QK and score spill", b);
      auto packed = h.vector((2+ib)*n*head_dim, h.read(2*n*head_dim, done));
      done = panel_gemm(h, rows, n, head_dim, packed,cfg);
      if (ob != 2) done = h.vector((ob+2)*rows*n, done); // native return -> FP16 spill
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
    h.trace_label("Row softmax");
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
    const Count pv_space = (2+ib)*kv + (2+ib)*score_scratch + (ob+4)*value_scratch;
    h.reserve(pv_space);
    for (Count b = 0; b < blocks; ++b) {
      const auto n = keys_at(b);
      h.trace_label("PV and accumulation", b);
      const auto v = h.read(2*n*head_dim, done);
      const auto p = h.read(2*rows*n, done);
      const auto packed = h.vector((2*rows+ib*padded)*n + (2+ib)*n*head_dim, std::max(p,v));
      done = panel_gemm(h, rows, head_dim, n, packed,cfg);
      done = h.vector(value_scratch*(ob+(b == 0 ? 4 : 8)), done);
    }
    h.trace_label("Attention output");
    done = h.vector(4*value_scratch+2*qb, done);
    h.write(2*qb, done);
    h.release(pv_space);
  } else if (algorithm == AttentionSchedule::RowBlocked) {
    // Keep all visible scores for a small query panel, normalize in place,
    // then consume probability slices. One reusable K or V staging slot.
    h.reserve(2*qb + ib*padded*head_dim + 2*rows*prefix + (2+ib)*kv +
              (ob+ib)*score_scratch + (ob+4)*value_scratch + 8*rows);
    done = h.vector(2*qb+ib*padded*head_dim,h.read(2*qb));
    for (Count b=0;b<blocks;++b) {
      const auto n=keys_at(b);
      h.trace_label("QK and score staging", b);
      const auto packed=h.vector((2+ib)*n*head_dim,h.read(2*n*head_dim,done));
      done=panel_gemm(h,rows,n,head_dim,packed,cfg);
      done=h.vector((ob*padded+2*rows)*n,done);
    }
    h.trace_label("Row softmax");
    done=h.vector(8*rows*prefix,done);
    for (Count b=0;b<blocks;++b) {
      const auto n=keys_at(b);
      h.trace_label("PV and accumulation", b);
      const auto packed=h.vector((2*rows+ib*padded)*n+(2+ib)*n*head_dim,h.read(2*n*head_dim,done));
      done=panel_gemm(h,rows,head_dim,n,packed,cfg);
      done=h.vector(value_scratch*(ob+(b ? 8 : 4)),done);
    }
    h.trace_label("Attention output");
    h.write(2*qb,h.vector(4*value_scratch+2*qb,done));
  } else {
    // Double-buffered K/V: prefetch b+1 while processing b. The slot for
    // b+2 is only available after b's PV has drained and its update is done.
    const Count allocation = 2*qb + ib*padded*head_dim + (cfg.double_buffer ? 2 : 1)*2*(2+ib)*kv +
                             (ob+4+ib)*score_scratch + (ob+4)*value_scratch + 8*rows;
    h.reserve(allocation);
    const auto query_ready = h.vector(2*qb+ib*padded*head_dim, h.read(2*qb));
    auto load = [&](Count b, Cycle slot_free) {
      const auto bytes = 2*keys_at(b)*head_dim;
      h.trace_label("K prefetch", b);
      h.read(bytes, slot_free);
      h.trace_label("V prefetch", b);
      return h.read(bytes, slot_free);
    };
    Cycle ready = load(0, 0), slot_ready = 0;
    for (Count b = 0; b < blocks; ++b) {
      const auto n = keys_at(b);
      const Cycle prior_done = done;
      Cycle next = cfg.double_buffer && b+1 < blocks ? load(b+1, done) : 0;
      h.trace_label("KV packing", b);
      const auto packed = h.vector(2*(2+ib)*n*head_dim, ready);
      Cycle first_scores = 0, last_pv = 0;
      for (Count offset = 0; offset < n; offset += compute_keys) {
        const Count width = std::min<Count>(compute_keys,n-offset);
        h.trace_label("QK", b);
        const auto scores = panel_gemm(h, rows, width, head_dim, std::max({packed, query_ready, done}),cfg);
        if (offset == 0) first_scores = scores;
        // SIMD reads logits and builds normalized-to-local-max INT8 weights;
        // FP32 temporaries allow two passes over exp values. m/l are FP32.
        h.trace_label("Online softmax", b);
        const auto weights = h.vector((ob*rows + 8*rows + ib*padded)*width + 16*rows, scores);
        h.trace_label("PV", b);
        last_pv = panel_gemm(h, rows, head_dim, width, weights,cfg);
        // Fused rescale of old numerator and addition of new widened partial.
        h.trace_label("Rescale and accumulate", b);
        done = h.vector(value_scratch*(ob+(b == 0 && offset == 0 ? 4 : 8)), last_pv);
      }
      if (plan) plan->push_back({first,q,static_cast<unsigned>(b*kv_block),
          static_cast<unsigned>(n),static_cast<unsigned>(cfg.double_buffer ? b%2 : 0),
          slot_ready,ready,first_scores,last_pv,done});
      slot_ready = cfg.double_buffer ? prior_done : done;
      ready = !cfg.double_buffer && b+1 < blocks ? load(b+1,done) : next;
    }
    h.trace_label("Attention output");
    done = h.vector(4*value_scratch+4*rows+2*qb, done);
    h.write(2*qb, done);
    h.release(allocation);
  }
  return metrics(h);
}

void record_phase(PhaseMetrics& total, OperatorProfile* profile, const char* name,
                  const PhaseMetrics& part, Count repetitions = 1) {
  append(total, part, repetitions);
  if (profile) {
    PhaseMetrics combined;
    append(combined, part, repetitions);
    profile->emplace_back(name, combined);
  }
}

}  // namespace

LayerMetrics run_decoder_layer(const LayerShape& shape, AttentionSchedule schedule,
                               const HardwareConfig& config, OperatorProfile* profile) {
  if (!shape.batch || !shape.queries || shape.context < shape.queries ||
      shape.context > max_context || shape.batch > 65535) {
    throw std::invalid_argument("invalid Llama batch, query length or visible context");
  }
  const auto tp = config.tensor_shards;
  if ((tp != 1 && tp != 2 && tp != 4) || !config.query_tile || !config.key_tile ||
      config.query_tile > 256 || config.key_tile > 256)
    throw std::invalid_argument("invalid mapping configuration");
  const auto q_block = schedule == AttentionSchedule::RowBlocked ? std::min(4u,config.query_tile) : config.query_tile;
  const Count m = Count(shape.batch)*shape.queries;
  LayerMetrics out;
  record_phase(out.layer, profile, "input_norm", vector_rows(m, hidden, VectorOp::Norm, config));
  record_phase(out.layer, profile, "qkv", dense(m, (hidden+2*kv_width)/tp, hidden, config)); // fused Q/K/V
  record_phase(out.layer, profile, "rope", vector_rows(m, (hidden+kv_width)/tp, VectorOp::Rotary, config));
  record_phase(out.layer, profile, "kv_append", vector_rows(m, 2*kv_width/tp, VectorOp::CopyKV, config));

  const unsigned past = shape.context - shape.queries;
  for (unsigned first = 0; first < shape.queries; first += q_block) {
    const auto q = std::min(q_block, shape.queries-first);
    // No command for fully future KV blocks; each issued rectangle includes
    // masked elements for early query rows. Identical completed GQA groups
    // are replayed exactly; distinct requests own distinct caches.
    auto panel_config = config;
    panel_config.trace_engines = config.trace_engines && first+q == shape.queries;
    append(out.attention, attention_group(q, past+first+q, schedule, panel_config,first,
           config.trace_panels ? &out.tile_plan : nullptr),
           Count(shape.batch)*kv_heads/tp*(config.share_kv ? 1 : group));
  }
  record_phase(out.layer, profile, "attention", out.attention);
  record_phase(out.layer, profile, "out_proj", dense(m, hidden, hidden/tp, config)); // attention output projection
  record_phase(out.layer, profile, "attention_residual", vector_rows(m, hidden, VectorOp::Residual, config));
  record_phase(out.layer, profile, "post_norm", vector_rows(m, hidden, VectorOp::Norm, config));
  record_phase(out.layer, profile, "gate_up", mlp_projection(shape,2*ffn/tp,hidden,true,config)); // fused gate and up
  record_phase(out.layer, profile, "swiglu", vector_rows(m, ffn/tp, VectorOp::SwiGLU, config));
  record_phase(out.layer, profile, "down_proj", mlp_projection(shape,hidden,ffn/tp,false,config)); // down
  record_phase(out.layer, profile, "ffn_residual", vector_rows(m, hidden, VectorOp::Residual, config));
  return out;
}

// Replicated embedding and vocabulary head; only last-position logits are needed.
PhaseMetrics run_endpoints(const LayerShape& shape, const HardwareConfig& cfg, Cycle& embedding,
                           OperatorProfile* profile) {
  PhaseMetrics result;
  Hardware embed(cfg);
  embed.reserve(3*hidden);
  auto t = embed.read(hidden);  // W8 embedding row
  t = embed.vector(3*hidden,t);
  embed.write(2*hidden,t);
  record_phase(result,profile,"embedding",metrics(embed),Count(shape.batch)*shape.queries);
  embedding = result.cycles;
  record_phase(result,profile,"final_norm",vector_rows(shape.batch,hidden,VectorOp::Norm,cfg));
  record_phase(result,profile,"lm_head",dense(shape.batch,128256,hidden,cfg));
  // Greedy argmax over A16 logits in SRAM-sized chunks, one INT32 id per user.
  Hardware argmax(cfg);
  argmax.reserve(2*16384+16);
  t = 0;
  for (unsigned offset=0; offset<128256; offset+=16384) {
    const Count bytes = 2*std::min(16384u,128256-offset);
    t = argmax.vector(bytes+16,argmax.read(bytes,t));
  }
  argmax.write(4,t);
  record_phase(result,profile,"argmax",metrics(argmax),shape.batch);
  return result;
}

}  // namespace arago
