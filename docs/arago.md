# Arago dual-array target for ONNXim

The current FlashAttention default fetches and reduces 256 keys. Performance tables in this guide use the archived 128-key mapping.

This target models one Llama 3.1 8B decoder block on a device with a 16×16 array, a 32×16 array, one DMA engine, one software-managed scratchpad and one vector unit. The default experiment compares a materialized attention implementation with an online, double-buffered implementation. It reports target cycles, DRAM payload traffic and peak live SRAM for attention and for the complete block. An independent PyTorch program compares numerical outputs.

The port starts from ONNXim revision `8599bee46b311fb267ac34b0677ad7db3d8a8ccd`. It uses ONNXim's actual request trace parser, FIFO admission, language-model batch generation, cache progression and retirement. For each admitted batch, the Arago path lowers the decoder block to target operations and advances the scheduler by the resulting cycles. The focused build bypasses the stock square-array mapper, private per-core scratchpads and BookSim/Ramulator transport. [Target entry point](../src/arago/Run.cc) · [Decoder block](../src/arago/DecoderLayer.cc) · [Hardware model](../src/arago/Hardware.cc)

For the baseline assumptions, full 16-user trace, custom prompt inputs, and open issues, see the [exercise decisions](../ARAGO.md).

## Build and reproduce

From the fork's root, use a C++20 compiler, CMake, Ninja, Python 3 and the development libraries for ONNX, Protobuf, Boost.Program_options and nlohmann JSON 3.11 or later. Suitable Debian/Ubuntu package names include `libonnx-dev`, `libprotobuf-dev`, `libboost-program-options-dev` and `nlohmann-json3-dev`. CMake fetches pinned spdlog and robin-hood hashing source archives. With dependencies installed in a private prefix, supply `-DCMAKE_PREFIX_PATH=/path/to/prefix/usr` at configure time and make the matching shared libraries available at runtime.

```bash
cmake -S . -B build/arago -G Ninja -DONNXIM_ARAGO_ONLY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/arago -j 4
ctest --test-dir build/arago --output-on-failure

build/arago/cmake/arago/arago-test-hardware
python3 tools/arago/run_comparison.py \
    --simulator build/arago/cmake/arago/onnxim-arago \
    --output example/arago/results/layer-comparison.json
```

On CMake 4, append `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` to the configure command for the pinned third-party projects.

The comparison runner executes both schedules on both traces and validates the batch shape, resource limits and logical counters. To inspect per-iteration JSON for a single trace, run:

```bash
build/arago/cmake/arago/onnxim-arago --mode arago \
    --config configs/arago_dual_array.json \
    --model models/language_models/llama3-8b.json \
    --trace_file example/arago/decode.csv --batch 16 --attention flashattention
```

For prefill plus all 255 cached steps, see the [batch-16 benchmark and command](../ARAGO.md#measured-improvement-over-naive-attention). The explicit `flashattention` mode runs the online algorithm also selectable as `streaming`.

The prefill trace admits 16 simultaneous prompts of 2,048 positions. The decode trace admits 16 requests with 2,174 prior positions, runs a one-position priming iteration and then runs one cached decode at **2,176 visible positions**, including the current token. The comparison reports only the latter iteration for decode. Both traces use the upstream 8B model preset in single-layer mode and homogeneous batches. Timings cover a decoder block from input hidden states to output hidden states. Embeddings, the final vocabulary projection, sampling, the remaining 31 blocks and host transfers fall outside the measured interval.

## Hardware contract

| Resource | Default behavior |
|---|---|
| Memory | 64 GiB device DRAM, 1 MiB shared SRAM, one 1 GHz clock. Workspace reservations enforce the SRAM limit. |
| DMA | Reads and writes share a FIFO and 64 B/cycle payload bandwidth. Each descriptor pays 200 setup cycles and 200 read or 150 write cycles to the first byte. DRAM→DMA and DMA→SRAM stream the same payload. |
| Arrays | Native signed 8-bit operands, K up to 256, 16-bit output. Both engines can execute simultaneously. Each native tile pays 100 launch cycles. Array input rates are 64 and 96 B/cycle respectively. Each array drains its own result at 8 B/cycle. |
| Vector unit | One FIFO, 300 cycles to issue, then one cycle per 128 bytes of total SRAM reads and writes, rounded up. Arithmetic is assumed bandwidth-limited. |
| Array service | `max(K, ceil(native input bytes / input port rate))` models one MAC per cell per cycle and overlapping input service. A new tile may execute while the previous output drains. Bank conflicts, extra wavefront fill and instruction-level behavior are outside this model. |

To sweep a parameter, copy the [target JSON](../configs/arago_dual_array.json), adjust the value, then pass the copy to the single-trace command. The comparison runner uses the checked-in defaults. Transfer counters count physical DMA payload, including rereads under the chosen schedule. They exclude protocol bytes and hypothetical scale-metadata transfers. Reported SRAM is the largest simultaneous reservation.

## Acceptance tests

**Cold score tile.** The starting operands are already packed into native INT8. For 32 Q vectors and 16 K vectors at width 128, the two DMA descriptors carry 4,096 and 2,048 bytes. They finish at cycles 464 and 896. Launch ends at 996, array service at 1,124, and the 1,024-byte INT16 output finishes its 128-cycle drain at **1,252**. The numerical test below also covers the separate packing choices for stored 16-bit activations.

**Shared DMA, concurrent arrays.** Both query-panel reads and one shared K read are ready at cycle zero. The DMA intervals are [0,432), [432,896), [896,1328). Both arrays then compute over [1428,1556). The small and large results reach SRAM at cycles **1,620** and **1,684**. The test checks the equal dependency on K as well as disjoint DMA intervals and overlapping compute.

**Decoder layer.** Both schedules have 32 query heads, 8 KV heads, width 128 per head, hidden size 4,096 and SwiGLU width 14,336. The layer includes RMSNorm, QKV projection, RoPE, KV append, causal GQA, output projection, residual additions and gated feed-forward projections. Query blocks contain up to 32 positions. Each group of four query heads shares K/V blocks of up to 128 positions. The dense projections stream 192-row by 128-column output slabs, with reduction panels up to 256. Repeated slabs and GQA groups can be counted exactly under the scheduler's explicit barriers. Changing cross-slab reuse or allowing further overlap would change these results.

- **Naive.** Compute QK, spill 16-bit scores to DRAM, reload score rows for stable softmax, spill 16-bit probabilities, then reload probabilities with V for PV. Both schedules omit wholly future blocks and mask future elements in the last issued rectangle.
- **Streaming.** Keep the per-query running maximum, normalization mass and value numerator in SRAM. Stage K/V blocks with two input slots. Readout and next-block prefetch can overlap where dependencies and the shared resources permit.

| Phase, batch 16 | Schedule | Attention cycles | Block cycles | Attention DRAM reads / writes (bytes) | Attention peak SRAM (bytes) |
|---|---|---:|---:|---:|---:|
| Prefill, 2,048 | Naive | 1,606,082,560 | 21,462,325,736 | 8,992,587,776 / 4,630,511,616 | 196,608 |
| Prefill, 2,048 | Streaming | 1,085,243,392 | 20,941,486,568 | 4,630,511,616 / 268,435,456 | 459,776 |
| Cached decode, 2,176 | Naive | 19,907,840 | 52,328,936 | 147,193,856 / 4,587,520 | 67,584 |
| Cached decode, 2,176 | Streaming | 12,691,456 | 45,112,552 | 142,737,408 / 131,072 | 226,336 |

At 1 GHz, a cycle is one target nanosecond. Streaming cuts attention traffic by 64.0% during prefill and 5.87% during decode. The measured virtual speedups are **1.480× / 1.569× for attention** and **1.025× / 1.160× for the full block** (prefill / decode). Both streaming cases trade additional SRAM for fewer spills and scheduling opportunities. The full JSON also reports complete-layer payload, peak allocation, per-array arithmetic service cycles and occupied DMA cycles (including setup and first-byte latency). Divide each busy count by the corresponding elapsed cycles. In streaming decode the 16×16 array computes for 40.1% of the layer and the 32×16 array for 0%. The mapper assigns the 16 dense rows and the four rows of each grouped-query attention operation to the small array. These are model predictions for this schedule. [Timing report](../example/arago/results/layer-comparison.json)

## Numerical check

The verified CUDA run used PyTorch 2.8.0+cu128, Transformers 4.57.6, bitsandbytes 0.49.2 and the [Unsloth Llama 3.1 8B NF4 checkpoint at revision 70d37f62](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit/tree/70d37f62f4f475a5a42dbbd5b7aad38f420e0960). Pass the local snapshot directory, including its tokenizer and README. A small seeded synthetic check also runs with a CPU installation of PyTorch:

```bash
python tools/arago/check_attention.py --device cpu --batch 1 --prompt 33 --context 131
python tools/arago/check_attention.py --checkpoint /path/to/local/70d37f62f4f475a5a42dbbd5b7aad38f420e0960 \
    --batch 16 --prompt 2048 --context 2176 --json example/arago/results/numerics.json
```

The checker derives Q/K/V from layer 0 for 16 distinct windows of the model-card text and applies RoPE using absolute positions. It evaluates an explicit float32 dense softmax reference and an online implementation, then exercises the native-format version of each schedule. Input Q/K and V are calibrated to signed INT8, probability or exponential weights to [0,127], and native output tiles are restored from INT16 with per-tile power-of-two scales. Naive stores FP16 scores and probabilities. Online accumulates its running state in FP32. All-masked rows must produce zero. Each schedule also passes its attention result through the checkpoint's output projection, residual, norm and MLP to compare decoder-block outputs. GPU execution serializes users and KV groups to bound VRAM, while preserving values for every tested position. This tests one checkpoint block under an illustrative native numeric contract. The checkpoint's other operators use pretrained NF4 weights with BF16 compute. Architecture W8 model-wide accuracy requires a corresponding quantized model.

The output records maximum absolute error, RMS error, and relative RMS aggregated over every user, head and output feature. It enforces relative RMS limits of 0.00003 for float online versus float dense, 0.02 for native online versus native naive, and 0.10 for native versus float reference. The numerical model assumes per-tile calibration. Validation on silicon would establish hardware support and cost. [PyTorch checker](../tools/arago/check_attention.py) · [Numerical report](../example/arago/results/numerics.json)

| Relative RMS difference on the checkpoint run | Prefill | Cached decode |
|---|---:|---:|
| Float32 online vs. float32 materialized attention | 1.79 × 10⁻⁷ | 2.44 × 10⁻⁷ |
| Native online vs. native materialized attention | 0.2373% | 0.2430% |
| Native online vs. float32 reference attention | 5.307% | 4.772% |
| Decoder-block outputs using native online vs. native materialized attention | 0.4166% | 0.4019% |
| Decoder-block output using native online vs. reference attention | 1.822% | 1.743% |

## Shared engine and Carlota

### Native formats and clock

The hardware JSON accepts `array_input_bits` (8 or 16), `array_output_bits`
(16 or 32), and integer `clock_hz` (1 through 1,000,000,000,000). The canonical
preset explicitly uses the assignment defaults: 8, 16, and 1,000,000,000.
Old presets without width fields retain the 8/16 defaults. Rates remain
bytes per **device cycle**, and all configured latencies remain cycles.

Widths change array input traffic, partial-output drain, packed SRAM buffers,
SIMD packing/widening and the attention numerical oracle. Stored model data
remain W8 weights and A16 activations/KV; changing the native interface does
not change that storage contract. Array geometry and nominal MACs per cycle
remain fixed. This is a port/format experiment, not an electrical estimate of
how a wider multiplier changes frequency, area or power.

`scaled`, `fixedshift`, `bounded` and `saturate` accept all supported width pairs.
`exact4` and `a16exact` are explicitly defined digit algorithms on the 8/16
interface; other interface widths with these policies are rejected. The
numerical checker accepts `--array-input-bits` and `--array-output-bits`.

For example, a full copy of the preset may replace these three fields:

```json
{"array_input_bits": 16, "array_output_bits": 32, "clock_hz": 750000000}
```

Pass the full JSON file to the standalone CLI's `--config`. Nondefault output
records include a `hardware` object identifying the clock and native widths.
Durations in seconds are `cycles / clock_hz`; scheduler arrival times in the
CSV remain cycle offsets. The Python study tools convert device cycles before
combining them with host nanoseconds. Default records preserve the frozen
format and values, and do not add the optional hardware object.

The focused build produces `build/arago/cmake/arago/libarago_device.so`. The CLI
uses ONNXim admission and retirement around `arago::Region::run`. Carlota calls
the same region through the C ABI in `src/arago/Binding.h`. ABI version 1
accepts UTF-8 hardware JSON, schedule, batch size, query length and visible
context. Copy the returned JSON before the next call on the same thread.
Failures return a JSON object with an error message. The canonical hardware
preset is `configs/arago_dual_array.json`. Operator profiles and engine-event
traces are opt-in. Explicit 128-key runs retain the original timing and output fields.
