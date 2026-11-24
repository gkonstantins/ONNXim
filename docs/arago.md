# Arago dual-array simulator

This tree is ONNXim at `8599bee` plus the Arago timing extension. The graded assignment answers, including 32-block time, host bandwidth, and accuracy, are in the [application report](../../arago/README.md). Evidence and traces are in the [appendix](../../arago/APENDIX.md).

One call to `arago::Region` times **one rank and one decoder block**, plus the endpoints when the caller asks for them. It does not walk 32 blocks and it does not move bytes across the host. The application repeats the block 32 times, adds embedding and the vocabulary head once, and prices the host link. GPU time from the RTX replay is not an Arago cycle.

The engine default fetches and reduces **256 keys**. The query panel default is **32** positions. The submitted study sets the prefill panel to **48**. Reproduce the tables in [Archived one-layer comparison](#archived-one-layer-comparison) with `key_tile=128`. Those rows are not the graded schedule.

## What the model charges

| Resource | Rule |
|---|---|
| Clock and memory | 1 GHz, so 1 cycle is 1 ns. 64 GiB DRAM. 1 MiB software-managed SRAM. Weights and KV stay in DRAM. The model rejects a reservation that exceeds SRAM. |
| DMA | One engine. DRAM↔DMA and DMA↔SRAM are both 64 B/cycle in the preset. A transfer runs at the slower of the two. Setup is 200 cycles. The first byte is 200 cycles on a read and 150 on a write. A cold 1,024-byte read is `200 + 200 + 1024/64 = 416` cycles. The matching write is 366. |
| Arrays | 16×16 and 32×16. INT8 operands, INT16 return, reduction depth at most 256. Launch is 100 cycles. Input ports are 64 and 96 B/cycle. Each array drains at 8 B/cycle. A later tile may compute while the previous result drains. Together the arrays peak at 768 MACs/cycle. |
| SIMD | 300-cycle command, then `ceil(bytes / 128)`. Arithmetic is treated as memory-bound. Online softmax state stays in FP32. |

Service for a native product is `max(K, ceil(input bytes / input rate))`, which is one MAC per cell per cycle with the operand feed overlapped. The model does not charge SRAM bank conflicts, a shared control-CPU issue port, or array pipeline fill. Those are named limits in the application report, not hidden costs in this clock. [Preset](../configs/arago_dual_array.json) · [Clock](../src/arago/Hardware.cc) · [Schedule](../src/arago/DecoderLayer.cc)

The submitted arithmetic stores activations at 16 bits and packs array operands to symmetric INT8. A dot of depth `d` is bounded by `d × 127²`. Full QK uses depth 128 and return shift 6. Full PV uses depth 256 and return shift 7. The array is assumed to round the wide dot after that shift. SIMD restores the shift and the operand scales. Direct INT16 saturation is a different policy. It is not the submitted one.

FlashAttention keeps the running maximum, mass, and numerator in SRAM. Two K/V slots let the next fetch use the free slot. Four query heads share one KV head. QK reduces over the 128 head features. PV reduces over the key tile, at most 256. Decode can place keys, and then value channels, on the array rows so both arrays get work when only four query rows are live. The 128-key comparison below did not do that.

## Build

From this repository, with a C++20 compiler, CMake, Ninja, Python 3, and the ONNX, Protobuf, Boost.Program_options, and nlohmann JSON 3.11 development packages:

```bash
cmake -S . -B build/arago -G Ninja -DONNXIM_ARAGO_ONLY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/arago -j 4
ctest --test-dir build/arago --output-on-failure
```

On CMake 4, add `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` for the pinned third-party archives. The focused build writes `build/arago/cmake/arago/libarago_device.so` and the CLI `onnxim-arago`. The application’s lightweight build compiles `src/arago/` without this ONNX frontend. Both call the same region.

One trace, one layer, current 256-key default:

```bash
build/arago/cmake/arago/onnxim-arago --mode arago \
    --config configs/arago_dual_array.json \
    --model models/language_models/llama3-8b.json \
    --trace_file example/arago/decode.csv --batch 16 --attention flashattention
```

`streaming` is the same online algorithm. This CLI path times the block from hidden state to hidden state. It does not add the other 31 blocks or the host. Full-request TTFT and tokens/s/user come from `python3 -m carlota` in the application tree.

## Acceptance tests

**Cold score tile.** Operands are already packed INT8. For 32 query vectors and 16 key vectors at width 128, the DMA reads are 4,096 and 2,048 bytes and finish at cycles 464 and 896. Launch ends at 996, depth-128 compute at 1,124, and the 1,024-byte INT16 drain finishes at **1,252**.

**Shared DMA, both arrays.** The DMA intervals are [0,432), [432,896), and [896,1328). Both arrays then compute over [1428,1556). Results reach SRAM at **1,620** and **1,684**. The test checks that both arrays wait on the same K and that their compute overlaps.

These two checks are in `tests/arago/test_hardware.cc`. `tests/arago/test_layer.cc` checks streaming attention bytes, SRAM rejection, and K/V slot lifetime.

## Archived one-layer comparison

The table below is the archived 128-key, query-panel-32, one-block measurement. Streaming keeps the online softmax state in SRAM and uses two K/V slots. Naive spills 16-bit scores and probabilities to DRAM. Query blocks hold at most 32 positions. KV blocks hold at most 128 keys. On decode, that mapper assigns the short rows to the 16×16 array, so the **32×16 compute share is 0% of the streaming decode layer**. The submitted 256-key decode schedule uses both arrays. Do not quote the 0% figure for it.

| Phase, batch 16, one block | Schedule | Attention cycles | Block cycles | Attention DRAM read / write (bytes) | Attention peak SRAM (bytes) |
|---|---|---:|---:|---:|---:|
| Prefill, 2,048 | Naive | 1,606,082,560 | 21,462,325,736 | 8,992,587,776 / 4,630,511,616 | 196,608 |
| Prefill, 2,048 | Streaming | 1,085,243,392 | 20,941,486,568 | 4,630,511,616 / 268,435,456 | 459,776 |
| Cached decode, context 2,176 | Naive | 19,907,840 | 52,328,936 | 147,193,856 / 4,587,520 | 67,584 |
| Cached decode, context 2,176 | Streaming | 12,691,456 | 45,112,552 | 142,737,408 / 131,072 | 226,336 |

At 1 GHz those streaming block times are 20.941 s of prefill and 45.113 ms of one decode step, for **one layer**. Streaming attention is 1.480× the naive attention on prefill and 1.569× on that decode step. The full block speedups are 1.025× and 1.160× because the projections dominate the layer. The submitted peaks are **937.5 KiB** prefill attention and **427.03 KiB** decode attention, at 48 query positions and 256 keys.

## Numerical check

The checker in `tools/arago/check_attention.py` compares online attention with a dense FP32 reference. The JSON below is the earlier per-tile scale run on checkpoint layer 0, not the submitted fixed-shift passage.

```bash
python tools/arago/check_attention.py --device cpu --batch 1 --prompt 33 --context 131
```

| Relative RMS, layer 0, per-tile scales | Prefill | Cached decode |
|---|---:|---:|
| Float32 online vs float32 dense | 1.79 × 10⁻⁷ | 2.44 × 10⁻⁷ |
| Native online vs float32 attention | 5.307% | 4.772% |

The submitted dynamic fixed-shift replay, the 48-position panel, and the 32-layer perplexity are reported in the [application report](../../arago/README.md). Saturating the INT16 return without the shift fails that check. This simulator does not treat an RTX wall-clock sample as device time.

## What Carlota calls

`src/arago/Binding.h` is ABI version 1. The call takes UTF-8 hardware JSON, the schedule name, batch, query length, and visible context. The returned JSON is the rank’s cycles, bytes, SRAM peak, and optional tile plan. Copy it before the next call on the same thread. Failures return a JSON object with an error string.

Carlota loads `libarago_device.so` and calls that entry. The CLI reaches the same `Region::run` after ONNXim admits the trace. Operator profiles and engine-event traces are opt-in. A record at the default 1 GHz, INT8-in, INT16-out clock omits the extra `hardware` object. A nondefault `clock_hz` or port width includes it. Seconds are `cycles / clock_hz`.
