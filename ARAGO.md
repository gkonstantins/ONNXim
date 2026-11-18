# Arago inference exercise

[Requirements coverage and open issues](docs/coverage.md) · [Architecture, accuracy and four-device comparison](docs/architecture-comparison.md)

Start with one accelerator, one synchronous batch of 16 users, and materialized causal attention. Run streaming attention on the same requests as the first optimization. ONNXim supplies target cycles. PyTorch on the RTX 4060 supplies numerical results. The two programs currently execute independent implementations. The build commands, cold-tile test and existing measurements are in the [simulator guide](docs/arago.md).

## How to supply batch 16

**One request is one sequence.** Use 16 independent prompts of 2,048 tokens each, one shared copy of the model weights, and 16 separate key/value histories. Each user asks for 256 new tokens. The batch dimension stays 16 while the context grows.

| Stage | Llama input token IDs | Dense projection input | Attention |
|---|---|---|---|
| Prefill | `[16, 2048]` | Hidden states `[16, 2048, 4096]` can be viewed as 32,768 rows of width 4,096 sharing the weights. | Each query sees only preceding and current positions in its own sequence. The last prompt position predicts output token 1 for each user. |
| Each cached step | `[16, 1]` | 16 fresh rows of width 4,096. | Each user's newest token extends its own cache and predicts the next token. Each step advances the batch by 16 output tokens. |

Across 32 layers, reserving 2,304 positions for 16 users with eight KV heads, head width 128 and two bytes per K or V element requires `16 × 2304 × 32 × 8 × 128 × 2 × 2 = 4.5 GiB` for KV. The two factors of two mean K plus V, and bytes per element. A target with 64 GiB of DRAM can hold this and the W8 weights. On the laptop GPU the checker stages user rows and KV groups successively to fit memory. That preserves independent per-user numerical checks. Only the target timing run models simultaneous resource use.

### Run the target workload

ONNXim accepts a CSV of request lengths. The [batch-16 trace](example/arago/batch16.csv) has 16 data rows, each `0,2048,256,0`: zero interarrival delay, 2,048 prompt tokens, 256 requested output tokens, zero cached positions at admission. The scheduler assigns a separate request ID to each row. The `--batch 16` option sets the admission limit; the rows supply the requests.

```bash
build/arago/cmake/arago/onnxim-arago --mode arago \
  --config configs/arago_dual_array.json --model models/language_models/llama3-8b.json \
  --trace_file example/arago/batch16.csv --batch 16 --attention naive \
  > /tmp/arago-b16-naive.jsonl
```

Repeat with `--attention flashattention` and a different output filename. This selects the tiled online-softmax kernel; `streaming` remains its original CLI name. Each run emits 256 JSON lines for one decoder layer across the workload. The first line has 2,048 queries **per user**. The next 255 lines have one query each and visible contexts from 2,049 through 2,303. The last prediction brings each sequence to 2,304 tokens. This represents 4,096 generated positions across the batch. Simulated traces carry shapes and counters; they do not emit token IDs. A decode step of duration T seconds gives 1/T tokens/s per user and 16/T tokens/s aggregate for an end-to-end model.

### Measured improvement over naive attention

Both runs used the same hardware configuration and all 16 requests, from the 2,048-token prompt through each user's 256th output. Times are from the 1 GHz target clock for one decoder layer. The decode mean includes all 255 cached steps with their actual context lengths. Totals sum execution cycles across iterations.

| Measured interval | Naive | FlashAttention | Speedup |
|---|---:|---:|---:|
| Prefill, attention only | 1.606 s | 1.085 s | 1.480× |
| Prefill, complete layer | 21.462 s | 20.941 s | 1.025× |
| Mean cached step, attention only | 20.195 ms | 12.832 ms | 1.574× |
| Mean cached step, complete layer | 52.616 ms | 45.253 ms | 1.163× |
| All 256 iterations, attention only | 6.756 s | 4.357 s | 1.550× |
| **All 256 iterations, complete layer** | **34.879 s** | **32.481 s** | **1.074×** |

The schedule saves **2.398 s (6.88%)** of complete-layer time across this batch. Both modes issue the same number of native attention matrix products. FlashAttention keeps logits and weights in SRAM and can prefetch K/V during compute. Across the run, attention DRAM reads plus writes fall from 48.734 to 38.492 GiB (21.0%). Peak prefill attention SRAM rises from 192 to 449 KiB. The projections and feed-forward network consume identical cycles in both runs, which limits the overall speedup.

To reproduce the [full-trace comparison](example/arago/results/batch16-flashattention.json), run:

```bash
python3 tools/arago/run_comparison.py --workload batch16 \
  --output /tmp/arago-batch16-flashattention.json
```

### Supply token values for the numerical check

For ordinary generation, tokenize 16 independent prompts into the rows of `input_ids[16, 2048]`, then feed back one sampled token ID per row as `input_ids[16, 1]` with the saved K/V. The fixed-length assignment needs no padding when every prompt has 2,048 tokens.

With a checkpoint and the default settings, the checker takes 16 distinct deterministic windows from the checkpoint's model card. For custom data, provide a UTF-8 JSON array of 16 strings, one text per user. With `--prompt 2048 --context 2176`, each text must supply at least 2,175 tokenizer tokens before the checker adds the beginning-of-sequence token. The first 2,048 positions form that user's prompt. The longer window supplies a known continuation for a teacher-forced cached-decode check at 2,176 positions. The checker uses the required prefix of longer texts. For example, if your source texts are `prompts/01.txt` through `prompts/16.txt`, create the JSON locally:

```python
import json
from pathlib import Path

texts = [Path(f"prompts/{i:02}.txt").read_text(encoding="utf-8") for i in range(1, 17)]
Path("prompts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
```

```bash
python tools/arago/check_attention.py --checkpoint "$MODEL_DIR" \
  --prompts-file /path/to/prompts.json --device cuda --batch 16 \
  --prompt 2048 --context 2176 --json /tmp/arago-b16-numerics.json
```

Omit `--prompts-file` to reproduce the original model-card experiment. CUDA computes Q/K/V and attention values for layer 0. In a full-model generation run the output token selected for each user becomes that user's next input. The numerical checker instead compares known-input results, so an early token-selection difference cannot obscure a later arithmetic error. [Checker](tools/arago/check_attention.py) · [Recorded checkpoint results](example/arago/results/numerics.json)

## Hardware and defaults

The assignment's architecture table and workload (September 18, 2026, pp. 2–4) supply the fixed widths, capacities, latencies, rates and model dimensions. The choices below specify how to run the initial experiment.

| Area | Working decision |
|---|---|
| Memory and DMA | 64 GiB DRAM, 1 MiB software-managed SRAM. Start with one FIFO for reads and writes. Each contiguous descriptor pays one first-byte wait (200 read, 150 write). Both 64 B/cycle links pipeline a payload. The baseline serializes descriptors and charges payload time once. |
| Control CPU | Use nonblocking command delivery: 100 cycles to an array, 200 to DMA, 300 to SIMD. Initially post when inputs and resource are ready. In general an engine starts at the latest of command arrival, dependency completion and resource availability. Early posting and queue depth are separate variables. |
| Arrays | One 16×16 and one 32×16 array, 8-bit inputs, 16-bit output, K up to 256. Assume one MAC per cell per cycle, overlap input feeding with arithmetic, and permit independent result drain to SRAM. Input rates are 64 and 96 B/cycle; each output port serves 8 B/cycle. |
| SIMD and clock | One 1 GHz target clock. SIMD is memory-bound, charged for SRAM reads plus writes at 128 B/cycle. Ignore SRAM bank conflicts as permitted. GPU elapsed time never enters these counters. |
| Numerical reference | For the mathematical reference use float attention. For an exact check of **already quantized signed INT8** dots, use four smaller integer products and wide SIMD reconstruction. This is a tile-level accuracy reference. The reported layer timings still use one native product per tile. |
| Fast numerical experiment | Store weights/activations as W8/A16 in the timing model. In the checker, pack operands to INT8, emulate calibrated INT16 tile returns, and retain FP32 online-softmax state. The current checker selects the scale using the wide output tile. Scale availability, its delivery cost, and the hardware narrowing rule are open. Surrounding checkpoint layers use NF4/BF16. |

**Why keep the integer reference?** At K=128 the all-127 dot equals 2,064,512. A signed INT16 return tops out at 32,767. An in-range exact fallback expands each signed byte as `x = 16h + l`, with both digits of magnitude at most eight. Each of the four digit products summed over at most 256 positions fits INT16. Reconstruct the dot in INT32 as `256 HH + 16 (HL + LH) + LL`. Timing this policy would require four commands and drains plus packing and reconstruction. The narrower input representation also needs its own accuracy check against A16. [Timing configuration](configs/arago_dual_array.json) · [Hardware implementation](src/arago/Hardware.cc)

## FlashAttention on the control CPU

The Linux host launches a kernel for a decoder layer. The device control CPU runs the loops below and issues DMA, array and vector commands. Its inputs are Q after rotary position encoding and the K/V history in device DRAM, including the current positions. The output is one 128-feature vector per user, query position and query head, ready for the output projection. The tile order lets K and V serve four query heads and lets almost all score and weight traffic stay on-chip. [Layer implementation](src/arago/DecoderLayer.cc) · [FlashAttention paper](https://arxiv.org/abs/2205.14135)

### Choose two levels of tiling

For one user and one of eight KV heads, take up to 32 query positions for each of its four query heads. Flatten these to at most 128 rows. Stream up to 128 key positions at a time. A full score product has shape `[128,128] × [128,128] → [128,128]`, with the shared head width as the reduction dimension. The second product multiplies tile weights by V, reducing over the key positions to give 128 output features per row. Each row retains its own query position and head. Flattening is only a layout choice.

Inside either full product, divide the 128 output columns into eight strips of 16. Divide the 128 output rows into two groups of 48 plus 32. Each group of 48 gives 16 rows to the small array and 32 to the large array. That makes 16 small-array and 24 large-array outputs per product. During single-token decode the same four query heads give just four live rows; the baseline uses a padded 16-row strip. Tail handling suppresses invalid rows, columns and future keys.

### Reserve 449 KiB at the peak of prefill

| Live SRAM data | Size | Lifetime |
|---|---:|---|
| Q16 input and packed Q8 | 32 + 16 KiB | Q8 stays resident while the kernel scans this panel's visible keys. |
| Two K/V staging slots | 2 × 96 KiB | Each slot holds K16, V16, K8 and V8 (32 + 32 + 16 + 16 KiB). Fill one while the other is in use. |
| QK return, vector exponential workspace, packed weights | 32 + 64 + 16 KiB | Reused on every key block. |
| PV return and FP32 running numerator U | 32 + 64 KiB | Reuse the return tile; preserve U across the scan. |
| FP32 running maximum m and mass l | 1 KiB | Two scalars for each of 128 rows. |
| **Total** | **449 KiB of 1,024 KiB** | One user, one KV head and one query panel at a time. |

The implementation reserves tensor storage. Alignment, scale metadata and descriptors would use part of the remaining 575 KiB if kept in SRAM. Final A16 output can occupy retired input or temporary storage until its DMA write completes. A full K/V block costs 64 KiB of DRAM reads, shared across all four heads for this query panel.

### Control program

`DMA`, `ARRAY_PANEL` and `SIMD` below represent device commands. The controller arranges dependencies and waits before consuming a result or overwriting a slot. Two independent commands can be in flight together. The two reads of K and V use separate descriptors on the single DMA FIFO. A panel command expands into the native tiles described above. The SIMD helpers use the documented fast-path scales. State initialization describes the starting values; the timing model folds it into first-block handling. It charges fused SIMD passes rather than a new command for every expression.

```text
for each user, KV head g, and query panel for the four heads sharing g:
    rows = valid (head, absolute query position) pairs for this panel
    key_end = 1 + max(absolute query position in rows)
    blocks = split [0, key_end) into blocks of at most 128 keys
    reserve the panel workspace
    Q_ready = DMA.read Q16 from this user and these rows
    Q8 = SIMD.pack(Q16) after Q_ready
    SIMD.initialize m = -infinity, l = 0, U = 0 for every row
    start DMA.read K16 and V16 for blocks[0] into slot[0]

    for b in increasing block order:
        cur = b mod 2
        if b+1 exists:
            start K16 and V16 reads for b+1 into the other slot
                when that slot's previous consumer is done
        K8, V8 = SIMD.pack(slot[cur]) after both current reads complete
        S16 = ARRAY_PANEL(Q8, transpose(K8)) after inputs are ready

        SIMD after S16 has drained to SRAM:
            restore the score scale, multiply by 1/sqrt(128)
            valid[row,key] = key is real and key_position <= query_position[row]
            for each row with at least one valid key in this block:
                new_m = max(m, maximum valid score)
                alpha = exp(m - new_m) if l > 0 else 0
                P = exp(score - new_m) on valid keys, 0 elsewhere
                l = alpha*l + sum(P)
                m = new_m
            for a row with no valid key: keep m,l; use alpha=1 and P=0
            pack P to P8 and retain its row scale

        T16 = ARRAY_PANEL(P8, V8) after P8 is ready
        SIMD after T16 has drained: U = alpha*U + restore_scale(T16)
        mark slot[cur] reusable after this update

    O16 = SIMD.convert_to_A16(U/l per row, or zero if l == 0)
    DMA.write valid O16 rows to this user's output addresses
    wait for the write before reusing the panel workspace
```

Each row holds a running maximum m, an exponential sum l and a weighted value sum U. When a later block raises m, multiplying both earlier sums by `exp(old_m − new_m)` puts them on the new scale. This makes the final division mathematically equivalent to applying one softmax across all valid keys in real arithmetic. Finite precision, INT8 packing and the assumed INT16 return contribute numerical error. The numerical reference compares both schedules against dense attention. [PyTorch checker](tools/arago/check_attention.py)

For the baseline, each DMA, array and vector command incurs 200, 100 and 300 delivery cycles respectively; their resource rates and DRAM first-byte waits still apply. Prefetch overlaps useful computation only when the other slot is free and DMA can actually serve it. In the recorded 16-user prefill, DRAM payload for the attention phase falls from 12.6875 to 4.5625 GiB; attention takes 1.606 billion cycles materialized and 1.085 billion streaming. These results cover the chosen mapping and numerical policy. [Recorded timing](example/arago/results/layer-comparison.json)

## Baseline observations and follow-up experiments

The [coverage audit](docs/coverage.md) records which of these experiments have run, their tests, and the remaining limitations.

| Baseline observation | Experiment |
|---|---|
| Independent C++ and CUDA implementations can agree on math while the scheduled buffer is stale. | Replay the same commands and named SRAM slots in both programs for one cold score tile and two consecutive KV blocks. Check data, dependencies, and slot reuse. |
| The current DMA and command posting policies can expose avoidable stalls. | Compare late posting with preposted descriptors and explicit dependencies. Give the two DMA links independent bandwidth knobs, then vary one parameter at a time to rank prefill and decode attention elasticity. |
| The 32×16 array is idle in the current batch-16 decode mapping. | For a dense projection, try splitting output features across both arrays. Account for input packing, result layout and DMA contention. |
| The scaled output depends on seeing the wide result before it crosses the 16-bit port. | Compare bounded input quantization with the four-product reference. Benchmark programmable pre-return shift/round as a proposed hardware change. Record clipping, numerical error and target cycles. |
| The original comparison covers one decoder layer. The architecture study now adds model endpoints and a shared host cut. | Include all 32 layers, final normalization, vocabulary head, sampling and return for TTFT and all 255 cached steps. For four devices, model shard dependencies, host collectives and shared directional bandwidth, then solve for 70% of peak interactivity. |

Two candidate hardware improvements are a queued DMA descriptor list to hide command delivery between transfers, and programmable array-output shifts before the INT16 return. Compare each independently with the fixed baseline and report its change in per-user decode rate.

For the report, use the control program and its 449 KiB prefill allocation above, rank the measured bottlenecks, and evaluate two architectural changes individually against the same baseline. Keep hardware proposals separate from software-only scheduling improvements. [One-layer timing results](example/arago/results/layer-comparison.json) · [Layer schedule](src/arago/DecoderLayer.cc)
