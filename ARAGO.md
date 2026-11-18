# Arago inference exercise

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

Repeat with `--attention streaming` and a different output filename. Each run emits 256 JSON lines for one decoder layer across the workload. The first line has 2,048 queries **per user**. The next 255 lines have one query each and visible contexts from 2,049 through 2,303. The last prediction brings each sequence to 2,304 tokens. This represents 4,096 generated positions across the batch. Simulated traces carry shapes and counters; they do not emit token IDs. A decode step of duration T seconds gives 1/T tokens/s per user and 16/T tokens/s aggregate for an end-to-end model.

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

## Problems to resolve and improvements to test

| Problem observed | Next experiment |
|---|---|
| Independent C++ and CUDA implementations can agree on math while the scheduled buffer is stale. | Replay the same commands and named SRAM slots in both programs for one cold score tile and two consecutive KV blocks. Check data, dependencies, and slot reuse. |
| The current DMA and command posting policies can expose avoidable stalls. | Compare late posting with preposted descriptors and explicit dependencies. Give the two DMA links independent bandwidth knobs, then vary one parameter at a time to rank prefill and decode attention elasticity. |
| The 32×16 array is idle in the current batch-16 decode mapping. | For a dense projection, try splitting output features across both arrays. Account for input packing, result layout and DMA contention. |
| The scaled output depends on seeing the wide result before it crosses the 16-bit port. | Compare bounded input quantization with the four-product reference. Benchmark programmable pre-return shift/round as a proposed hardware change. Record clipping, numerical error and target cycles. |
| Timing currently covers one decoder layer. Four independent simulator processes would miss contention on a shared host link. | Include all 32 layers, final normalization, vocabulary head, sampling and return for TTFT and all 255 cached steps. For four devices, model shard dependencies, host collectives and shared directional bandwidth, then solve for 70% of peak interactivity. |

Two candidate hardware improvements are a queued DMA descriptor list to hide command delivery between transfers, and programmable array-output shifts before the INT16 return. Compare each independently with the fixed baseline and report its change in per-user decode rate.

For the report, match the FlashAttention pseudocode to the measured SRAM allocation (449 KiB for streaming prefill), rank the measured bottlenecks, and evaluate two architectural changes individually against the same baseline. Keep hardware proposals separate from software-only scheduling improvements. [One-layer timing results](example/arago/results/layer-comparison.json) · [Layer schedule](src/arago/DecoderLayer.cc)
