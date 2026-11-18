# Coverage of the Arago inference assignment

Reviewed November 19, 2026 against all five pages of the September 18 assignment. Source SHA-256 `1239ad1e634bafa119519932c749b488ec7b509301cbb0b2ef322b9a979b22b5`. The audit tests conformance to the stated constants and to declared interpretations. Actual command traces, silicon measurements and a numerical ISA would be needed to validate those interpretations against a physical device.

[Complete workload case report](case-metrics.md) · [Four-device limits and host bandwidth](four-case-metrics.md) · [Interactivity recommendations](interactivity-recommendations.md) · [Measured comparison](architecture-comparison.md) · [Experiment manifest](../configs/arago_experiments.json) · [Target defaults](../configs/arago_dual_array.json) · [Kernel and SRAM map](../ARAGO.md#flashattention-on-the-control-cpu)

## Requirement and evidence

| PDF item | Implementation and independent check | Coverage boundary |
|---|---|---|
| p1, pp2,4–5 methodology, scope and documentation | This audit records source, assumptions, alternatives, commands, complete traces and numerical failures. | Exercise model; no vendor reference traces were supplied. |
| §2.1 host, Linux, control CPU | Trace admission and retirement, 1 GHz virtual time, per-resource dependency timestamps. Four-device host-star model counts token IDs, activations and reductions. Native cold-tile test checks each delivery boundary. | CPU instruction execution, Linux scheduling, tokenization, real host reduction implementation and unmodeled contention need measurements. Host service assumptions can be costed separately. |
| 64 GiB DRAM, 1 MiB scratchpad, DMA-only compute access | Independent size/overflow, invalid config and low-capacity tests. Named streaming slots, trace replay, measured 449 KiB peak. | Logical buffers and packed layouts. The rank capacity estimate partitions projections and KV, includes the largest stage’s endpoint matrices and retains KV for paused microbatches and a conservative activation allowance for the active microbatch. Metadata, alignment and bank conflicts are omitted. |
| DRAM↔DMA and DMA↔SRAM each 512 bits/cycle | Separate rates, pipelined payload at their minimum. Isolated-link tests at 32/64/128 B/cycle. Both widths need increasing for sustained payload improvement. DMA configuration 200 and first byte 200/150 are charged per descriptor. Hand oracle: 1,024-byte read/write take 416/366 cycles. | FIFO shared direction is default; optional independent read/write directions. No DRAM banks, burst granularity, address effects, or controller reorder model. |
| Array input 8 bits, output 16 bits, 16×16 and 32×16, K≤256 | Port, size, invalid-depth, pipeline and concurrent-work tests. 64/96 B/cycle in, 8 B/cycle out per array. Cold tile takes 1,252 cycles. Array command delivery is 100 cycles. | Baseline assumes one MAC/cell/cycle and overlapping feed. Separate MAC-rate, wavefront-fill and pipeline/no-overlap variants. No transistor-level simulation. |
| Command issue and pipeline semantics | Default late posting, 200/100/300 cycle delivery to DMA/array/SIMD. Queued mode permits device commands to overlap earlier service, FIFO deliveries cannot overtake on a resource. A native test explicitly schedules array posting at zero and checks 952 cycles for the same cold tile. Nine repeated native commands equal individually issued ones under all tested drain/queue combinations. | Queued layer scheduler posts data-dependent array work once the operands become ready. The 952-cycle separate test demonstrates an additional explicitly preposted dependency. Queue capacity, CPU issue width and hidden device fill are unspecified. |
| SIMD bandwidth 1,024 bits/cycle, unlimited compute | Ceil(total SRAM port bytes/128) plus 300-cycle delivery. Sweep both parameters. Fractional accesses round up. | Free arithmetic is specified. The number of passes per algorithm, fused SIMD operations and negligible SRAM first-byte delay are modeling choices. |
| Llama 3.1 8B, W8/A16, 2,048 input, 256 output, B16 | All full traces validate B, phases, first output from prefill, 255 cached steps ending at 2,303 visible positions. B1 control runs. 32 decoder layers, 32:8 grouped-query attention, independent per-user KV. Activation and weight movement is explicit. | No numeric data in timing-only runs. Each simulation reuses the deterministic timing of an identical layer 32 times; stages drain at their boundaries. |
| §2.2.1 control-CPU pseudocode, SRAM allocation, tiling, dataflow | Naive, SRAM row-blocked and streaming schedules. Double buffering, tile sizes, four-head K/V sharing and independent-head alternatives. Replay the emitted streaming block plan with PyTorch and inject a slot-lifetime violation. | Replay at KV-block granularity on one representative head group, for B1/B16 prefill and cached decode, checks float arithmetic. Per-PE hazards and every low-level instruction are outside the plan. |
| §2.2.2 largest FlashAttention elasticity | One-variable perturbations on fixed workloads. Compute absolute fractional attention-time change divided by actual fractional parameter change. Rank prefill separately from all 255 decode steps. | Among swept knobs, including the assumed MAC rate, MAC throughput ranks first (B16, 0.323 prefill / 0.299 decode). Among explicit PDF parameters it is array command latency (0.299 / 0.276). Small perturbations, discrete cycles and workload/mapping dependence matter. Capacity can be critical at a fit boundary despite zero local elasticity above it. |
| §2.2.3 TTFT and interactivity | Repeated native decoder schedule plus embedding gathers, last-position final norm, vocabulary projection and greedy argmax. B16 default warm-device TTFT 670.25 s, mean cached step 1.526 s, 0.6552 tokens/s/user (10.48 aggregate). | Includes device work. Add input/output transfers, dispatch and host processing for host-visible TTFT. Cold loading and arbitrary sampling policies are outside the estimate. |
| §2.2.4 two architectural improvements | Against identical B16 FlashAttention baseline, queued engine delivery gives 1.262× decode rate; doubling both DMA link rates gives 1.053×. Alternative software mapping transposes small dense projections and uses both arrays, giving 1.077×. Wider array output alone gives 1.008×. | Each intervention is independently compared. Queued delivery covers all three engine types. Performance changes depend on the chosen command granularity. |
| §2.2.5 four devices, parallelism, theoretical ceiling, host link for 70% | DP/TP/PP and hybrids, actual rank-local shapes, pipeline microbatches, five network models. Test directional byte counts, monotonic bandwidth, 70% crossing, infeasible startup, stage recurrence and functional split/reduce on a small dense reference. See below. | Analytical host transfers and shared-cut bounds. There is no packet simulator or physical multi-NPU benchmark. |
| §3–4 report, code, build, ZIP, share and meeting; p5 conclusion | Reproduction steps below, Markdown reports, test sources and JSON evidence. A source ZIP can be produced with `git archive --format=zip -o arago-study.zip HEAD`. | Checkpoint and toolchain obtained separately. Delivery to employer and booking a review at least 48 hours after sharing are external actions; the report records no submission or booking. |

## Numerical contract and precision costs

The specification gives widths and leaves rounding, accumulation, zero handling and scale transport open. The independent [integer tests](../tests/arago/test_precision.py) cover extreme operands, K=1/127/128/256, both output tile shapes, zero scales, signed rounding ties and clipping. CUDA checker results use sixteen (or one) distinct checkpoint windows and a known cached token. The model is an NF4 checkpoint with BF16 surrounding operators; weights and activations are timed as W8/A16. Relative RMS against dense float attention measures layer-0 numerical sensitivity. Task-level language quality, perplexity and an end-to-end W8 checkpoint remain open.

| Parameter | Arithmetic | Cost represented |
|---|---|---|
| scaled | Symmetric INT8, output scale chosen from the maximum of the wide native tile, nearest-even INT16. | Optimistic fused one-pass hardware; assumes scales available. |
| fixedshift | Choose safe power-of-two shift from K and input range before nearest-even INT16 narrowing. | Proposed programmable pre-return scale. One native pass, same ports. |
| saturate | Clip native dot to signed 16 bits. | One native pass. Quality threshold failures remain in the report. |
| bounded | Limit both quantized operands to ±11 so any K≤256 fits. | One native pass, lower input precision. |
| exact4 | Decompose each quantized INT8 operand into two balanced base-16 digits. Four safe INT16 native dots reconstruct in wide SIMD. | Extra digit buffers, four issues/drains, split and merge vector traffic. Input INT8 approximation remains. |
| a16exact | Four digits per A16 operand, two per W8. At most 16 native dots per attention tile and eight per dense projection tile; each native magnitude ≤16,384. | Software-only wide reconstruction. Wide integer result subsequently converts to the model's arithmetic format. Conservative staging/merge estimates; scale calibration, exact tensor layout and wide SIMD cost need hardware validation. |

The reference suite retains failed accuracy policies as experimental outcomes. The default checker still raises on tolerance failures. In comparison mode float-reference drift and nonfinite values remain fatal. Naive and streaming native outputs are checked independently; row-blocked causal math is checked on small B1/B16 tensors. Its full-checkpoint accuracy is still unmeasured.

## Four-device accounting

DP assigns disjoint users and replicates weights. TP splits Q and KV heads, fused QKV and gate/up columns, and output/down reduction dimensions; each rank retains full residual width. Two row-projection reductions per layer traverse the host. PP splits contiguous layers across stages. For TP+PP, one representative uploads each boundary state and the host forwards it to all next-stage ranks. Embedding and vocabulary weights are replicated where used. A synchronized greedy output needs one representative response per replica. Only predicted IDs and the next input cross the host cut in pure DP. Pipeline microbatches of 1, 2 or 4 groups use first-traversal time plus each additional slowest-stage service interval. A generation step waits for the synchronized batch. Continuous asynchronous serving would have another critical path.

For TP4, B16 decode, a hidden state contains 131,072 bytes. Across 64 reductions per step, each direction carries 33,554,432 bytes of activation payload plus token IDs. For a compute-only step C and target fraction f=0.7, the total time budget is C/f. The serial model pays compute, startup, host service and both directions in succession. The chunked model uses the larger direction; the optimistic overlap model also overlaps transfers with compute. The finite-host variant charges reduction memory service and dispatch. If fixed costs exceed budget, bandwidth cannot recover the target.

**Two distinct ceilings.** Dividing ideal work and DRAM traffic across four devices gives 21.213 tokens/s/user. It assumes ideal work balance, one useful MAC per cell per cycle, complete compute/DMA overlap, W8 weights read once and grouped KV reuse. Its 70% target is 14.849 tokens/s/user; host-star transport alone requires at least 0.498 GB/s **each direction** with full overlap, or 3.322 GB/s **shared** in the serialized case. The timed TP4 mapping reaches only 2.259 tokens/s/user even with a free link. It therefore cannot reach 70% of the ideal roof at any host bandwidth. For 70% of **its own** 2.259 ceiling, the serialized host model requires 0.354 GB/s shared. TTFT for microbatched configurations is time for the last first-token in the batch. Lower host-transfer bounds assume chunk overlap and instantaneous reduction unless configured otherwise.

## Open issues and the evidence that would resolve them

| Priority | Question for Arago | Discriminating test or measurement |
|---|---|---|
| 1 | What reaches the INT16 output when the exact dot overflows? Can software choose scales or obtain a wide result? | Send all-127 operands, signed cancellation, and half-way rounding ties at K=128 and 256. Compare target outputs with the six software policies before accepting any one-pass W8/A16 speed estimate. |
| 2 | Can the control CPU post a dependent command early? How many commands are outstanding per engine? | Timestamp two back-to-back DMA descriptors and two independent tiles on each array. Compare late posting and queued traces. Sweep queue depth on hardware once its interface is known. |
| 3 | What are the array initiation interval, fill latency and readout overlap rules? | Issue 1, 2 and 32 identical tiles at K=1, 128 and 256. Compare first completion and steady-state stride separately. |
| 4 | Can reads and writes proceed simultaneously, and does first-byte latency recur at every descriptor or burst? | Sweep isolated and concurrent reads/writes of 64 B, 1 KiB and 64 KiB. Compare slope and intercept with the independent DMA-link and duplex tests. |
| 5 | What host-visible arrival/response boundary and transport should the four-device answer use? | Measure upload, download, simultaneous transfers, host reduction and command dispatch separately. Re-run the 70% crossing with measured parameters. |
| 6 | Does the chosen policy preserve model quality at the requested W8/A16 precision? | Run held-out prompts through all 32 layers using a calibrated W8 reference, compare logits, perplexity and generated IDs; repeat with row-blocked and streaming schedules. Current checkpoint checks cover layer 0. |

## Managing experiments

Keep the assignment hardware in one versioned baseline JSON preset. In the versioned manifest specify a workload, an algorithm and named overrides. Keep functional precision and timing linked by explicit policy names. Use single changes for attribution and crossed cases for interactions. Preserve input hashes, measured counters, units, failure outcomes and the baseline. Rebuild the C++ target before comparing. Re-run the same trace to check byte-identical reports.

- SCALE-Sim separates architecture configuration from workload topology and emits compute/bandwidth reports. We apply that separation to hardware JSON, request CSVs and the results directory. [SCALE-Sim run inputs](https://github.com/scalesim-project/SCALE-Sim#tool-inputs)
- Timeloop distinguishes the problem, architecture, constraints and mapper. That is why tile layout and parallel placement are experiment choices even when device constants are fixed. [Timeloop input formats](https://timeloop.csail.mit.edu/v4/input-formats)
- gem5 illustrates configurable architectural components and command-line experiments. ASTRA-sim treats communication and network behavior as explicit simulation concerns. Here the host-star model is a separate layer with directional byte oracles. [gem5 configuration examples](https://www.gem5.org/documentation/learning_gem5/part1/example_configs/) · [ASTRA-sim](https://github.com/astra-sim/astra-sim)

## Reproduce

Follow the [build guide](arago.md). Set `ARAGO_TORCH_PYTHON` to the interpreter with PyTorch to enable the numerical and replay CTests. Run the following from the repository root:

```bash
cmake -S . -B build/arago -DONNXIM_ARAGO_ONLY=ON -DARAGO_TORCH_PYTHON="$(command -v python)"
cmake --build build/arago -j4
ctest --test-dir build/arago --output-on-failure
python3 tools/arago/architecture_study.py --output example/arago/results/architecture-study.json
python tools/arago/replay_panels.py --device cuda
python tests/arago/test_precision.py cuda
for b in 1 16; do
  for p in scaled fixedshift saturate bounded exact4 a16exact; do
    python tools/arago/check_attention.py --checkpoint "$MODEL_DIR" --device cuda \
      --batch "$b" --prompt 2048 --context 2176 --precision-policy "$p" --report-accuracy \
      --json "example/arago/results/precision/$b-$p.json"
  done
done
python3 tools/arago/render_study.py
python3 tools/arago/case_elasticity.py
python3 tools/arago/four_case_metrics.py
python3 tools/arago/recommendation_study.py
python3 tools/arago/report_cases.py
```

The timing runner generates 32 design cases per batch, the full 2×3×6 workload/schedule/precision grid and 60 placement/microbatch/host combinations. Tests use smaller cases for independent oracles, plus the full B16 compatibility fixture. Hardware and layer tests live in [test_hardware.cc](../tests/arago/test_hardware.cc) and [test_layer.cc](../tests/arago/test_layer.cc). The [architecture tests](../tests/arago/test_architecture.py) check each sweep direction, invariant payloads and capacity rejection. Full-rate results are estimates conditional on these scheduling decisions. The suite does not exhaust arbitrary future chip designs.
