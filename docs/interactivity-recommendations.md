# Improving interactivity for 16 simultaneous Llama users

With a 2,048-token prompt, the warm one-device FlashAttention model takes 670.25 s to produce the first token for all 16 users. Then each user receives one further token every 1.526 s on average over the remaining 255 steps. All recommendations below use the same request trace. Decode improvements are changes in whole-model tokens/s per user; a 10% rate improvement corresponds to a 9.09% shorter step.

## One device

Rows 1–5 compare an alternate implementation with capabilities already in the reference FlashAttention mapping. Row 6 is an additional software mapping on the fixed device. Rows 7–15 quantify proposed command or silicon changes. Comparisons vary one policy per pair; gains interact and cannot be added.

| # | Change | Before → after, tokens/s/user | Rate gain | What to verify |
|---:|---|---:|---:|---|
| 1 | Share each K/V panel across its four query heads | 0.3637 → 0.6552 | +80.13% | Grouping also fills otherwise padded array rows. Check KV reads and native tile counts. |
| 2 | Stream softmax and accumulate PV in SRAM | 0.5676 → 0.6552 | +15.44% | Eliminates score/probability spills. Check per-layer attention bytes and whole-step time separately. |
| 3 | Overlap output drain with the next array feed | 0.5910 → 0.6552 | +10.87% | Uses the pipeline described in the assignment. Verify first tile latency and steady-state tile spacing. |
| 4 | Increase the KV tile from 64 to 128 keys | 0.6015 → 0.6552 | +8.94% | Fewer panels mean fewer DMA descriptors, commands and online updates. Recheck the SRAM high-water mark. |
| 5 | Double-buffer the KV panels | 0.6070 → 0.6552 | +7.94% | Prefetch the next block while the current block executes. Reuse a slot only after its last consumer. |
| 6 | Transpose small dense decode projections | 0.6552 → 0.7056 | +7.69% | Both arrays can now consume output rows. The model charges the extra transpose traffic. |
| 7 | Queue commands while engines are busy | 0.6552 → 0.8266 | +26.15% | Retains the stated delivery latency, allows earlier posting. Measure queue capacity, issue rate and dependencies. |
| 8 | Raise both DRAM–DMA and DMA–SRAM bandwidths from 64 to 128 B/cycle | 0.6552 → 0.6899 | +5.29% | The slower link limits a transfer. Increasing either link alone gave zero gain in its one-variable sweep. |
| 9 | Increase array MAC service rate by 10% | 0.6552 → 0.6794 | +3.70% | Modeled throughput change at fixed clock. Silicon cost and power are open. |
| 10 | Shorten array command delivery from 100 to 90 cycles | 0.6552 → 0.6682 | +1.98% | Targets the most elastic explicit PDF parameter for baseline attention. Check the gain again with early posting enabled. |
| 11 | Shorten SIMD command delivery from 300 to 270 cycles | 0.6552 → 0.6612 | +0.91% | Benefits the repeated small pack, scale and reduction passes. The model already assumes unlimited SIMD arithmetic. |
| 12 | Increase each array output port from 8 to 16 B/cycle | 0.6552 → 0.6604 | +0.79% | Accelerates native INT16 readout. The small total gain favors testing queues and mapping first. |
| 13 | Shorten DMA configuration latency from 200 to 180 cycles | 0.6552 → 0.6595 | +0.65% | Every descriptor pays this setup. Measure descriptor delivery separately from DRAM time to first byte. |
| 14 | Reduce DRAM read time to first byte from 200 to 180 cycles | 0.6552 → 0.6593 | +0.62% | Moves the intercept for short reads. The sustained link payload rate remains fixed. |
| 15 | Widen the SRAM–SIMD port from 128 to 141 B/cycle | 0.6552 → 0.6593 | +0.63% | A roughly 10% modeled effective-bandwidth increase. Test pack and online-softmax passes before selecting a physical bus width. |

Per-tile scaling is assumed for these comparisons. A production decision also needs all-layer W8/A16 accuracy. Naive saturation produced about 65% relative RMS error on the B16 checkpoint prefill attention test. Treat it as a failed numerical alternative even though its simulated throughput matches the scaled policy. [Precision results and checks](architecture-comparison.md#batch--attention--precision)

## Four accelerators sharing one host

The rates below model a host star and resident weights. A 16 GB/s serialized link is shared across both directions and all devices. Rank-local tensor shapes and device service are simulated. The host transfer time is analytical.

16. **Use four-way tensor parallelism for interactive decode on an ample shared link.** At 16 GB/s, splitting the attention heads and dense projections gives 2.2373 tokens/s/user, versus 0.8390 for four independent four-user replicas. This improves decode rate by 166.7%. The tradeoff is a longer TTFT, 179.6 s versus 167.8 s, and 64 host-mediated reductions per decode step.

17. **Adapt placement to measured shared-link bandwidth.** The DP2×TP2 hybrid reduces transfer from 67.11 to 33.55 MB/step. DP4 sends just 128 bytes of IDs per decode step. Among all tested placements, the best decode choice changes at the two crossovers below. Measure effective payload bandwidth after other devices have claimed their share.

| Effective shared bandwidth, GB/s | Fastest placement (DP × TP × PP) | Device-only step, ms | Host payload/step |
|---:|---|---:|---:|
| 0.00000–0.06704 | 4 × 1 × 1 | 1191.91 | 128 B |
| 0.06704–0.13498 | 2 × 2 × 1 | 691.37 | 33,554,624 B |
| above 0.13498 | 1 × 4 × 1 | 442.77 | 67,109,184 B |

For each placement the time in ns is device service plus payload bytes divided by shared bandwidth in GB/s. At a crossover both placements tie. These thresholds omit startup, CPU reduction time, and overlap; the effective bandwidth that selects the fastest placement answers a different question from the bandwidth needed for 70% of a specific ceiling.

18. **If layer pipelining is required, use phase-specific microbatch counts.** With PP4, four microbatches reduce the last-user TTFT from 670.4 to 294.0 s. During decode, one microbatch yields 0.6552 tokens/s/user compared with 0.4429 at four microbatches, a 47.9% rate improvement. Smaller decode microbatches repeat weight reads and the vocabulary head. The two phases are measured independently. Changing the grouping between phases requires a scheduler and a KV layout that preserves user ownership.

19. **Chunk host reductions to use independent upload/download directions when available.** TP4 gives 2.2373 tokens/s/user at 16 GB/s shared serial versus 2.2479 with 16 GB/s per direction and chunked transport. The latter is a modeled 0.47% gain. This comparison grants full duplex additional simultaneous physical bandwidth. With a shared fabric of fixed aggregate throughput the same gain does not follow.

20. **Pipeline TP reductions with independent device work where data dependencies allow.** The ideal chunk-overlap bound reaches 2.2585 tokens/s/user at 16 GB/s per direction, only 0.47% above the chunked model. Before implementing overlap, trace the completion of the output/down projections, synchronization and earliest dependent consumer. Full overlap is an upper bound.

The strongest software experiments on this trace are GQA panel reuse and decode remapping. For a hardware change, test early command posting first, then both DMA links together. For four devices, establish effective host throughput and choose placement before increasing link width. A bandwidth upgrade has limited benefit once the device dominates the step.

## Reproduce

The table is generated from [raw experiment results](../example/arago/results/architecture-study.json). The simulator, workload definitions, and CUDA numerical checks are described in the [coverage report](coverage.md).

```bash
python3 tools/arago/recommendation_study.py
python3 tests/arago/test_recommendation_study.py
```
