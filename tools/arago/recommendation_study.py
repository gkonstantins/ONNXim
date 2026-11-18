"""Render paired decode comparisons and solve the serial-link placement crossover."""
import argparse
from itertools import combinations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "example/arago/results/architecture-study.json"
DOCUMENT = ROOT / "docs/interactivity-recommendations.md"

# Labels describe the actual difference between two previously executed traces.
PAIRS = [
    ("Share each K/V panel across its four query heads", "independent_heads", "baseline",
     "Grouping also fills otherwise padded array rows. Check KV reads and native tile counts."),
    ("Stream softmax and accumulate PV in SRAM", "naive", "baseline",
     "Eliminates score/probability spills. Check per-layer attention bytes and whole-step time separately."),
    ("Overlap output drain with the next array feed", "overlap_drain", "baseline",
     "Uses the pipeline described in the assignment. Verify first tile latency and steady-state tile spacing."),
    ("Increase the KV tile from 64 to 128 keys", "key_tile", "baseline",
     "Fewer panels mean fewer DMA descriptors, commands and online updates. Recheck the SRAM high-water mark."),
    ("Double-buffer the KV panels", "double_buffer", "baseline",
     "Prefetch the next block while the current block executes. Reuse a slot only after its last consumer."),
    ("Transpose small dense decode projections", "baseline", "transpose_decode",
     "Both arrays can now consume output rows. The model charges the extra transpose traffic."),
    ("Queue commands while engines are busy", "baseline", "queued_delivery",
     "Retains the stated delivery latency, allows earlier posting. Measure queue capacity, issue rate and dependencies."),
    ("Raise both DRAM–DMA and DMA–SRAM bandwidths from 64 to 128 B/cycle", "baseline", "double_both_dma_links",
     "The slower link limits a transfer. Increasing either link alone gave zero gain in its one-variable sweep."),
    ("Increase array MAC service rate by 10%", "baseline", "sweep_mac_rate",
     "Modeled throughput change at fixed clock. Silicon cost and power are open."),
    ("Shorten array command delivery from 100 to 90 cycles", "baseline", "sweep_array_launch",
     "Targets the most elastic explicit PDF parameter for baseline attention. Check the gain again with early posting enabled."),
    ("Shorten SIMD command delivery from 300 to 270 cycles", "baseline", "sweep_vector_launch",
     "Benefits the repeated small pack, scale and reduction passes. The model already assumes unlimited SIMD arithmetic."),
    ("Increase each array output port from 8 to 16 B/cycle", "baseline", "double_output_port",
     "Accelerates native INT16 readout. The small total gain favors testing queues and mapping first."),
    ("Shorten DMA configuration latency from 200 to 180 cycles", "baseline", "sweep_dma_setup",
     "Every descriptor pays this setup. Measure descriptor delivery separately from DRAM time to first byte."),
    ("Reduce DRAM read time to first byte from 200 to 180 cycles", "baseline", "sweep_dram_read_start",
     "Moves the intercept for short reads. The sustained link payload rate remains fixed."),
    ("Widen the SRAM–SIMD port from 128 to 141 B/cycle", "baseline", "sweep_vector_bytes_per_cycle",
     "A roughly 10% modeled effective-bandwidth increase. Test pack and online-softmax passes before selecting a physical bus width."),
]


def curve(row):
    """Serialized per-batch service in ns is C + D / bandwidth in decimal GB/s."""
    return 1e9 / row["device_limit_tokens_s_user"], row["up_B_step"] + row["down_B_step"]


def service(row, bandwidth):
    compute, payload = curve(row)
    return compute + payload / bandwidth


def winner_ranges(rows):
    positive = set()
    for a, b in combinations(rows, 2):
        ca, da = curve(a)
        cb, db = curve(b)
        if ca != cb:
            crossing = (db - da) / (ca - cb)
            if crossing > 0:
                positive.add(crossing)
    bounds = [0., *sorted(positive), float("inf")]
    intervals = []
    for lower, upper in zip(bounds, bounds[1:]):
        probe = (lower + upper) / 2 if upper != float("inf") else 2 * lower + 1
        winner = min(rows, key=lambda row: service(row, probe))
        if intervals and intervals[-1][2] is winner:
            intervals[-1] = (intervals[-1][0], upper, winner)
        else:
            intervals.append((lower, upper, winner))
    return intervals


def render(study):
    single = study["single"]["16"]
    four = study["four"]
    def row(dp, tp, pp, micro=1, host="serial"):
        return next(r for r in four if (r["dp"], r["tp"], r["pp"], r["microbatches"], r["network"])
                    == (dp, tp, pp, micro, host))
    lines = [
        "# Improving interactivity for 16 simultaneous Llama users", "",
        "With a 2,048-token prompt, the warm one-device FlashAttention model takes "
        f"{single['baseline']['ttft_device_s']:.2f} s to produce the first token for all 16 users. "
        f"Then each user receives one further token every {single['baseline']['decode_ms']/1000:.3f} s on average "
        "over the remaining 255 steps. All recommendations below use the same request trace. Decode improvements "
        "are changes in whole-model tokens/s per user; a 10% rate improvement corresponds to a 9.09% shorter step.",
        "", "## One device", "",
        "Rows 1–5 compare an alternate implementation with capabilities already in the reference FlashAttention mapping. "
        "Row 6 is an additional software mapping on the fixed device. Rows 7–15 quantify proposed command or silicon changes. "
        "Comparisons vary one policy per pair; gains interact and cannot be added.", "",
        "| # | Change | Before → after, tokens/s/user | Rate gain | What to verify |",
        "|---:|---|---:|---:|---|",
    ]
    for i, (name, before, after, evidence) in enumerate(PAIRS, 1):
        a = single[before]["tokens_s_user"]
        b = single[after]["tokens_s_user"]
        lines.append(f"| {i} | {name} | {a:.4f} → {b:.4f} | +{(b/a-1)*100:.2f}% | {evidence} |")
    dp4, tp4, hybrid = row(4,1,1), row(1,4,1), row(2,2,1)
    pp_one, pp_four = row(1,1,4), row(1,1,4,4)
    duplex, overlap = row(1,4,1,host="chunked"), row(1,4,1,host="overlap_bound")
    rate = "decode_tokens_s_user_at_link"
    lines.extend([
        "", "Per-tile scaling is assumed for these comparisons. A production decision also needs all-layer W8/A16 accuracy. "
        "Naive saturation produced about 65% relative RMS error on the B16 checkpoint prefill attention test. "
        "Treat it as a failed numerical alternative even though its simulated throughput matches the scaled policy. "
        "[Precision results and checks](architecture-comparison.md#batch--attention--precision)", "",
        "## Four accelerators sharing one host", "",
        "The rates below model a host star and resident weights. A 16 GB/s serialized link is shared across both directions and all devices. "
        "Rank-local tensor shapes and device service are simulated. The host transfer time is analytical.", "",
        f"16. **Use four-way tensor parallelism for interactive decode on an ample shared link.** At 16 GB/s, splitting "
        f"the attention heads and dense projections gives {tp4[rate]:.4f} tokens/s/user, versus {dp4[rate]:.4f} "
        f"for four independent four-user replicas. This improves decode rate by {(tp4[rate]/dp4[rate]-1)*100:.1f}%. "
        f"The tradeoff is a longer TTFT, {tp4['ttft_at_link_s']:.1f} s versus {dp4['ttft_at_link_s']:.1f} s, "
        "and 64 host-mediated reductions per decode step.", "",
        f"17. **Adapt placement to measured shared-link bandwidth.** The DP2×TP2 hybrid reduces transfer from "
        f"{curve(tp4)[1]/1e6:.2f} to {curve(hybrid)[1]/1e6:.2f} MB/step. DP4 sends just "
        f"{curve(dp4)[1]} bytes of IDs per decode step. Among all tested placements, the best decode choice changes at the "
        "two crossovers below. Measure effective payload bandwidth after other devices have claimed their share.", "",
        "| Effective shared bandwidth, GB/s | Fastest placement (DP × TP × PP) | Device-only step, ms | Host payload/step |",
        "|---:|---|---:|---:|",
    ])
    for lower, upper, r in winner_ranges([r for r in four if r["network"]=="serial"]):
        band = (f"above {lower:.5f}" if upper == float("inf") else f"{lower:.5f}–{upper:.5f}")
        cycles, size = curve(r)
        lines.append(f"| {band} | {r['dp']} × {r['tp']} × {r['pp']} | {cycles/1e6:.2f} | {size:,} B |")
    lines.extend(["", "For each placement the time in ns is device service plus payload bytes divided by shared bandwidth in GB/s. "
        "At a crossover both placements tie. These thresholds omit startup, CPU reduction time, and overlap; the effective "
        "bandwidth that selects the fastest placement answers a different question from the bandwidth needed for 70% of a specific ceiling.", "",
        f"18. **If layer pipelining is required, use phase-specific microbatch counts.** With PP4, four microbatches reduce "
        f"the last-user TTFT from {pp_one['ttft_at_link_s']:.1f} to {pp_four['ttft_at_link_s']:.1f} s. "
        f"During decode, one microbatch yields {pp_one[rate]:.4f} tokens/s/user compared with {pp_four[rate]:.4f} "
        f"at four microbatches, a {(pp_one[rate]/pp_four[rate]-1)*100:.1f}% rate improvement. "
        "Smaller decode microbatches repeat weight reads and the vocabulary head. The two phases are measured independently. "
        "Changing the grouping between phases requires a scheduler and a KV layout that preserves user ownership.", "",
        f"19. **Chunk host reductions to use independent upload/download directions when available.** TP4 gives "
        f"{tp4[rate]:.4f} tokens/s/user at 16 GB/s shared serial versus {duplex[rate]:.4f} with 16 GB/s per direction and "
        f"chunked transport. The latter is a modeled {(duplex[rate]/tp4[rate]-1)*100:.2f}% gain. This comparison grants full duplex "
        "additional simultaneous physical bandwidth. With a shared fabric of fixed aggregate throughput the same gain does not follow.", "",
        f"20. **Pipeline TP reductions with independent device work where data dependencies allow.** The ideal chunk-overlap "
        f"bound reaches {overlap[rate]:.4f} tokens/s/user at 16 GB/s per direction, only "
        f"{(overlap[rate]/duplex[rate]-1)*100:.2f}% above the chunked model. Before implementing overlap, trace the completion of the "
        "output/down projections, synchronization and earliest dependent consumer. Full overlap is an upper bound.", "",
        "The strongest software experiments on this trace are GQA panel reuse and decode remapping. For a hardware change, "
        "test early command posting first, then both DMA links together. For four devices, establish effective host throughput "
        "and choose placement before increasing link width. A bandwidth upgrade has limited benefit once the device dominates the step.",
        "", "## Reproduce", "",
        "The table is generated from [raw experiment results](../example/arago/results/architecture-study.json). "
        "The simulator, workload definitions, and CUDA numerical checks are described in the [coverage report](coverage.md).", "",
        "```bash", "python3 tools/arago/recommendation_study.py",
        "python3 tests/arago/test_recommendation_study.py", "```", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=DOCUMENT)
    args = parser.parse_args()
    args.output.write_text(render(json.loads(args.study.read_text())))
