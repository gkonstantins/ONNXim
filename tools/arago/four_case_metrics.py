"""Generate per-case four-device service, host-link sensitivity and bandwidth thresholds."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from statistics import mean

from architecture_study import (
    DEFAULT_BINARY, DESIGN, ROOT, host_payload, host_service, ideal_roof,
    pipeline_cycles, simulate, transfer_work,
)

RESULT = ROOT / "example/arago/results/four-case-metrics.json"
REPORT = ROOT / "docs/four-case-metrics.md"
USERS, PROMPT, OUTPUTS = 16, 2048, 256
FRACTION = 0.70
LINK_PROBE = 1.10


def host_timing(queries, placement, options):
    dp, tp, pp, micro = placement
    up, down, phases = host_payload(USERS, queries, dp, tp, pp)
    return {
        "up_B": up, "down_B": down, "phases": phases * micro,
        "link_B": transfer_work(up, down, options["duplex"]),
        "startup_ns": phases * micro * options["startup_ns"],
        "host_ns": host_service(USERS, queries, tp, options),
        "overlap": options["overlap"],
    }


def exposed_link_ns(cycles, traffic, rate):
    """Mean uncovered transfer time, evaluated at each step before averaging."""
    transfer = traffic["link_B"] / rate
    if traffic["overlap"]:
        return mean(max(transfer - step, 0.0) for step in cycles)
    return transfer


def latency_ns(cycles, traffic, rate):
    return mean(cycles) + traffic["startup_ns"] + traffic["host_ns"] + exposed_link_ns(cycles, traffic, rate)


def minimum_bandwidth(cycles, traffic, target_tokens_s):
    """Smallest GB/s at the requested absolute rate; None if finite link cannot reach it."""
    budget = 1e9 / target_tokens_s - traffic["startup_ns"] - traffic["host_ns"]
    device = mean(cycles)
    if budget < device:
        return None
    payload = traffic["link_B"]
    if not traffic["overlap"]:
        return payload / (budget - device) if budget > device else None
    # At or above this upper bound, the link is entirely hidden even on the fastest step.
    low, high = 0.0, payload / min(cycles)
    for _ in range(55):
        middle = (low + high) / 2
        if device + exposed_link_ns(cycles, traffic, middle) > budget:
            low = middle
        else:
            high = middle
    return high


def case_metrics(placement, options, network, prefill, decode, peak, best):
    dp, tp, pp, micro = placement
    rate = options["GB_s"]
    prompt = host_timing(PROMPT, placement, options)
    cached = host_timing(1, placement, options)
    ttft = latency_ns([prefill], prompt, rate) / 1e9
    tbt = latency_ns(decode, cached, rate)
    exposed = exposed_link_ns(decode, cached, rate)
    service = mean(decode)
    mapped = 1e9/service
    tbt_probe = latency_ns(decode, cached, rate * LINK_PROBE)
    limit = service + cached["host_ns"] + cached["startup_ns"]
    return {
        "dp": dp, "tp": tp, "pp": pp, "microbatches": micro, "network": network,
        "host_GB_s": rate, "bandwidth_basis": "each direction" if options["duplex"] else "shared",
        "ttft_s": ttft, "mean_tbt_ms": tbt/1e6, "tokens_s_user": 1e9/tbt,
        "tokens_s_aggregate": USERS*1e9/tbt, "completion_256_s": ttft + len(decode)*tbt/1e9,
        "mapped_ceiling_tokens_s_user": mapped, "unbounded_link_tokens_s_user": 1e9/limit,
        "local_prefill_s": prefill/1e9, "mean_local_decode_ms": service/1e6,
        "mean_exposed_transfer_ms": exposed/1e6, "wire_service_ms": cached["link_B"]/rate/1e6,
        "host_fixed_ms": cached["host_ns"]/1e6, "link_startup_ms": cached["startup_ns"]/1e6,
        "decode_upload_B": cached["up_B"], "decode_download_B": cached["down_B"],
        "decode_host_phases": cached["phases"], "prefill_upload_B": prompt["up_B"],
        "prefill_download_B": prompt["down_B"],
        "link_time_elasticity": (tbt_probe / tbt - 1)/(LINK_PROBE-1),
        "GB_s_70pct_own_mapped": minimum_bandwidth(decode,cached,FRACTION*mapped),
        "GB_s_70pct_best_mapped": minimum_bandwidth(decode,cached,FRACTION*best),
        "GB_s_70pct_ideal": minimum_bandwidth(decode,cached,FRACTION*peak),
    }


def build(binary, manifest):
    settings = json.loads(manifest.read_text())
    roof = ideal_roof(USERS, 4)
    service = {}
    with tempfile.TemporaryDirectory() as tmp:
        for dp,tp,pp in settings["placements"]:
            if dp*tp*pp != 4:
                raise ValueError("placement must use four devices")
            for micro in settings["pipeline_microbatches"] if pp > 1 else [1]:
                if USERS % (dp*micro):
                    raise ValueError("unequal microbatches")
                rows = simulate(binary, Path(tmp), USERS//(dp*micro), "flashattention",
                                {"tensor_shards":tp, "pipeline_stages":pp},PROMPT,OUTPUTS)
                service[(dp,tp,pp,micro)] = [pipeline_cycles(row,pp,micro) for row in rows]
    best = max(1e9/mean(cycles[1:]) for cycles in service.values())
    cases = [case_metrics(placement, opts, name, cycles[0], cycles[1:],roof["tokens_s_user"],best)
             for placement,cycles in service.items() for name,opts in settings["host_models"].items()]
    return {"workload":{"users":USERS,"prompt":PROMPT,"outputs":OUTPUTS},
            "ideal_resource_ceiling":roof, "best_mapped_ceiling_tokens_s_user":best,
            "target_fraction":FRACTION, "link_probe_ratio":LINK_PROBE,
            "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),"cases":cases}


def format_rate(value):
    return "—" if value is None else f"{value:.6g}"


def case_name(row):
    return f"{row['dp']}/{row['tp']}/{row['pp']} · {row['microbatches']} · {row['network']}"


def render(result):
    peak = result["ideal_resource_ceiling"]
    best = result["best_mapped_ceiling_tokens_s_user"]
    cases = result["cases"]
    lines = ["# Four accelerators on one host", "",
        "The 16 users start together with 2,048 tokens each. The first pass produces one token per user. "
        "The following 255 cached steps each produce 16 tokens. TTFT ends when the last of the 16 first "
        "tokens reaches the modeled host. Each device already holds its weights. Local stage service comes "
        "from a native ONNXim rank schedule for each visible context. Host transfers and reductions are analytical. "
        "[Raw per-case metrics](../example/arago/results/four-case-metrics.json)", "",
        "## Parallelism and bandwidth", "",
        "| DP/TP/PP | Work assigned to each rank |", "|---|---|",
        "| 4/1/1 | Four independent replicas each serve four users through 32 layers. Only IDs cross the host link. |",
        "| 1/4/1 | All users share four tensor shards. Each rank owns eight Q heads, two KV heads and a quarter of each block projection. Two host reductions per layer combine the row-sharded projections. |",
        "| 1/1/4 | Each rank owns eight consecutive layers. Activations pass through the host at three stage boundaries. |",
        "| 2/2/1 | Two replicas each serve eight users across two tensor shards. |",
        "| 2/1/2 | Two replicas each serve eight users through two 16-layer stages. |",
        "| 1/2/2 | Each of two 16-layer stages uses a pair of tensor shards. At the stage boundary a representative uploads the hidden state and the host forwards it to both next-stage ranks. |", "",
        "Pipeline experiments split a synchronized cohort into 1, 2 or 4 equal microbatches. Within each stage the microbatches serialize; stages can process different microbatches in parallel. The host width is an aggregate cut across all four devices. Full duplex quotes GB/s per direction; serialized transport quotes shared GB/s. Decoder blocks use W8/A16 traffic and the baseline scaled native return. Device vocabulary heads are replicated across tensor ranks.", "",
        f"The common four-device ideal ceiling is **{peak['tokens_s_user']:.3f} tokens/s/user** "
        f"(**{peak['tokens_s_batch']:.2f} aggregate**). It assumes 4 × 768 GMAC/s, 4 × 64 GB/s, one weight fetch, ideal KV reuse, perfect partition and resource overlap. "
        f"Reaching 70% means **{FRACTION*peak['tokens_s_user']:.3f} tokens/s/user**. On the hypothetical ideal TP4 mapping the host payload alone needs **{peak['GB_s_for_70pct_ideal_overlap']:.3f} GB/s per direction** under complete overlap, or **{peak['GB_s_for_70pct_serial']:.3f} GB/s shared** with serialized transfer and compute. "
        f"The fastest mapped device schedule is **{best:.3f} tokens/s/user**, so every tested schedule misses the ideal 70% target even at infinite host bandwidth.", "",
        f"The second practical comparison is 70% of the **fastest mapped schedule** ({FRACTION*best:.3f} tokens/s/user). The third comparison is 70% of **each row’s own device limit**. A dash means the requested target exceeds the modeled infinite-link rate. The own-limit and best-limit bandwidths include that row’s fixed startup and host costs.", "",
        "## All four-device cases", "",
        "Case names give DP/TP/PP, microbatch count, and host model. TBT is mean time between successive tokens for one user. Aggregate tokens/s sums all 16 users. The common ideal 70% bandwidth answer for every row is unreachable. Columns for the two attainable-target calculations use decimal GB/s.", "",
        "| Case | TTFT (s) | TBT (ms) | Tokens/s/user | Tokens/s total | Device limit /user | GB/s for 70% own | GB/s for 70% best |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in cases:
        lines.append(f"| {case_name(r)} | {r['ttft_s']:.3f} | {r['mean_tbt_ms']:.3f} | {r['tokens_s_user']:.4f} | {r['tokens_s_aggregate']:.3f} | {r['mapped_ceiling_tokens_s_user']:.4f} | {format_rate(r['GB_s_70pct_own_mapped'])} | {format_rate(r['GB_s_70pct_best_mapped'])} |")
    lines.extend(["", "## Host critical path", "",
        "At nominal 16 GB/s, serial pays both directions. Chunked allows simultaneous directional transfers; overlap_bound also permits complete transfer/device overlap. Serial_1us adds 1 µs per host communication phase. Finite_host adds host reduction at 50 GB/s plus 10 µs dispatch per step. These host options are assumptions. Each row below adds rank-local service, exposed link time, startup and host work to recover its TBT. For overlap, exposed link time is max(wire time − local time, 0) evaluated per cached step. The signed elasticity is the fractional change in full decode time when increasing host bandwidth by 10%, divided by 0.10. A value near zero means additional host bandwidth buys little under this scenario. The separate [per-case kernel sweep](case-metrics.md) identifies the most elastic accelerator parameter for each rank-local attention kernel.", "",
        "| Case | Local (ms) | Exposed link (ms) | Link startup (ms) | CPU service (ms) | Host bandwidth elasticity |",
        "|---|---:|---:|---:|---:|---:|"])
    for r in cases:
        lines.append(f"| {case_name(r)} | {r['mean_local_decode_ms']:.3f} | {r['mean_exposed_transfer_ms']:.7f} | {r['link_startup_ms']:.4f} | {r['host_fixed_ms']:.4f} | {r['link_time_elasticity']:.8f} |")
    indexed = {(r["dp"],r["tp"],r["pp"],r["microbatches"],r["network"]):r for r in cases}
    tp, dp, duplex, overlapped, pp1, pp4 = [indexed[k] for k in
        ((1,4,1,1,'serial'),(4,1,1,1,'serial'),(1,4,1,1,'chunked'),(1,4,1,1,'overlap_bound'),
         (1,1,4,1,'serial'),(1,1,4,4,'serial'))]
    lines.extend(["", "## Decisions supported by the comparisons", "",
        f"- **Favor tensor parallelism for steady decode with this host.** TP4 achieves {tp['tokens_s_user']:.3f} tokens/s/user versus {dp['tokens_s_user']:.3f} for DP4 at 16 GB/s shared, a {tp['tokens_s_user']/dp['tokens_s_user']:.2f}× gain. DP4 returns the synchronized first tokens sooner ({dp['ttft_s']:.1f} versus {tp['ttft_s']:.1f} s).",
        f"- **If four pipeline stages are required, choose the microbatch count for the latency goal.** Moving from four microbatches to one improves steady interactivity from {pp4['tokens_s_user']:.3f} to {pp1['tokens_s_user']:.3f}. Moving in the other direction improves TTFT from {pp1['ttft_s']:.1f} to {pp4['ttft_s']:.1f} s. The skinny decode projections lose utilization while prefilling can keep more stages busy.",
        f"- **Prioritize rank service over faster links at 16 GB/s for TP4.** Serial link exposure is {tp['mean_exposed_transfer_ms']:.3f} ms per token against {tp['mean_local_decode_ms']:.2f} ms of local work. Chunking the two directions raises rate to {duplex['tokens_s_user']:.3f}; ideal compute/transfer overlap reaches {overlapped['tokens_s_user']:.3f}. Host-mediated dependencies may prevent realizing that last bound.", "",
        "To reproduce, run `python3 tools/arago/four_case_metrics.py` after building the focused target. The program replays all 256 iterations for each placement/microbatch and solves each host crossing at its absolute tokens/s target. Controls and raw directional byte counts are in the JSON.", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--manifest", type=Path, default=DESIGN)
    parser.add_argument("--json", type=Path, default=RESULT)
    parser.add_argument("--markdown", type=Path, default=REPORT)
    args = parser.parse_args()
    result = build(args.simulator.resolve(), args.manifest)
    args.json.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    args.markdown.write_text(render(result))
