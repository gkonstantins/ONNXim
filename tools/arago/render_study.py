"""Render the comparison tables from saved timing and numerical measurements."""
import json
from pathlib import Path
from architecture_study import ROOT

def main():
    results = ROOT/"example/arago/results"
    study=json.loads((results/"architecture-study.json").read_text())
    lines=["# Architecture comparison","",
           "Each device result uses 2,048 prompt tokens and 256 outputs per user. TTFT includes embedding, 32 decoder blocks, final norm, vocabulary head and greedy selection on a warm device. Decode uses the mean over all 255 cached positions. A synchronized batch emits one token per user per step. [Assumptions and requirements](coverage.md) · [Raw results](../example/arago/results/architecture-study.json)",""]
    for batch,cases in study["single"].items():
        lines.extend([f"## Batch {batch}, one variable at a time","",
            "| Case | Device TTFT (s) | Decode (ms/step) | Tokens/s/user | Speedup vs baseline | Prefill / decode attention elasticity |",
            "|---|---:|---:|---:|---:|---:|"])
        for name,v in cases.items():
            elasticity=(f"{v['prefill_elasticity']:.3f} / {v['decode_elasticity']:.3f}"
                        if "prefill_elasticity" in v else "—")
            lines.append(f"| {name} | {v['ttft_device_s']:.3f} | {v['decode_ms']:.3f} | {v['tokens_s_user']:.4f} | {v['decode_speedup']:.4f}× | {elasticity} |")
        lines.append("")
    lines.extend(["## What changes inside attention","",
        "Batch 16, one decoder block. DRAM traffic is read plus write in decimal GB per step. Utilization is the sum of engine busy cycles divided by elapsed attention cycles across the phase. DMA includes first-byte wait and setup. Array service includes input-port stalls. Engines can overlap.","",
        "| Mapping | Phase | DRAM GB/step | DMA busy | 16×16 busy | 32×16 busy |",
        "|---|---|---:|---:|---:|---:|"])
    for r in study["factorial"]:
        if r["batch"] != 16 or r["policy"] != "scaled":
            continue
        for phase,c in r["attention_counters"].items():
            u=c["utilization"]
            gb=(c["dram_read_bytes"]+c["dram_write_bytes"])/c["steps"]/1e9
            lines.append(f"| {r['schedule']} | {phase} | {gb:.4f} | {u['dma_busy_cycles']:.1%} | {u['array16_compute_cycles']:.1%} | {u['array32_compute_cycles']:.1%} |")
    lines.append("")
    lines.extend(["## Batch × attention × precision","","Costs below are per-user tokens/s for the entire model. The precision modes share the same one-pass timing where they transfer the same widths and perform the same number of native operations. The assumed scaling hardware and the software digit decompositions have different numerical contracts.","",
        "| B | Precision | Naive | SRAM row-blocked | FlashAttention | CUDA FlashAttention relative RMS, prefill / decode | Thresholds |",
        "|---:|---|---:|---:|---:|---:|---|"])
    for batch in (1,16):
        for policy in sorted({r["policy"] for r in study["factorial"]}):
            cells={r["schedule"]:r["tokens_s_user"] for r in study["factorial"] if r["batch"]==batch and r["policy"]==policy}
            q=json.loads((results/f"precision/{batch}-{policy}.json").read_text())
            err=[100*q["errors"][phase]["native_online_vs_float_dense"]["relative_rms"] for phase in ("prefill","decode")]
            status="pass" if q["result"]=="PASS" else "outside tolerance"
            lines.append(f"| {batch} | {policy} | {cells['naive']:.4f} | {cells['rowblocked']:.4f} | {cells['flashattention']:.4f} | {err[0]:.4f}% / {err[1]:.4f}% | {status} |")
    lines.extend(["","## Four accelerators and one host","",
        "Local shard service comes from ONNXim. All rates are per user for 16 synchronized users. The configured link is 16 decimal GB/s, aggregate across the host cut. For duplex cases GB/s is per direction. The 70% denominator in this table is each placement’s own compute-only limit. Microbatches share each stage in order. TTFT is the latency until all 16 first tokens are available.","",
        "| DP / TP / PP | Microbatches | Host model | Device limit (tokens/s/user) | TTFT with link (s) | Tokens/s/user with link | GB/s for 70% |",
        "|---|---:|---|---:|---:|---:|---:|"])
    for r in study["four"]:
        bw=r["minimum_GB_s_for_70pct"]
        text="unreachable" if bw is None else f"{bw:.7g}"
        lines.append(f"| {r['dp']} / {r['tp']} / {r['pp']} | {r['microbatches']} | {r['network']} | {r['device_limit_tokens_s_user']:.4f} | {r['ttft_at_link_s']:.3f} | {r['decode_tokens_s_user_at_link']:.4f} | {text} |")
    ideal=study["ideal_four"]
    lines.extend(["","### Ideal four-device resource ceiling","",
        f"Perfect partition, ideal utilization and unbounded issue give {ideal['tokens_s_user']:.3f} tokens/s/user ({ideal['tokens_s_batch']:.2f} aggregate). At 70% of this ceiling, TP4 host-star traffic requires at least {ideal['GB_s_for_70pct_ideal_overlap']:.3f} GB/s **each way** with ideal overlap, or {ideal['GB_s_for_70pct_serial']:.3f} GB/s of **shared** payload with serialized upload/download and compute. Even infinite host bandwidth leaves the tested TP4 schedule below 70% of this silicon roof.",""])
    (ROOT/"docs/architecture-comparison.md").write_text("\n".join(lines))

if __name__=="__main__":
    main()
