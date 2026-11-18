"""Join whole-workload timing to the paired local kernel sweeps for each design."""
import json

from architecture_study import ROOT

RESULTS = ROOT / "example/arago/results"
REPORT = ROOT / "docs/case-metrics.md"
DATA = RESULTS / "case-metrics.json"
LABELS = {
    "array_launch": "array command latency", "vector_launch": "SIMD command latency",
    "dma_setup": "DMA command latency", "dram_read_start": "DRAM read latency",
    "dram_write_start": "DRAM write latency", "mac_percent": "MAC throughput",
    "dma_bytes_per_cycle": "DRAM–DMA width", "sram_dma_rate": "DMA–SRAM width",
    "array16_input_bytes_per_cycle": "16×16 input width",
    "array32_input_bytes_per_cycle": "32×16 input width",
    "array_output_bytes_per_cycle": "array output width",
    "vector_bytes_per_cycle": "SIMD port width", "scratchpad_bytes": "SRAM capacity",
    "dram_bytes": "DRAM capacity",
}


def load(name):
    return json.loads((RESULTS/name).read_text())


def rates(ttft_s, mean_step_s, batch):
    if min(ttft_s, mean_step_s, batch) <= 0:
        raise ValueError("Expected positive elapsed times and batch size")
    return {"ttft_s":ttft_s, "tbt_s":mean_step_s, "tokens_s_user":1/mean_step_s,
            "tokens_s_aggregate":batch/mean_step_s, "last_token_s":ttft_s+255*mean_step_s}


def join(study, design, elastic, four):
    cases = []
    configs = {case["name"]:case for case in design["cases"]}

    def add_single(key, batch, schedule, policy, rec, overrides):
        profile = elastic["cases"][key]
        metrics = rates(rec["ttft_device_s"], rec["decode_ms"]/1000, batch)
        if abs(metrics["tokens_s_user"]-rec["tokens_s_user"]) > 1e-9:
            raise ValueError(f"Inconsistent decode rate for {key}")
        metrics["attention_share_prefill"] = 32*rec["prefill_attention_cycles"]/1e9/metrics["ttft_s"]
        metrics["attention_share_decode"] = 32*rec["decode_attention_mean_cycles"]/1e9/metrics["tbt_s"]
        counts = elastic["profiles"][profile]["baseline_cycles"]
        for phase in ("prefill","decode"):
            if counts[phase] != rec["attention_counters"][phase]["cycles"]:
                raise ValueError(f"Kernel sweep has a different baseline for {key} {phase}")
        cases.append({"case":key, "batch":batch, "schedule":schedule, "precision":policy, "overrides":overrides,
                      **metrics, "elasticity_profile":profile, "winner":elastic["profiles"][profile]["winner"]})

    for batch, records in study["single"].items():
        for name,rec in records.items():
            cfg = configs[name]
            add_single(f"single/{batch}/{name}",int(batch),cfg.get("attention","flashattention"),
                       cfg.get("override",{}).get("precision_policy","scaled"),rec,cfg.get("override",{}))
    for rec in study["factorial"]:
        b,s,p = rec["batch"],rec["schedule"],rec["policy"]
        add_single(f"factorial/{b}/{s}/{p}",b,s,p,rec,{"precision_policy":p})
    for row in four:
        dp,tp,pp,m = (row[k] for k in ("dp","tp","pp","microbatches"))
        key = f"four/{dp}/{tp}/{pp}/{m}"
        profile = elastic["cases"][key]
        cases.append({"case":f"{key}/{row['network']}", "batch":16, "schedule":"flashattention",
                      "precision":"scaled", "link_time_elasticity":row["link_time_elasticity"],
                      "mapped_ceiling_tokens_s_user":row["mapped_ceiling_tokens_s_user"],
                      **{k:row[k] for k in ("GB_s_70pct_own_mapped","GB_s_70pct_best_mapped","GB_s_70pct_ideal")},
                      **rates(row["ttft_s"],row["mean_tbt_ms"]/1000,16),
                      "elasticity_profile":profile, "winner":elastic["profiles"][profile]["winner"]})
    return cases


def winner_text(winner):
    names = ", ".join(LABELS.get(k,k) for k in winner["parameters"])
    return f"{names} {winner['absolute']:.3f}"


def render(cases):
    text = [
        "# First token, sustained generation and kernel bottleneck by case",
        "",
        "The simulated workload gives every user a 2,048-token prompt and generates 256 output tokens. "
        "With 16 simultaneous users, naive attention produces the first token at 686.92 s (11 min 27 s). "
        "FlashAttention produces it at 670.25 s (11 min 10 s). Once prefill ends, each user receives "
        "one token every 1.526 s on average with FlashAttention. The group receives 16 tokens per step, "
        "or 10.48 tokens/s. Mean time between tokens (TBT) covers the 255 cached steps. "
        "Token 256 arrives at 1,059.44 s (17 min 39 s) under this schedule.",
        "",
        "The earlier 309.25 s prefill figure divides about 237.51 trillion useful MACs by the ideal 768 GMAC/s. "
        "The scheduled model accounts for padded or underfilled tiles, data movement, command delivery and output drain. "
        "It also includes embedding and final-token endpoints.",
        "",
        "One device spends 34.73 s of the 670.25 s prefill in attention across 32 layers. "
        "The remaining 635.52 s cover projections, the feed-forward network, other block operations and endpoints. "
        "In decode, attention takes 0.411 s of each 1.526 s step. A kernel optimization therefore has much more scope "
        "to change decode than prefill in this mapping.",
        "",
        "Device TTFT starts with weights resident and all prompts admitted together. Four-device TTFT additionally "
        "includes the configured host transfers/service and covers the last user’s first token. Wall-clock loading, "
        "tokenization and streaming to the client are outside these times. Precision choices in the timing model "
        "are assumptions; layer-0 numerical accuracy is available in the comparison.",
        "",
        "[Four-device parallelism, ceilings and 70% bandwidth by case](four-case-metrics.md) · "
        "[Interactivity recommendations](interactivity-recommendations.md) · [Accuracy and utilization](architecture-comparison.md) · "
        "[Machine-readable case table](../example/arago/results/case-metrics.json) · "
        "[Paired perturbations and signed elasticities](../example/arago/results/case-elasticity.json)",
        "",
        "## Reading the bottleneck columns",
        "",
        "Each row has its own kernel sensitivity experiment. The PDF columns give the largest absolute time elasticity "
        "among the swept hardware parameters specified by Arago. The extended columns also include the assumed per-cell MAC throughput. "
        "Each phase has its own winner. An elasticity of 0.30 means a 10% parameter change in the tested direction changes "
        "kernel time by about 3% near that operating point. Winners apply to the selected attention algorithm and, for "
        "four devices, to the local shard. Whole-model time and host-link sensitivity have different denominators. "
        "The raw sweep file records each parameter, perturbation, and cycle count.",
        "",
        "An array command latency win means the control CPU issues enough small tiles for command delivery to remain "
        "on the critical path. A SIMD command latency win points to frequently issued vector or reduction passes. "
        "A port-width win reflects bytes on that port. A MAC throughput win reflects time in matrix service. "
        "A zero local slope can arise from another bottleneck, integer cycle rounding, or an allocation that already fits.",
    ]
    text.extend(["", "The saturation and bounded-input modes failed the layer-0 accuracy checks for both batch sizes. "
                 "Their timing rows show how quickly a numerically unacceptable result could be generated. "
                 "The full-checkpoint row-blocked schedule still needs an accuracy run."])
    for prefix,title in (("single/1/","Batch 1 design variants"),("single/16/","Batch 16 design variants"),
                         ("factorial/","Batch, attention schedule and precision"),("four/","Four accelerators on one host")):
        text.extend(["",f"## {title}",""])
        if prefix=="four/":
            text.extend(["Case names give DP / TP / PP / microbatches / host model. "
                         "Each local kernel ranking is reused across host models that run the same shard. "
                         "[Transport assumptions and per-case bandwidth](four-case-metrics.md).",""])
        text.extend(["| Case | TTFT s | TBT ms | Tokens/s/user | Tokens/s, all | Token 256 at s | PDF prefill | PDF decode | Extended prefill | Extended decode |",
                     "|---|---:|---:|---:|---:|---:|---|---|---|---|"])
        for r in cases:
            if not r["case"].startswith(prefix):
                continue
            w=r["winner"]
            name = r["case"][len(prefix):]
            if prefix.startswith("single/") and r["overrides"]:
                settings = ", ".join(f"{k}={json.dumps(v)}" for k,v in r["overrides"].items())
                name += f" (`{settings}`)"
            columns=[name,f"{r['ttft_s']:.2f}",f"{1000*r['tbt_s']:.2f}",
                     f"{r['tokens_s_user']:.4f}",f"{r['tokens_s_aggregate']:.3f}",f"{r['last_token_s']:.2f}"]
            columns.extend(winner_text(w[phase][scope]) for scope in ("pdf","all") for phase in ("prefill","decode"))
            text.append("| "+" | ".join(columns)+" |")
    text.extend(["","## Reproduce","",
        "Build the focused ONNXim target and run the commands in the [coverage report](coverage.md#reproduce). "
        "The original case manifest defines 64 single-device rows, 36 crossed workload rows and 60 four-device rows. "
        "Several comparisons share the same baseline. Hardware unit tests for a descriptor, tile, buffer hazard or integer dot have their own cycle, dependency or numerical oracle. TTFT requires an entire model and request trace.",""])
    return "\n".join(text)

def reproduce():
    study = load("architecture-study.json")
    design = json.loads((ROOT/"configs/arago_experiments.json").read_text())
    return join(study, design, load("case-elasticity.json"), load("four-case-metrics.json")["cases"])


if __name__ == "__main__":
    cases = reproduce()
    DATA.write_text(json.dumps(cases, indent=2, allow_nan=False)+"\n")
    REPORT.write_text(render(cases))
