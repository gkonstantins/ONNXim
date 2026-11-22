"""Run paired one-factor experiments against the real trace-driven target."""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from statistics import mean
from units import clock_hz, cycles_ns

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = ROOT / "build/arago/cmake/arago/onnxim-arago"
BASE = Path(os.environ.get("ARAGO_HARDWARE_CONFIG", ROOT / "configs/arago_dual_array.json")).resolve()
MODEL = ROOT / "models/language_models/llama3-8b.json"
DESIGN = ROOT / "configs/arago_experiments.json"
LAYERS, HIDDEN = 32, 4096


def simulate(binary, directory, batch, schedule, overrides, prompt=2048, outputs=256):
    base = json.loads(BASE.read_text())
    config = {**base, **overrides, "model_endpoints": True}
    trace = directory / "requests.csv"
    with trace.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("time","prompt_length","target_length","cached_length"))
        writer.writerows([(0,prompt,outputs,0)] * batch)
    cfg = directory / "target.json"
    cfg.write_text(json.dumps(config))
    cmd = [str(binary),"--mode","arago","--config",str(cfg),"--model",str(MODEL),
           "--trace_file",str(trace),"--batch",str(batch),"--attention",schedule]
    output = subprocess.check_output(cmd, text=True)
    rows = [json.loads(line) for line in output.splitlines()]
    expected = [("prefill",prompt,prompt)] + [("decode",1,prompt+i) for i in range(1,outputs)]
    if [(r["phase"],r["queries"],r["visible_context"]) for r in rows] != expected:
        raise ValueError("incorrect request progression")
    for row in rows:
        if row["batch"] != batch or row["layer"]["peak_sram_bytes"] > config["scratchpad_bytes"]:
            raise ValueError("batch or SRAM invariant failed")
    return rows


def device_cycles(row):
    return LAYERS * row["layer"]["cycles"] + row["endpoints"]["cycles"]


def attention_counters(rows):
    totals = {key:sum(row["attention"][key] for row in rows) for key in
              ("cycles","dram_read_bytes","dram_write_bytes","array16_compute_cycles",
               "array32_compute_cycles","dma_busy_cycles")}
    return {**totals, "steps":len(rows),
            "utilization":{key:totals[key]/totals["cycles"] for key in
                           ("array16_compute_cycles","array32_compute_cycles","dma_busy_cycles")}}


def summarize(rows, baseline):
    att = lambda group: mean(r["attention"]["cycles"] for r in group)
    base_time = mean(cycles_ns(device_cycles(r),r) for r in baseline[1:])
    time = mean(cycles_ns(device_cycles(r),r) for r in rows[1:])
    return {"attention_counters":{"prefill":attention_counters(rows[:1]),"decode":attention_counters(rows[1:])},
            "ttft_device_s":cycles_ns(device_cycles(rows[0]),rows[0])/1e9, "decode_ms":time/1e6,
            "tokens_s_user":1e9/time,"decode_speedup":base_time/time,
            "prefill_attention_cycles":att(rows[:1]), "decode_attention_mean_cycles":att(rows[1:]),
            "peak_layer_sram_B":max(r["layer"]["peak_sram_bytes"] for r in rows),
            "prefill_layer_dram_B":sum(rows[0]["layer"][k] for k in ("dram_read_bytes","dram_write_bytes")),
            "prefill_attention_speedup":cycles_ns(att(baseline[:1]),baseline[0])/cycles_ns(att(rows[:1]),rows[0]),
            "decode_attention_speedup":cycles_ns(att(baseline[1:]),baseline[0])/cycles_ns(att(rows[1:]),rows[0]),
            **({"clock_hz":clock_hz(rows[0])} if clock_hz(rows[0])!=10**9 else {})}


def host_payload(batch, queries, dp, tp, pp):
    """Bytes crossing the single host cut. Activations in A16, IDs in 32 bits."""
    if min(batch,queries,dp,tp,pp) < 1 or batch % dp or LAYERS % pp:
        raise ValueError("unsupported partition")
    state = batch * queries * HIDDEN * 2
    # For each TP group: receive all partials then broadcast reduced states.
    reduced = 2 * LAYERS * tp * state if tp > 1 else 0
    # One representative per replica sends the hidden state to the host per PP
    # boundary. All TP ranks at the next stage need a copy.
    upload = reduced + (pp-1)*state + 4*batch
    download = reduced + (pp-1)*tp*state + 4*batch*queries*tp
    phases = (2 if tp > 1 else 0)*2*LAYERS + 2*(pp-1) + 2
    return upload, download, phases


def transfer_work(up, down, duplex):
    # Full duplex is a chunk-pipelined lower bound; serial preserves direction barriers.
    return max(up, down) if duplex else up + down


def network_time(compute_ns, up, down, phases, bandwidth, options, host_ns=0):
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    fixed = phases * options["startup_ns"] + host_ns
    transfer = transfer_work(up,down,options["duplex"]) / bandwidth  # GB/s = B/ns
    if options["overlap"]:
        return fixed + max(compute_ns, transfer)
    return fixed + compute_ns + transfer


def bandwidth_for_fraction(compute_ns, up, down, phases, fraction, options, host_ns=0):
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0,1)")
    # Target is a fraction of the compute-only performance for this placement.
    budget = compute_ns / fraction - phases*options["startup_ns"] - host_ns
    if budget < compute_ns:
        return None
    if not options["overlap"]:
        budget -= compute_ns
    return transfer_work(up,down,options["duplex"])/budget if budget > 0 else None


def host_service(batch, queries, tp, options):
    fixed = options.get("host_dispatch_ns",0)
    bandwidth = options.get("reduce_GB_s")
    if tp == 1 or bandwidth is None:
        return fixed
    if bandwidth <= 0:
        raise ValueError("host reduction rate must be positive")
    # Each reduction reads tp A16 inputs and writes one A16 output.
    return fixed + 2*LAYERS*(tp+1)*batch*queries*HIDDEN*2 / bandwidth


def pipeline_cycles(row, pp, microbatches):
    # Identical microbatches on a linear pipeline: first traversal plus one
    # bottleneck-stage service interval for each subsequent microbatch.
    block = (LAYERS // pp) * row["layer"]["cycles"]
    stages = [block] * pp
    stages[0] += row["embedding_cycles"]
    stages[-1] += row["endpoints"]["cycles"] - row["embedding_cycles"]
    return sum(stages) + (microbatches-1)*max(stages)


def rank_resident_bytes(row, tp, pp, microbatches):
    # The local trace already includes the active microbatch's KV and workspace.
    # Every other admitted microbatch retains its KV while this stage serves it.
    other_users = row["batch"] * (microbatches - 1)
    kv_per_position = (LAYERS // pp) * 2 * (1024 // tp) * 2
    return row["resident_estimate_B"] + other_users * row["visible_context"] * kv_per_position


def ideal_roof(batch=16, devices=4):
    """Perfect work/memory partition and utilization, no fixed costs or stalls."""
    weights = LAYERS*(2*4096**2 + 2*4096*1024 + 3*4096*14336) + 4096*128256
    step_ns = []
    for context in range(2049,2304):
        mac = batch*(weights + 2*LAYERS*4096*context)
        traffic = weights + 2*batch*LAYERS*1024*2*context
        step_ns.append(max(mac/(768*devices),traffic/(64*devices)))
    compute = cycles_ns(mean(step_ns),json.loads(BASE.read_text()))
    up,down,phases=host_payload(batch,1,1,devices,1)
    return {"tokens_s_user":1e9/compute,"tokens_s_batch":batch*1e9/compute,
            "GB_s_for_70pct_ideal_overlap":bandwidth_for_fraction(compute,up,down,phases,.70,
                {"duplex":True,"overlap":True,"startup_ns":0}),
            "GB_s_for_70pct_serial":bandwidth_for_fraction(compute,up,down,phases,.70,
                {"duplex":False,"overlap":False,"startup_ns":0})}


def run(binary, manifest):
    study = json.loads(manifest.read_text())
    result = {"clock_hz":json.loads(BASE.read_text())["clock_hz"],"config_sha256":hashlib.sha256(BASE.read_bytes()).hexdigest(),
              "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),"single":{},"four":[],"factorial":[], "ideal_four":ideal_roof()}
    sources = sorted((ROOT/"src/arago").glob("*.[ch]*")) + [Path(__file__)]
    result["timing_source_sha256"] = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()
    allowed = set(json.loads(BASE.read_text())) | {"share_kv","mac_percent","queued_delivery","duplex_dma",
        "overlap_drain","wavefront_fill","transpose_decode","double_buffer","pipeline_stages","tensor_shards","query_tile",
        "key_tile","precision_policy","model_endpoints","trace_panels"}
    for case in study["cases"]:
        unknown = set(case.get("override",{})) - allowed
        if unknown:
            raise ValueError(f"unknown override: {unknown}")
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        cache = {}
        def timed(batch, mode="flashattention", overrides=None):
            overrides = overrides or {}
            key = (batch, mode, json.dumps(overrides,sort_keys=True))
            if key not in cache:
                cache[key] = simulate(binary,directory,batch,mode,overrides)
            return cache[key]
        for batch in study["batches"]:
            base = timed(batch)
            cases = result["single"][str(batch)] = {}
            for item in study["cases"]:
                rows = timed(batch,item.get("attention","flashattention"),item.get("override"))
                cases[item["name"]] = summarize(rows,base)
                if "parameter_ratio" in item:
                    for phase in ("prefill","decode"):
                        speed = cases[item["name"]][phase+"_attention_speedup"]
                        cases[item["name"]][phase+"_elasticity"] = abs(1/speed-1)/abs(item["parameter_ratio"]-1)
        for batch in study["batches"]:
            for mode in study["schedules"]:
                for policy in study["precision_policies"]:
                    rows = timed(batch,mode,{"precision_policy":policy})
                    result["factorial"].append({"batch":batch,"schedule":mode,"policy":policy,
                        **summarize(rows,timed(batch))})
        for dp,tp,pp in study["placements"]:
            batch = 16
            if dp*tp*pp != 4:
                raise ValueError("expected four devices")
            for micro in (study["pipeline_microbatches"] if pp>1 else [1]):
                rows = timed(batch//(dp*micro),overrides={"tensor_shards":tp,"pipeline_stages":pp})
                # All DP groups run concurrently; each PP stage owns 32/pp layers.
                resident = max(rank_resident_bytes(row,tp,pp,micro) for row in rows)
                if resident > json.loads(BASE.read_text())["dram_bytes"]:
                    raise ValueError("pipeline rank exceeds DRAM capacity with resident microbatches")
                decode = mean(cycles_ns(pipeline_cycles(row,pp,micro),row) for row in rows[1:])
                prefill = cycles_ns(pipeline_cycles(rows[0],pp,micro),rows[0])
                up,down,phases = host_payload(batch,1,dp,tp,pp)
                phases *= micro
                for label,opts in study["host_models"].items():
                    host = host_service(batch,1,tp,opts)
                    b70 = bandwidth_for_fraction(decode,up,down,phases,0.7,opts,host)
                    pre_up,pre_down,pre_phases = host_payload(batch,2048,dp,tp,pp)
                    payload = pre_up,pre_down,pre_phases*micro
                    result["four"].append({"microbatches":micro,"dp":dp,"tp":tp,"pp":pp,"network":label,
                        "rank_resident_estimate_B":resident,
                        "device_limit_tokens_s_user":1e9/decode,"up_B_step":up,"down_B_step":down,
                        "minimum_GB_s_for_70pct":b70,
                        "predicted_fraction_at_min":None if b70 is None else decode/network_time(decode,up,down,phases,b70,opts,host),
                        "ttft_at_link_s":network_time(prefill,*payload,opts["GB_s"],opts,host_service(batch,2048,tp,opts))/1e9,
                        "decode_tokens_s_user_at_link":1e9/network_time(decode,up,down,phases,opts["GB_s"],opts,host)})
        return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--simulator",type=Path,default=DEFAULT_BINARY)
    p.add_argument("--manifest",type=Path,default=DESIGN)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.output.write_text(json.dumps(run(args.simulator.resolve(),args.manifest),indent=2,allow_nan=False)+"\n")
