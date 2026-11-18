"""Measure one-sided local attention-time elasticities at every study design point."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import tempfile

from architecture_study import BASE, DEFAULT_BINARY, DESIGN, MODEL, ROOT, simulate


# All time/bandwidth/capacity numbers in the assignment for which the timing
# model accepts scalar overrides. Clock, element widths and geometry define
# the target ISA and workload; changing those is a different experiment.
PDF_PARAMETERS = (
    "dma_setup", "dram_read_start", "dram_write_start", "array_launch", "vector_launch",
    "dma_bytes_per_cycle", "sram_dma_rate", "array16_input_bytes_per_cycle",
    "array32_input_bytes_per_cycle", "array_output_bytes_per_cycle", "vector_bytes_per_cycle",
    "scratchpad_bytes", "dram_bytes",
)
LATENCIES = {"dma_setup", "dram_read_start", "dram_write_start", "array_launch", "vector_launch"}
PARAMETERS = (*PDF_PARAMETERS, "mac_percent")
PHASES = ("prefill", "decode")
OUTPUT = ROOT / "example/arago/results/case-elasticity.json"


def experiment_points(study):
    for batch in study["batches"]:
        for case in study["cases"]:
            yield (f"single/{batch}/{case['name']}", batch,
                   case.get("attention", "flashattention"), case.get("override", {}))
        for schedule in study["schedules"]:
            for policy in study["precision_policies"]:
                yield (f"factorial/{batch}/{schedule}/{policy}", batch, schedule,
                       {"precision_policy": policy})
    for dp, tp, pp in study["placements"]:
        for micro in study["pipeline_microbatches"] if pp > 1 else [1]:
            yield (f"four/{dp}/{tp}/{pp}/{micro}", 16 // (dp * micro), "flashattention",
                   {"tensor_shards": tp, "pipeline_stages": pp})


def varied_value(parameter, value):
    # Integer nearest to a 10% move in the favorable direction, at least one unit.
    # Use the actual fractional change in the subsequent quotient.
    delta = max(1, (value + 5) // 10)
    return max(0, value - delta) if parameter in LATENCIES else value + delta


def time_elasticity(base_cycles, changed_cycles, base_value, changed_value):
    return ((changed_cycles - base_cycles) * base_value /
            (base_cycles * (changed_value - base_value)))


def make_profile(config, measure):
    baseline = measure(config)
    probes = {}
    for parameter in PARAMETERS:
        original = config[parameter]
        value = varied_value(parameter, original)
        if value == original:
            raise ValueError(f"cannot perturb {parameter} from {original} in the favorable direction")
        cycles = measure({**config, parameter: value})
        probes[parameter] = {
            "from": original, "to": value, "cycles": cycles,
            "elasticity": {phase: time_elasticity(baseline[phase], cycles[phase], original, value)
                           for phase in PHASES},
        }
    winners = {}
    for phase in PHASES:
        winners[phase] = {}
        for group, candidates in (("all", PARAMETERS), ("pdf", PDF_PARAMETERS)):
            magnitude = max(abs(probes[p]["elasticity"][phase]) for p in candidates)
            tied = [p for p in candidates if math.isclose(abs(probes[p]["elasticity"][phase]),
                                                          magnitude, rel_tol=1e-12, abs_tol=1e-14)]
            winners[phase][group] = {"absolute": magnitude, "parameters": tied}
    return {"baseline_cycles": baseline, "probes": probes, "winner": winners}


def run(binary, manifest, workers=4, prompt=2048, outputs=256):
    if min(workers, prompt) < 1 or outputs < 2:
        raise ValueError("need positive jobs, prompt and at least one cached token")
    base = {**json.loads(BASE.read_text()), "mac_percent": 100,
            "precision_policy": "scaled", "tensor_shards": 1, "pipeline_stages": 1}
    plans, lookup = {}, {}
    for key, batch, schedule, overrides in experiment_points(json.loads(manifest.read_text())):
        # Drop explicit defaults only; two precision policies with equal service
        # still get separate native executions and separate auditable profiles.
        changes = {k: v for k, v in overrides.items() if base.get(k) != v}
        point = {"batch": batch, "schedule": schedule, "overrides": changes}
        digest = hashlib.sha256(json.dumps(point, sort_keys=True).encode()).hexdigest()[:16]
        if digest in plans and plans[digest] != point:
            raise ValueError("profile key collision")
        plans[digest] = point
        lookup[key] = digest

    def evaluate(item):
        digest, point = item
        with tempfile.TemporaryDirectory() as tmp:
            def measure(cfg):
                rows = simulate(binary, Path(tmp), point["batch"], point["schedule"], cfg, prompt, outputs)
                return {"prefill": rows[0]["attention"]["cycles"],
                        "decode": sum(row["attention"]["cycles"] for row in rows[1:])}
            profile = make_profile({**base, **point["overrides"]}, measure)
        return digest, {**point, **profile}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        profiles = dict(pool.map(evaluate, plans.items()))

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    timing_sources = [*sorted((ROOT/"src/arago").glob("*.[ch]*")),
                      ROOT/"src/main.cc", ROOT/"tools/arago/architecture_study.py", Path(__file__)]
    return {
        "scope": "One decoder layer's native attention kernel on one rank. Decode cycles sum all cached steps.",
        "method": "Signed (delta cycles / base cycles) / (delta parameter / base parameter), one variable at "
                  "a time, ~10% favorable integer step. Winners maximize absolute elasticity. Communication "
                  "and fixed-clock or ISA geometry changes are outside the kernel ranking.",
        "prompt": prompt, "outputs": outputs, "pdf_parameters": PDF_PARAMETERS,
        "assumed_parameters": ["mac_percent"], "manifest_sha256": sha(manifest),
        "target_sha256": sha(BASE), "model_sha256": sha(MODEL),
        "timing_sources_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in timing_sources)).hexdigest(),
        "cases": lookup, "profiles": profiles,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--manifest", type=Path, default=DESIGN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    results = run(args.simulator.resolve(), args.manifest, args.jobs)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(f"{len(results['cases'])} case keys, {len(results['profiles'])} local kernel profiles")
