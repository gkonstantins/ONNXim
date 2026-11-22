"""Run both attention schedules through ONNXim and compare one decoder block."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/arago_dual_array.json"
MODEL = "models/language_models/llama3-8b.json"
DEFAULT_SIMULATOR = ROOT / "build/arago/cmake/arago/onnxim-arago"
ADDITIVE = ("cycles", "dram_read_bytes", "dram_write_bytes", "array16_tiles", "array32_tiles")
METRICS = (*ADDITIVE, "peak_sram_bytes")
ACTIVITY = ("array16_compute_cycles", "array32_compute_cycles", "dma_busy_cycles")
# The decode trace has one priming iteration followed by a cached step.
# Fields are phase, batch, query positions per sequence, visible context.
CASES = {
    "prefill": [("prefill", 16, 2048, 2048)],
    "decode": [("prefill", 16, 1, 2175), ("decode", 16, 1, 2176)],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run_trace(simulator, case, schedule, sram_capacity, expected):
    trace = f"example/arago/{case}.csv"
    config = os.environ.get("ARAGO_HARDWARE_CONFIG", CONFIG)
    command = [str(simulator), "--mode", "arago", "--model", MODEL, "--config", config,
               "--batch", "16", "--trace_file", trace, "--attention", schedule]
    process = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, check=True)
    rows = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    tag = f"{case}/{schedule}"
    require(len(rows) == len(expected), f"{tag}: expected {len(expected)} iterations, got {len(rows)}")
    previous_end = 0
    for row, shape in zip(rows, expected):
        require(row["scope"] == "one_decoder_layer" and row["target"] == "arago-dual-array"
                and row["schedule"] == schedule, f"{tag}: wrong target, scope or schedule")
        actual_shape = tuple(row[k] for k in ("phase", "batch", "queries", "visible_context"))
        require(actual_shape == shape, f"{tag}: expected shape {shape}, got {actual_shape}")
        require(row["start_cycle"] >= previous_end, f"{tag}: overlapping sequential batch iterations")
        for phase in ("attention", "layer"):
            counters = row[phase]
            require(all(type(counters[k]) is int and counters[k] >= 0 for k in METRICS),
                    f"{tag}: invalid {phase} counters")
            require(counters["cycles"] > 0 and 0 < counters["peak_sram_bytes"] <= sram_capacity,
                    f"{tag}: invalid {phase} timing or scratchpad usage")
            for k in ACTIVITY:
                if k in counters:
                    require(type(counters[k]) is int and 0 <= counters[k] <= counters["cycles"],
                            f"{tag}: invalid {phase} {k}")
        require(all(row["attention"][k] <= row["layer"][k] for k in METRICS),
                f"{tag}: attention exceeds whole-layer counters")
        previous_end = row["start_cycle"] + row["layer"]["cycles"]
    return rows


def compare(naive, streaming):
    activity = []
    for field in ACTIVITY:
        availability = [field in row[phase] for row in (naive, streaming) for phase in ("layer", "attention")]
        require(all(availability) or not any(availability), f"inconsistent reporting of {field}")
        if all(availability):
            activity.append(field)
    for field in (*ADDITIVE, *activity):
        other_naive = naive["layer"][field] - naive["attention"][field]
        other_stream = streaming["layer"][field] - streaming["attention"][field]
        require(other_naive == other_stream, f"nonattention {field} changed between schedules")
    for field in ("array16_tiles", "array32_tiles"):
        require(naive["attention"][field] == streaming["attention"][field],
                f"attention {field} differs; comparison must use the same native products")
    for field in ("dram_read_bytes", "dram_write_bytes"):
        require(naive["attention"][field] > streaming["attention"][field],
                f"streaming attention did not remove expected intermediate {field}")


def run_pair(simulator, case, optimized, sram_capacity, expected):
    naive = run_trace(simulator, case, "naive", sram_capacity, expected)
    optimized_rows = run_trace(simulator, case, optimized, sram_capacity, expected)
    for before, after in zip(naive, optimized_rows):
        compare(before, after)
    return naive, optimized_rows


def interval_summary(naive, flash):
    summary = {"iterations": len(naive), "first_visible_context": naive[0]["visible_context"],
               "last_visible_context": naive[-1]["visible_context"]}
    for component in ("attention", "layer"):
        metrics = {}
        for rows in (naive, flash):
            cycles = sum(row[component]["cycles"] for row in rows)
            reads = sum(row[component]["dram_read_bytes"] for row in rows)
            writes = sum(row[component]["dram_write_bytes"] for row in rows)
            metrics[rows[0]["schedule"]] = {
                "cycles": cycles, "mean_cycles_per_iteration": cycles / len(rows),
                "dram_read_bytes": reads, "dram_write_bytes": writes, "dram_bytes": reads + writes,
                "peak_sram_bytes": max(row[component]["peak_sram_bytes"] for row in rows),
            }
        before, after = metrics["naive"], metrics["flashattention"]
        metrics["speedup"] = before["cycles"] / after["cycles"]
        metrics["cycles_saved"] = before["cycles"] - after["cycles"]
        metrics["dram_bytes_saved"] = before["dram_bytes"] - after["dram_bytes"]
        metrics["time_reduction_pct"] = 100 * metrics["cycles_saved"] / before["cycles"]
        metrics["dram_reduction_pct"] = 100 * metrics["dram_bytes_saved"] / before["dram_bytes"]
        summary[component] = metrics
    return summary


def report(simulator, workload="probes"):
    cfg = json.loads((ROOT / os.environ.get("ARAGO_HARDWARE_CONFIG", CONFIG)).read_text(encoding="utf-8"))
    result = {"scope": "one_decoder_layer", "target": cfg["target"], "clock_hz": cfg["clock_hz"],
              "model_preset": MODEL, "hardware_config": CONFIG}
    if cfg.get("key_tile",256) != 128:
        result["key_tile"] = cfg.get("key_tile",256)
    if workload == "batch16":
        expected = [("prefill", 16, 2048, 2048)] + [("decode", 16, 1, n) for n in range(2049, 2304)]
        naive, flash = run_pair(simulator, workload, "flashattention", cfg["scratchpad_bytes"], expected)
        result.update(trace="example/arago/batch16.csv", batch=16, prompt_tokens_per_user=2048,
                      output_positions_per_user=256, intervals={
                          "prefill": interval_summary(naive[:1], flash[:1]),
                          "cached_decode": interval_summary(naive[1:], flash[1:]),
                          "total": interval_summary(naive, flash),
                      })
        return result

    result["cases"] = {}
    for case, expected in CASES.items():
        naive, streamed = run_pair(simulator, case, "streaming", cfg["scratchpad_bytes"], expected)
        n, s = naive[-1], streamed[-1]
        summary = {k: n[k] for k in ("phase", "batch", "queries", "visible_context")}
        summary["trace"] = f"example/arago/{case}.csv"
        summary["priming_iterations_excluded"] = len(naive) - 1
        for row in (n, s):
            summary[row["schedule"]] = {
                p: {k: row[p][k] for k in (*METRICS, *ACTIVITY) if k in row[p]}
                for p in ("attention", "layer")
            }
        summary["speedup_naive_over_streaming"] = {
            p: n[p]["cycles"] / s[p]["cycles"] for p in ("attention", "layer")
        }
        result["cases"][case] = summary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("probes", "batch16"), default="probes",
                        help="compare isolated probes or the full batch of 16 sequences")
    parser.add_argument("--simulator", type=Path, default=DEFAULT_SIMULATOR,
                        help="path to the built ONNXim --mode arago executable")
    parser.add_argument("--output", type=Path, help="write comparison JSON here (default: stdout)")
    args = parser.parse_args()
    try:
        serialized = json.dumps(report(args.simulator.resolve(), args.workload), indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(serialized, encoding="utf-8")
        else:
            sys.stdout.write(serialized)
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"comparison failed: {error}\n")


if __name__ == "__main__":
    main()
