"""Independent arithmetic, native pairing, coverage and cross-report checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"tools/arago"))
import case_elasticity as elastic
from architecture_study import DEFAULT_BINARY, DESIGN

BINARY = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else DEFAULT_BINARY


class ElasticityTests(unittest.TestCase):
    def test_actual_rounded_denominator_and_sign(self):
        self.assertEqual(elastic.varied_value("array_output_bytes_per_cycle", 8), 9)
        self.assertEqual(elastic.varied_value("dram_write_start", 150), 135)
        self.assertEqual(elastic.time_elasticity(1000, 950, 8, 9), -.4)
        self.assertEqual(elastic.time_elasticity(1000, 970, 100, 90), .3)

    def test_ties_and_local_operating_point(self):
        base = {p: 100 for p in elastic.PARAMETERS}
        base["dma_setup"] = 200
        def toy(c):
            latency = c["array_launch"] + c["vector_launch"]
            return {"prefill": latency, "decode": 2*c["dma_setup"] + latency}
        profile = elastic.make_profile(base, toy)
        self.assertEqual(set(profile["winner"]["prefill"]["pdf"]["parameters"]),
                         {"array_launch", "vector_launch"})
        self.assertEqual(profile["winner"]["decode"]["all"]["parameters"], ["dma_setup"])
        self.assertEqual(profile["probes"]["dma_setup"]["to"], 180)

    def test_short_native_sweep_matches_manual_pair(self):
        manifest = {"cases": [{"name":"queued", "override":{"queued_delivery":True}}],
                    "batches": [1], "schedules": [], "placements": []}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory/"study.json"
            path.write_text(json.dumps(manifest))
            result = elastic.run(BINARY, path, workers=1, prompt=32, outputs=3)
            profile = result["profiles"][result["cases"]["single/1/queued"]]
            for latency, observed in ((100, profile["baseline_cycles"]),
                                      (90, profile["probes"]["array_launch"]["cycles"])):
                rows = elastic.simulate(BINARY, directory, 1, "flashattention",
                                        {"queued_delivery":True, "array_launch":latency}, 32, 3)
                self.assertEqual(observed["prefill"], rows[0]["attention"]["cycles"])
                self.assertEqual(observed["decode"], sum(r["attention"]["cycles"] for r in rows[1:]))
            for capacity in ("dram_bytes","scratchpad_bytes"):
                self.assertEqual(profile["probes"][capacity]["cycles"], profile["baseline_cycles"])

    def test_saved_design_and_per_case_baselines(self):
        # Cross-check baseline counts against the independent main study: this
        # detects comparing a perturbation against a different operating point.
        study = json.loads(DESIGN.read_text())
        record = json.loads(elastic.OUTPUT.read_text())
        timings = json.loads((elastic.OUTPUT.parent/"architecture-study.json").read_text())
        self.assertEqual(set(record["cases"]), {key for key, *_ in elastic.experiment_points(study)})
        def check(key, row):
            profile = record["profiles"][record["cases"][key]]
            self.assertEqual(set(profile["probes"]),set(elastic.PARAMETERS))
            self.assertEqual(profile["baseline_cycles"]["prefill"],row["prefill_attention_cycles"])
            self.assertAlmostEqual(profile["baseline_cycles"]["decode"] / (record["outputs"]-1),
                                   row["decode_attention_mean_cycles"])
        for batch, cases in timings["single"].items():
            for name, row in cases.items():
                check(f"single/{batch}/{name}", row)
        for row in timings["factorial"]:
            check(f"factorial/{row['batch']}/{row['schedule']}/{row['policy']}", row)
        for row in timings["four"]:
            key="four/{dp}/{tp}/{pp}/{microbatches}".format(**row)
            profile=record["profiles"][record["cases"][key]]
            self.assertEqual(profile["batch"],16//(row["dp"]*row["microbatches"]))
            self.assertEqual(profile["overrides"].get("tensor_shards",1),row["tp"])
            self.assertEqual(profile["overrides"].get("pipeline_stages",1),row["pp"])


if __name__ == "__main__":
    unittest.main()
