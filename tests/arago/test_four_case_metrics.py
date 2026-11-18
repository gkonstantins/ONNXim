"""Independent checks for four-device host thresholds and all 60 published cases."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/arago"))
import four_case_metrics as four


class FourCaseTests(unittest.TestCase):
    def test_overlap_checks_each_step(self):
        # 100 B over 5 GB/s costs 20 ns. The first step exposes 10 ns, the second 0.
        traffic = {"link_B":100, "startup_ns":0, "host_ns":0, "overlap":True}
        steps = [10,30]
        self.assertEqual(four.exposed_link_ns(steps,traffic,5),5)
        self.assertEqual(four.latency_ns(steps,traffic,5),25)
        self.assertAlmostEqual(four.minimum_bandwidth(steps,traffic,1e9/25),5)
        self.assertAlmostEqual(four.minimum_bandwidth(steps,traffic,1e9/20),10)
        self.assertIsNone(four.minimum_bandwidth(steps,traffic,1e9/19))

    def test_serial_fixed_overhead_infeasibility(self):
        traffic = {"link_B":200, "startup_ns":5, "host_ns":15, "overlap":False}
        steps = [10,30]
        self.assertEqual(four.latency_ns(steps,traffic,10),60)
        self.assertAlmostEqual(four.minimum_bandwidth(steps,traffic,1e9/60),10)
        self.assertIsNone(four.minimum_bandwidth(steps,traffic,1e9/40))

    def test_all_cases_against_original_rank_schedules(self):
        study = json.loads((ROOT/"example/arago/results/architecture-study.json").read_text())
        report = json.loads(four.RESULT.read_text())
        original = {(r["dp"],r["tp"],r["pp"],r["microbatches"],r["network"]):r for r in study["four"]}
        self.assertEqual(len(original),60)
        self.assertEqual(len(report["cases"]),60)
        for row in report["cases"]:
            key = tuple(row[k] for k in ("dp","tp","pp","microbatches","network"))
            other = original.pop(key)
            with self.subTest(key=key):
                self.assertAlmostEqual(row["ttft_s"],other["ttft_at_link_s"])
                self.assertAlmostEqual(row["tokens_s_user"],other["decode_tokens_s_user_at_link"])
                self.assertAlmostEqual(row["GB_s_70pct_own_mapped"],other["minimum_GB_s_for_70pct"])
                self.assertAlmostEqual(row["mean_tbt_ms"],1000/row["tokens_s_user"])
                self.assertAlmostEqual(row["tokens_s_aggregate"],16*row["tokens_s_user"])
                self.assertAlmostEqual(row["completion_256_s"],row["ttft_s"]+255/row["tokens_s_user"])
                self.assertAlmostEqual(row["mean_tbt_ms"],sum(row[k] for k in
                    ("mean_local_decode_ms","mean_exposed_transfer_ms","host_fixed_ms","link_startup_ms")))
                self.assertLessEqual(row["link_time_elasticity"],0)
                self.assertIsNone(row["GB_s_70pct_ideal"])
                self.assertEqual(row["tp"]==4,row["GB_s_70pct_best_mapped"] is not None)
                if row["network"]=="overlap_bound":
                    self.assertEqual(row["link_time_elasticity"],0)
        self.assertFalse(original)


if __name__=="__main__":
    unittest.main()
