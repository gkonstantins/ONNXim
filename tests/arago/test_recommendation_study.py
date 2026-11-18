"""Regress paired gains and verify the envelope against an independent scan."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/arago"))
import recommendation_study as study


class RecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(study.RESULTS.read_text())
        cls.serial = [row for row in cls.data["four"] if row["network"] == "serial"]

    def test_pairs_improve_whole_decode(self):
        cases = self.data["single"]["16"]
        for label, before, after, _ in study.PAIRS:
            with self.subTest(change=label):
                a, b = cases[before], cases[after]
                self.assertGreater(a["decode_ms"], b["decode_ms"])
                self.assertAlmostEqual(b["tokens_s_user"] / a["tokens_s_user"],
                                       a["decode_ms"] / b["decode_ms"])
        self.assertEqual(len(set((before, after) for _, before, after, _ in study.PAIRS)),
                         len(study.PAIRS))

    def test_crossover_against_scan_and_at_ties(self):
        ranges = study.winner_ranges(self.serial)
        self.assertEqual([(r["dp"], r["tp"], r["pp"]) for _, _, r in ranges],
                         [(4, 1, 1), (2, 2, 1), (1, 4, 1)])
        for (left, upper, current), (lower, right, following) in zip(ranges, ranges[1:]):
            self.assertEqual(upper, lower)
            self.assertAlmostEqual(study.service(current, upper) / study.service(following, lower), 1.)
        for k in range(-60, 61):
            bandwidth = 10 ** (k / 10)
            winner = next(candidate for left, right, candidate in ranges if left <= bandwidth < right)
            def seconds(r):
                payload_s = (r["up_B_step"] + r["down_B_step"]) / (bandwidth * 1e9)
                return 1 / r["device_limit_tokens_s_user"] + payload_s
            self.assertEqual(seconds(winner), min(seconds(candidate) for candidate in self.serial))

    def test_default_host_rates_match_report(self):
        self.assertEqual(len(self.serial), 12)
        for row in self.serial:
            self.assertAlmostEqual(1e9 / study.service(row, 16), row["decode_tokens_s_user_at_link"])
        self.assertEqual(study.render(self.data), study.DOCUMENT.read_text())


if __name__ == "__main__":
    unittest.main()
