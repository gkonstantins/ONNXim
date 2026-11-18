"""Check case coverage, independent rates and consistency with the paired experiments."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/arago"))
import report_cases as report


class ReportTests(unittest.TestCase):
    def test_synchronized_token_accounting(self):
        result = report.rates(100, 2, 16)
        self.assertEqual(result["tokens_s_user"], 0.5)
        self.assertEqual(result["tokens_s_aggregate"], 8)
        self.assertEqual(result["last_token_s"], 610)

    def test_all_cases_reproduce(self):
        rows = report.reproduce()
        self.assertEqual(len(set(row["case"] for row in rows)), 160)
        self.assertEqual(rows, json.loads(report.DATA.read_text()))
        self.assertEqual(report.render(rows), report.REPORT.read_text())
        for row in rows:
            self.assertAlmostEqual(row["tokens_s_user"]*row["tbt_s"], 1)
            self.assertAlmostEqual(row["tokens_s_user"]*row["batch"],row["tokens_s_aggregate"])
            for phase in ("prefill","decode"):
                winners = row["winner"][phase]
                self.assertGreaterEqual(winners["all"]["absolute"]+1e-13,winners["pdf"]["absolute"])
            if row["case"].startswith("four/"):
                self.assertLessEqual(row["tokens_s_user"],row["mapped_ceiling_tokens_s_user"]+1e-10)
            else:
                self.assertTrue(0 <= row["attention_share_decode"] <= 1)
                self.assertTrue(0 <= row["attention_share_prefill"] <= 1)


if __name__ == "__main__":
    unittest.main()
