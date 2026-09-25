import json
import unittest

from experiments.phase4b.audit_hf_live import REVISION, build_report, summarize_rows


class Phase4BLiveAuditTests(unittest.TestCase):
    def test_schema_summary_contains_metadata_not_text(self):
        rows = [
            {
                "row": {
                    "instance_id": "must-not-be-retained",
                    "turns": [
                        {"isl": 1200, "isl_new": 1200, "osl": 50, "isl_text": "private prompt"},
                        {"isl": 1300, "isl_new": 100, "osl": 60, "osl_text": "private completion"},
                    ],
                }
            }
        ]
        result = summarize_rows("test", rows)
        encoded = json.dumps(result)
        self.assertNotIn("private prompt", encoded)
        self.assertNotIn("private completion", encoded)
        self.assertNotIn("must-not-be-retained", encoded)
        self.assertEqual(result["sample_turns_received"], 2)
        self.assertEqual(result["token_count_stats"]["isl_new"]["count"], 2)

    def test_report_requires_exact_pinned_hf_revision(self):
        metadata = {"sha": REVISION, "cardData": {"dataset_info": []}}
        report = build_report(metadata, {"gpt-oss-120b-pro": [], "gpt-oss-120b-verified": []})
        self.assertTrue(report["revision_verified_against_hub_api"])
        with self.assertRaises(ValueError):
            build_report({"sha": "moving-main"}, {"gpt-oss-120b-pro": [], "gpt-oss-120b-verified": []})


if __name__ == "__main__":
    unittest.main()
