from __future__ import annotations

import unittest

from experiments.phase3v.ledger import BlockReservationLedger, required_blocks
from experiments.phase3w.policy import (
    ORACLE,
    RAW,
    SAFE,
    PREDICTION_ERROR_MS,
    RESERVATION_BUDGET_BLOCKS,
    build_predictions,
    prediction_sign,
    reservation_covers_actual,
)


ACTUAL = {"A": 300.0, "E": 900.0, "I": 1500.0}
ERRORS = {"A": 200.0, "E": -50.0, "I": 120.0}


class Phase3WPolicyTests(unittest.TestCase):
    def test_frozen_prediction_examples(self) -> None:
        oracle = build_predictions(ACTUAL, ORACLE, ERRORS)
        raw = build_predictions(ACTUAL, RAW, ERRORS)
        safe = build_predictions(ACTUAL, SAFE, ERRORS)
        self.assertEqual([raw[s].predicted_deadline_ms for s in "AEI"], [500.0, 850.0, 1620.0])
        self.assertEqual([oracle[s].scheduler_deadline_ms for s in "AEI"], [300.0, 900.0, 1500.0])
        self.assertEqual([raw[s].scheduler_deadline_ms for s in "AEI"], [500.0, 850.0, 1620.0])
        self.assertEqual([safe[s].scheduler_deadline_ms for s in "AEI"], [300.0, 650.0, 1420.0])

    def test_actual_resume_deadline_is_unchanged(self) -> None:
        rows = build_predictions(ACTUAL, SAFE, ERRORS)
        self.assertEqual(rows["A"].actual_deadline_ms, 300.0)
        self.assertEqual(rows["E"].actual_deadline_ms, 900.0)
        self.assertEqual(rows["I"].actual_deadline_ms, 1500.0)

    def test_reservation_expiry_and_coverage(self) -> None:
        raw = build_predictions(ACTUAL, RAW, ERRORS)
        safe = build_predictions(ACTUAL, SAFE, ERRORS)
        self.assertEqual(raw["A"].reservation_end_ms, 600.0)
        self.assertEqual(safe["E"].reservation_end_ms, 950.0)
        self.assertTrue(reservation_covers_actual(safe["E"].reservation_end_ms, 900.0))
        self.assertFalse(reservation_covers_actual(890.0, 900.0))

    def test_dynamic_reservation_budget_and_uniform_job_blocks(self) -> None:
        self.assertEqual(RESERVATION_BUDGET_BLOCKS, 512)
        self.assertEqual(required_blocks(4096), 256)
        ledger = BlockReservationLedger(RESERVATION_BUDGET_BLOCKS)
        ledger.grant("A", required_blocks(4096), 200.0, 400.0)
        self.assertTrue(ledger.can_admit(required_blocks(4096)))
        ledger.grant("B", required_blocks(4096), 220.0, 420.0)
        self.assertEqual(ledger.reserved_blocks, 512)
        self.assertFalse(ledger.can_admit(required_blocks(4096)))

    def test_error_sign_labels(self) -> None:
        self.assertEqual(prediction_sign(200), "LATE_PREDICTION")
        self.assertEqual(prediction_sign(-50), "EARLY_PREDICTION")
        self.assertEqual(prediction_sign(0), "ZERO_ERROR")
        self.assertEqual(PREDICTION_ERROR_MS["D"], 200.0)
        self.assertEqual(PREDICTION_ERROR_MS["H"], -50.0)
        self.assertEqual(PREDICTION_ERROR_MS["L"], 120.0)

    def test_non_oracle_scheduler_interface_does_not_need_actual_deadline(self) -> None:
        # Promotion scheduling consumes only these returned inputs, not the
        # actual-deadline map used by the resume dispatcher.
        safe = build_predictions(ACTUAL, SAFE, ERRORS)
        scheduler_inputs = {s: row.scheduler_deadline_ms for s, row in safe.items()}
        self.assertEqual(scheduler_inputs, {"A": 300.0, "E": 650.0, "I": 1420.0})


if __name__ == "__main__":
    unittest.main()
