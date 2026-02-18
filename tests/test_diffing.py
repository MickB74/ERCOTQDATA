from __future__ import annotations

import unittest

import pandas as pd

from ercot_queue.diffing import calculate_diff


class DiffingTests(unittest.TestCase):
    def test_identical_dataframes_report_no_changes(self) -> None:
        previous = pd.DataFrame(
            {
                "record_key": ["k1", "k2"],
                "project_name": ["A", "B"],
                "capacity_mw": [50.0, 75.0],
            }
        )
        current = previous.copy()

        report = calculate_diff(previous, current)
        summary = report["summary"]

        self.assertEqual(summary["added"], 0)
        self.assertEqual(summary["removed"], 0)
        self.assertEqual(summary["changed"], 0)
        self.assertEqual(summary["unchanged"], 2)

    def test_row_update_is_reported_as_changed(self) -> None:
        previous = pd.DataFrame(
            {
                "record_key": ["k1"],
                "status": ["UNDER_REVIEW"],
                "capacity_mw": [100.0],
            }
        )
        current = pd.DataFrame(
            {
                "record_key": ["k1"],
                "status": ["APPROVED"],
                "capacity_mw": [100.0],
            }
        )

        report = calculate_diff(previous, current)
        summary = report["summary"]

        self.assertEqual(summary["added"], 0)
        self.assertEqual(summary["removed"], 0)
        self.assertEqual(summary["changed"], 1)
        self.assertEqual(summary["unchanged"], 0)
        self.assertEqual(len(report["changed_field_details"]), 1)


if __name__ == "__main__":
    unittest.main()
