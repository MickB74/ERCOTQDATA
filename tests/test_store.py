from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ercot_queue import store


class StorePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name)

        self.original_data_dir = store.DATA_DIR
        self.original_snapshot_dir = store.SNAPSHOT_DIR
        self.original_change_dir = store.CHANGE_DIR
        self.original_metadata_path = store.METADATA_PATH
        self.original_current_snapshot_path = store.CURRENT_SNAPSHOT_PATH

        store.DATA_DIR = self.base / "data"
        store.SNAPSHOT_DIR = store.DATA_DIR / "snapshots"
        store.CHANGE_DIR = store.DATA_DIR / "changes"
        store.METADATA_PATH = store.DATA_DIR / "metadata.json"
        store.CURRENT_SNAPSHOT_PATH = store.DATA_DIR / "current_snapshot.parquet"

    def tearDown(self) -> None:
        store.DATA_DIR = self.original_data_dir
        store.SNAPSHOT_DIR = self.original_snapshot_dir
        store.CHANGE_DIR = self.original_change_dir
        store.METADATA_PATH = self.original_metadata_path
        store.CURRENT_SNAPSHOT_PATH = self.original_current_snapshot_path
        self.temp_dir.cleanup()

    def test_save_snapshot_writes_parquet_and_current_snapshot(self) -> None:
        df = pd.DataFrame(
            {
                "record_key": ["a", "b"],
                "capacity_mw": [100.0, 200.0],
                "cod_date": ["2026-01-01", "2026-02-01"],
            }
        )

        meta = store.save_snapshot(
            df,
            source_metadata={"source": "test", "source_url": "https://example.com/report.xlsx"},
            diff_report={"summary": {"added": 2, "removed": 0, "changed": 0}},
        )

        snapshot_path = Path(meta["snapshot_path"])
        self.assertEqual(snapshot_path.suffix, ".parquet")
        self.assertTrue(snapshot_path.exists())
        self.assertTrue(store.CURRENT_SNAPSHOT_PATH.exists())

        loaded_df, loaded_meta = store.load_latest_snapshot()
        self.assertIsNotNone(loaded_df)
        self.assertIsNotNone(loaded_meta)
        assert loaded_df is not None
        self.assertEqual(len(loaded_df), 2)
        self.assertEqual(set(loaded_df["record_key"]), {"a", "b"})
        self.assertEqual(loaded_meta.get("snapshot_format"), "parquet")

    def test_load_latest_snapshot_supports_legacy_csv(self) -> None:
        store.ensure_data_dirs()
        legacy_path = store.SNAPSHOT_DIR / "legacy.csv"
        pd.DataFrame(
            {
                "record_key": ["legacy-1"],
                "capacity_mw": ["150.5"],
                "service_date": ["2026-03-01"],
            }
        ).to_csv(legacy_path, index=False)

        metadata = {
            "snapshots": [
                {
                    "snapshot_id": "legacy",
                    "snapshot_path": str(legacy_path),
                    "change_path": str(store.CHANGE_DIR / "legacy.json"),
                    "source": "legacy",
                    "source_url": "https://example.com/legacy.csv",
                }
            ]
        }
        with store.METADATA_PATH.open("w", encoding="utf-8") as file_handle:
            json.dump(metadata, file_handle)

        loaded_df, loaded_meta = store.load_latest_snapshot()
        self.assertIsNotNone(loaded_df)
        self.assertIsNotNone(loaded_meta)
        assert loaded_df is not None
        assert loaded_meta is not None
        self.assertEqual(loaded_meta["snapshot_path"], str(legacy_path))
        self.assertEqual(float(loaded_df.loc[0, "capacity_mw"]), 150.5)
        self.assertTrue(pd.notna(loaded_df.loc[0, "service_date"]))

    def test_load_latest_snapshot_uses_current_snapshot_without_metadata(self) -> None:
        store.ensure_data_dirs()
        pd.DataFrame({"record_key": ["current-1"], "capacity_mw": [88.0]}).to_parquet(
            store.CURRENT_SNAPSHOT_PATH,
            index=False,
        )

        loaded_df, loaded_meta = store.load_latest_snapshot()
        self.assertIsNotNone(loaded_df)
        self.assertIsNotNone(loaded_meta)
        assert loaded_df is not None
        assert loaded_meta is not None
        self.assertEqual(loaded_meta["snapshot_id"], "current")
        self.assertEqual(len(loaded_df), 1)


if __name__ == "__main__":
    unittest.main()
