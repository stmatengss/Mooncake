#!/usr/bin/env python3
"""Unit tests for scripts/ci/run_store_perf_ci.py helpers."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "ci" / "run_store_perf_ci.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_store_perf_ci", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class StorePerfCiHelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def test_default_cases_include_plain_and_zcopy(self):
        cases = self.mod.default_cases(skip_zcopy=False)
        names = [case["name"] for case in cases]
        self.assertIn("verify_write_plain", names)
        self.assertIn("write_perf_plain", names)
        self.assertIn("read_perf_plain", names)
        self.assertIn("write_perf_zcopy", names)
        self.assertIn("read_perf_zcopy", names)

    def test_default_cases_can_skip_zcopy(self):
        names = [case["name"] for case in self.mod.default_cases(skip_zcopy=True)]
        self.assertNotIn("write_perf_zcopy", names)
        self.assertNotIn("read_perf_zcopy", names)

    def test_threshold_pass_and_fail(self):
        thresholds = {
            "write_perf_plain": {
                "phase": "write_perf",
                "max_failed_requests": 0,
                "min_MiB_per_sec": 5.0,
                "min_kv_per_sec": 100.0,
            }
        }
        good = {
            "phases": [
                {
                    "name": "write_perf",
                    "failed_requests": 0,
                    "MiB_per_sec": 40.0,
                    "kv_per_sec": 1000.0,
                }
            ]
        }
        bad = {
            "phases": [
                {
                    "name": "write_perf",
                    "failed_requests": 0,
                    "MiB_per_sec": 1.0,
                    "kv_per_sec": 10.0,
                }
            ]
        }
        self.assertEqual(
            self.mod.check_thresholds("write_perf_plain", good, thresholds),
            [],
        )
        failures = self.mod.check_thresholds("write_perf_plain", bad, thresholds)
        self.assertTrue(any("MiB_per_sec" in item for item in failures))
        self.assertTrue(any("kv_per_sec" in item for item in failures))

    def test_threshold_file_loads(self):
        path = REPO_ROOT / "mooncake-store" / "benchmarks" / "ci_perf_thresholds.json"
        cases = self.mod.load_thresholds(path)
        self.assertIn("write_perf_plain", cases)
        self.assertGreaterEqual(cases["write_perf_plain"]["min_MiB_per_sec"], 1.0)

    def test_summary_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.json"
            payload = {"results": [{"name": "demo"}], "gate_failures": []}
            summary.write_text(json.dumps(payload), encoding="utf-8")
            loaded = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(loaded["results"][0]["name"], "demo")


if __name__ == "__main__":
    unittest.main()
