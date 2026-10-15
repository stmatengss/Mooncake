#!/usr/bin/env python3
"""Unit tests for scripts/ci/run_store_perf_ci.py helpers."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
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
        cases = self.mod.default_cases(skip_zcopy=False, protocol="tcp")
        names = [case["name"] for case in cases]
        self.assertIn("verify_write_plain", names)
        self.assertIn("write_perf_plain", names)
        self.assertIn("read_perf_plain", names)
        self.assertIn("write_perf_zcopy", names)
        self.assertIn("read_perf_zcopy", names)

    def test_default_cases_can_skip_zcopy(self):
        names = [
            case["name"]
            for case in self.mod.default_cases(skip_zcopy=True, protocol="tcp")
        ]
        self.assertNotIn("write_perf_zcopy", names)
        self.assertNotIn("read_perf_zcopy", names)

    def test_rdma_cases_use_smaller_workloads(self):
        tcp = {
            case["name"]: case
            for case in self.mod.default_cases(skip_zcopy=True, protocol="tcp")
        }
        rdma = {
            case["name"]: case
            for case in self.mod.default_cases(skip_zcopy=True, protocol="rdma")
        }
        self.assertLess(
            rdma["write_perf_plain"]["nr_objects"],
            tcp["write_perf_plain"]["nr_objects"],
        )

    def test_threshold_pass_and_fail_dict_schema(self):
        thresholds = {
            "write_perf_plain": {
                "phase": "write_perf",
                "max_failed_requests": 0,
                "min_MiB_per_sec": 5.0,
                "min_kv_per_sec": 100.0,
            }
        }
        good = {
            "phases": {
                "write_perf": {
                    "failed_requests": 0,
                    "MiB_per_sec": 40.0,
                    "kv_per_sec": 1000.0,
                }
            }
        }
        bad = {
            "phases": {
                "write_perf": {
                    "failed_requests": 0,
                    "MiB_per_sec": 1.0,
                    "kv_per_sec": 10.0,
                }
            }
        }
        self.assertEqual(
            self.mod.check_thresholds("write_perf_plain", good, thresholds),
            [],
        )
        failures = self.mod.check_thresholds("write_perf_plain", bad, thresholds)
        self.assertTrue(any("MiB_per_sec" in item for item in failures))
        self.assertTrue(any("kv_per_sec" in item for item in failures))

    def test_find_phase_supports_legacy_list_schema(self):
        result = {
            "phases": [{"name": "write_perf", "MiB_per_sec": 9.0, "kv_per_sec": 200.0}]
        }
        phase = self.mod.find_phase(result, "write_perf")
        self.assertIsNotNone(phase)
        self.assertEqual(phase["MiB_per_sec"], 9.0)

    def test_threshold_files_load(self):
        for rel in (
            "mooncake-store/benchmarks/ci_perf_thresholds.json",
            "mooncake-store/benchmarks/ci_perf_thresholds_rdma.json",
        ):
            path = REPO_ROOT / rel
            cases = self.mod.load_thresholds(path)
            self.assertIn("write_perf_plain", cases)
            self.assertGreaterEqual(cases["write_perf_plain"]["min_MiB_per_sec"], 0.1)

    def test_resolve_rdma_device_prefers_env(self):
        with mock.patch.dict(os.environ, {"RDMA_DEVICE": "rxe0"}, clear=False):
            self.assertEqual(self.mod.resolve_rdma_device(""), "rxe0")
        self.assertEqual(self.mod.resolve_rdma_device("mlx5_0"), "mlx5_0")

    def test_resolve_local_hostname_uses_rdma_bind_ip(self):
        with mock.patch.dict(os.environ, {"RDMA_BIND_IP": "10.0.0.5"}, clear=False):
            hostname = self.mod.resolve_local_hostname("127.0.0.1:50071", "rdma")
        self.assertEqual(hostname, "10.0.0.5:50071")

    def test_console_results_lines_include_metrics_and_status(self):
        lines = self.mod.build_console_results_lines(
            [
                {
                    "name": "write_perf_plain",
                    "phase": "write_perf",
                    "MiB_per_sec": 42.5,
                    "kv_per_sec": 1000.0,
                    "lat_p50_ms": 1.234,
                    "lat_p99_ms": 3.456,
                    "gate_failures": [],
                },
                {
                    "name": "read_perf_plain",
                    "phase": "read_perf",
                    "MiB_per_sec": 12.0,
                    "kv_per_sec": 500.0,
                    "lat_p50_ms": 2.0,
                    "lat_p99_ms": 4.0,
                    "gate_failures": ["below threshold"],
                },
            ]
        )
        rendered = "\n".join(lines)
        self.assertIn("Case", rendered)
        self.assertIn("write_perf_plain", rendered)
        self.assertIn("42.50", rendered)
        self.assertIn("3.456", rendered)
        self.assertIn("PASS", rendered)
        self.assertIn("FAIL", rendered)

    def test_summary_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.json"
            payload = {"results": [{"name": "demo"}], "gate_failures": []}
            summary.write_text(json.dumps(payload), encoding="utf-8")
            loaded = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(loaded["results"][0]["name"], "demo")


if __name__ == "__main__":
    unittest.main()
