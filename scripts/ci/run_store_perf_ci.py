#!/usr/bin/env python3
"""Independent Mooncake Store performance CI runner.

Reuses ``mooncake-store/benchmarks/store_kv_bench.py`` with CI-sized workloads
(similar to SGLang's separate benchmark CI jobs) and enforces conservative
throughput floors so severe performance regressions fail the job.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_SCRIPT = REPO_ROOT / "mooncake-store" / "benchmarks" / "store_kv_bench.py"
DEFAULT_THRESHOLDS = (
    REPO_ROOT / "mooncake-store" / "benchmarks" / "ci_perf_thresholds.json"
)


def is_in_ci() -> bool:
    """Detect CI environments (inspired by SGLang ``is_in_ci``)."""
    markers = (
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "CIRCLECI",
        "TRAVIS",
        "BUILDKITE",
        "MOONCAKE_IS_IN_CI",
    )
    return any(os.environ.get(name, "").lower() in {"1", "true", "yes"} for name in markers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Mooncake Store performance CI checks via store_kv_bench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--local-hostname", default="127.0.0.1:50071")
    parser.add_argument("--metadata-server", default="http://127.0.0.1:8080/metadata")
    parser.add_argument("--master-server", default="127.0.0.1:50051")
    parser.add_argument("--protocol", default="tcp")
    parser.add_argument("--device-name", default="")
    parser.add_argument(
        "--global-segment-size",
        type=int,
        default=256 * 1024 * 1024,
        help="Segment size for CI runners (keep modest for shared VMs).",
    )
    parser.add_argument(
        "--local-buffer-size",
        type=int,
        default=64 * 1024 * 1024,
    )
    parser.add_argument("--output-dir", default="store-perf-ci-results")
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument(
        "--runtime",
        type=int,
        default=0,
        help="Override timed-run seconds. 0 selects CI object-count mode.",
    )
    parser.add_argument(
        "--skip-zcopy",
        action="store_true",
        help="Skip zero-copy API cases (plain put/get only).",
    )
    parser.add_argument(
        "--cases",
        default="",
        help="Comma-separated case names to run. Empty means the default CI suite.",
    )
    return parser


def default_cases(*, skip_zcopy: bool) -> List[Dict[str, Any]]:
    """CI-shrunk store_kv_bench cases (SGLang-style reduced CI ranges)."""
    common = {
        "nr_objects": 64,
        "batch_size": 4,
        "value_size": 4096,
        "key_size": 20,
        "memory_replica_num": 1,
        "nof_replica_num": 0,
        "numjobs": 1,
        "iodepth": 1,
    }
    cases: List[Dict[str, Any]] = [
        {
            "name": "verify_write_plain",
            "scenario": "verify_write",
            "io_api": "plain",
            "verify": True,
            "pattern": "0xab",
            "key_prefix": "civfy",
            "nr_objects": 16,
            **{k: v for k, v in common.items() if k != "nr_objects"},
        },
        {
            "name": "write_perf_plain",
            "scenario": "write_perf",
            "io_api": "plain",
            "verify": False,
            "key_prefix": "ciwrt",
            "nr_objects": 512,
            **{k: v for k, v in common.items() if k != "nr_objects"},
        },
        {
            "name": "read_perf_plain",
            "scenario": "read_perf",
            "io_api": "plain",
            "verify": True,
            "pattern": "0xcd",
            "prepare_mode": "auto",
            "key_prefix": "cird",
            "nr_objects": 256,
            "prepare_objects": 256,
            **{k: v for k, v in common.items() if k != "nr_objects"},
        },
    ]
    if skip_zcopy:
        return cases
    cases.extend(
        [
            {
                "name": "write_perf_zcopy",
                "scenario": "write_perf",
                "io_api": "zcopy",
                "verify": False,
                "key_prefix": "cizwrt",
                "nr_objects": 512,
                **{k: v for k, v in common.items() if k != "nr_objects"},
            },
            {
                "name": "read_perf_zcopy",
                "scenario": "read_perf",
                "io_api": "zcopy",
                "verify": True,
                "pattern": "0xef",
                "prepare_mode": "auto",
                "key_prefix": "cizrd",
                "nr_objects": 256,
                "prepare_objects": 256,
                **{k: v for k, v in common.items() if k != "nr_objects"},
            },
        ]
    )
    return cases


def case_cli_args(case: Dict[str, Any], args: argparse.Namespace, json_path: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(BENCH_SCRIPT),
        f"--scenario={case['scenario']}",
        f"--io-api={case['io_api']}",
        f"--local-hostname={args.local_hostname}",
        f"--metadata-server={args.metadata_server}",
        f"--master-server={args.master_server}",
        f"--protocol={args.protocol}",
        f"--device-name={args.device_name}",
        f"--global-segment-size={args.global_segment_size}",
        f"--local-buffer-size={args.local_buffer_size}",
        f"--nr-objects={case['nr_objects']}",
        f"--batch-size={case['batch_size']}",
        f"--value-size={case['value_size']}",
        f"--key-size={case['key_size']}",
        f"--key-prefix={case['key_prefix']}",
        f"--memory-replica-num={case['memory_replica_num']}",
        f"--nof-replica-num={case['nof_replica_num']}",
        f"--numjobs={case['numjobs']}",
        f"--iodepth={case['iodepth']}",
        f"--json-output={json_path}",
        "--phase-gap-mode=none",
        "--log-level=INFO",
    ]
    runtime = args.runtime if args.runtime > 0 else case.get("runtime", 0)
    if runtime:
        cmd.append(f"--runtime={runtime}")
    if case.get("prepare_mode"):
        cmd.append(f"--prepare-mode={case['prepare_mode']}")
    if case.get("prepare_objects"):
        cmd.append(f"--prepare-objects={case['prepare_objects']}")
    if case.get("verify"):
        cmd.append("--verify")
    if case.get("pattern"):
        cmd.append(f"--pattern={case['pattern']}")
    return cmd


def load_thresholds(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("cases", data)


def find_phase(result: Dict[str, Any], phase_name: str) -> Optional[Dict[str, Any]]:
    for phase in result.get("phases", []):
        if phase.get("name") == phase_name:
            return phase
    overall = result.get("overall")
    if overall and overall.get("name") == phase_name:
        return overall
    return None


def check_thresholds(
    case_name: str,
    result: Dict[str, Any],
    thresholds: Dict[str, Any],
) -> List[str]:
    rules = thresholds.get(case_name)
    if not rules:
        return [f"{case_name}: no thresholds configured"]
    phase_name = rules.get("phase")
    phase = find_phase(result, phase_name) if phase_name else result.get("overall")
    if phase is None:
        return [f"{case_name}: missing phase '{phase_name}' in benchmark JSON"]

    failures: List[str] = []

    def require_max(metric: str, key: str) -> None:
        if key not in rules:
            return
        actual = float(phase.get(metric, 0.0))
        limit = float(rules[key])
        if actual > limit:
            failures.append(
                f"{case_name}.{phase_name}: {metric}={actual} exceeds max {limit}"
            )

    def require_min(metric: str, key: str) -> None:
        if key not in rules:
            return
        actual = float(phase.get(metric, 0.0))
        limit = float(rules[key])
        if actual < limit:
            failures.append(
                f"{case_name}.{phase_name}: {metric}={actual:.2f} below min {limit}"
            )

    require_max("verify_failures", "max_verify_failures")
    require_max("misses", "max_misses")
    require_max("failed_requests", "max_failed_requests")
    require_min("successful_kvs", "min_successful_kvs")
    require_min("MiB_per_sec", "min_MiB_per_sec")
    require_min("kv_per_sec", "min_kv_per_sec")
    require_min("req_per_sec", "min_req_per_sec")
    return failures


def append_step_summary(lines: List[str]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_case(
    case: Dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    thresholds: Dict[str, Any],
) -> Dict[str, Any]:
    case_name = case["name"]
    json_path = output_dir / f"{case_name}.json"
    log_path = output_dir / f"{case_name}.log"
    cmd = case_cli_args(case, args, json_path)
    print(f"==> Running {case_name}")
    print(" ".join(cmd), flush=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
            env={**os.environ, "MOONCAKE_IS_IN_CI": "1" if is_in_ci() else "0"},
        )
    elapsed = time.time() - started
    if proc.returncode != 0:
        print(log_path.read_text(encoding="utf-8", errors="replace"))
        raise RuntimeError(f"{case_name} failed with exit code {proc.returncode}")
    if not json_path.exists():
        raise RuntimeError(f"{case_name} did not produce JSON output at {json_path}")
    result = json.loads(json_path.read_text(encoding="utf-8"))
    gate_failures = check_thresholds(case_name, result, thresholds)
    phase_name = thresholds.get(case_name, {}).get("phase", "overall")
    phase = find_phase(result, phase_name) or result.get("overall", {})
    return {
        "name": case_name,
        "elapsed_sec": elapsed,
        "phase": phase_name,
        "MiB_per_sec": float(phase.get("MiB_per_sec", 0.0)),
        "kv_per_sec": float(phase.get("kv_per_sec", 0.0)),
        "lat_p50_ms": float(phase.get("lat_p50_ms", 0.0)),
        "lat_p99_ms": float(phase.get("lat_p99_ms", 0.0)),
        "failed_requests": int(phase.get("failed_requests", 0)),
        "verify_failures": int(phase.get("verify_failures", 0)),
        "gate_failures": gate_failures,
        "json_path": str(json_path),
        "log_path": str(log_path),
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not BENCH_SCRIPT.exists():
        print(f"ERROR: missing benchmark script at {BENCH_SCRIPT}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(args.thresholds)

    selected = {
        name.strip() for name in args.cases.split(",") if name.strip()
    }
    cases = default_cases(skip_zcopy=args.skip_zcopy)
    if selected:
        cases = [case for case in cases if case["name"] in selected]
        missing = selected - {case["name"] for case in cases}
        if missing:
            print(f"ERROR: unknown cases: {sorted(missing)}", file=sys.stderr)
            return 2

    print(f"CI mode: {is_in_ci()}")
    print(f"Output directory: {output_dir}")
    print(f"Cases: {[case['name'] for case in cases]}")

    results: List[Dict[str, Any]] = []
    all_gate_failures: List[str] = []
    for case in cases:
        result = run_case(case, args, output_dir, thresholds)
        results.append(result)
        all_gate_failures.extend(result["gate_failures"])
        status = "FAIL" if result["gate_failures"] else "PASS"
        print(
            f"[{status}] {result['name']}: "
            f"MiB/s={result['MiB_per_sec']:.2f} "
            f"kv/s={result['kv_per_sec']:.2f} "
            f"p50={result['lat_p50_ms']:.3f}ms "
            f"p99={result['lat_p99_ms']:.3f}ms "
            f"({result['elapsed_sec']:.1f}s)"
        )
        for failure in result["gate_failures"]:
            print(f"  - {failure}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"results": results, "gate_failures": all_gate_failures}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "## Mooncake Store Performance CI",
        "",
        f"- CI mode: `{is_in_ci()}`",
        f"- Cases: {len(results)}",
        f"- Gate failures: {len(all_gate_failures)}",
        "",
        "| Case | Phase | MiB/s | kv/s | p50 (ms) | p99 (ms) | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        status = "FAIL" if result["gate_failures"] else "PASS"
        summary_lines.append(
            f"| `{result['name']}` | `{result['phase']}` | "
            f"{result['MiB_per_sec']:.2f} | {result['kv_per_sec']:.2f} | "
            f"{result['lat_p50_ms']:.3f} | {result['lat_p99_ms']:.3f} | {status} |"
        )
    if all_gate_failures:
        summary_lines.extend(["", "### Threshold failures", ""])
        summary_lines.extend(f"- {item}" for item in all_gate_failures)
    append_step_summary(summary_lines)

    if all_gate_failures:
        print("Store performance CI failed threshold checks:", file=sys.stderr)
        for item in all_gate_failures:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print(f"Store performance CI passed. Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
