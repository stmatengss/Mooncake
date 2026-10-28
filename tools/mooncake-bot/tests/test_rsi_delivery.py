import asyncio
import json
from pathlib import Path

from mooncake_bot import rsi
from mooncake_bot.config import Config
from mooncake_bot.rsi import Settings, Worker


def test_promoted_instruction_is_saved_with_lineage_and_benchmark_identity(tmp_path):
    config = Config(root=tmp_path, backend="kimi", installation_ids=[1])
    worker = Worker(config, Settings())
    cases = json.loads(Path(rsi.__file__).with_name("rsi_cases.json").read_text())
    labels = {case["source"]: case["expected_lines"] for case in cases}

    async def review(context, directory, instruction):
        expected = labels[context["files"][0]["content"]]
        findings = []
        if instruction and expected:
            findings = [{"path": "fixture.py", "line": expected[0]}]
        return {"findings": findings}

    async def ask(*args):
        return {"instruction": "Check the stated function contract before reporting a defect."}

    worker.review, worker.ask = review, ask
    asyncio.run(worker.improve_policy())
    active = json.loads(config.path("policy_file").read_text())
    assert active["candidate"]["holdout"] == {"tp": 2, "fp": 0, "fn": 0}
    assert active["baseline"]["holdout"] == {"tp": 0, "fp": 0, "fn": 2}
    assert len(active["benchmark_sha256"]) == len(active["parent_hash"]) == 64
    assert worker.state.recent()[0]["state"] == "promoted"


def test_publisher_only_creates_new_candidate_branch_and_draft(tmp_path):
    requests = []

    class GitHub:
        prefix = "/repos/stmatengss/Mooncake"

        def token(self, installation, write=False, contents_write=False):
            assert installation == 1 and write and contents_write
            return "installation-token"

        def get(self, path, task, **kwargs):
            if "/git/commits/" in path:
                return {"tree": {"sha": "base-tree"}}
            return {"sha": "base"}

        def request(self, method, path, token, **kwargs):
            requests.append((method, path, kwargs["json"]))
            assert method == "POST"
            if path.endswith("/pulls"):
                return {"number": 7, "html_url": "https://example.invalid/7"}
            return {"sha": "generated"}

    config = Config(root=tmp_path, backend="kimi", installation_ids=[1])
    worker = Worker(config, Settings(targets=["scripts/a.py"]), github=GitHub())
    result = worker.publish(
        "base",
        "scripts/a.py",
        "pass\n",
        {"summary": "Fix", "test_plan": "test"},
        "a" * 32,
        tmp_path,
    )
    assert result["pr_number"] == 7
    assert requests[0][2]["base_tree"] == "base-tree"
    assert requests[1][2]["parents"] == ["base"]
    assert requests[2][2]["ref"] == "refs/heads/rsi/project-" + "a" * 32
    assert requests[3][2]["draft"] is True and requests[3][2]["base"] == "main"
    assert "No repository code, unit tests" in requests[3][2]["body"]


def test_fixed_benchmark_defects_and_clean_controls_have_executable_evidence():
    cases = json.loads(Path(rsi.__file__).with_name("rsi_cases.json").read_text())
    functions = []
    for case in cases:
        namespace = {}
        exec(compile(case["source"], "fixed-benchmark", "exec"), namespace)
        functions.append(next(value for key, value in namespace.items() if key != "__builtins__"))
    try:
        functions[0]([1])
    except IndexError:
        pass
    else:
        raise AssertionError("Off-by-one fixture must fail")
    assert functions[1]([0, 1], 1) is False
    assert functions[2]([1], 2) is None and functions[3]([1, 2]) == 3
    try:
        functions[4]([])
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("Empty-mean fixture must fail")
    values = [1]
    functions[5](values, 2)
    assert values == [1, 2]
    assert functions[6]([]) == 0
    values = [1]
    assert functions[7](values, 2) == [1, 2] and values == [1]
