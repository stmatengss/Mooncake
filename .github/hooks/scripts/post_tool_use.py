#!/usr/bin/env python3

import json
import sys
from typing import Any


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=True)
        except TypeError:
            return str(value)
    return str(value)


def collect_text(payload: Any) -> str:
    if isinstance(payload, dict):
        return "\n".join(collect_text(value) for value in payload.values())
    if isinstance(payload, list):
        return "\n".join(collect_text(item) for item in payload)
    return stringify(payload)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        payload = {}

    text = collect_text(payload)
    reminders = []

    if "mooncake-integration/" in text:
        reminders.append(
            "Bindings changed. Check whether matching C++ API or Python surface updates are needed and run a focused rebuild plus targeted tests."
        )
    if "mooncake-store/" in text or "mooncake-transfer-engine/" in text:
        reminders.append(
            "Core module source changed. Prefer module-scoped validation first, then use scripts/run_tests.sh if the change crosses module boundaries."
        )
    if "scripts/" in text or ".github/workflows/" in text:
        reminders.append(
            "Automation or workflow files changed. Recheck the local reproduction path against .github/workflows/ci.yml and scripts/run_tests.sh."
        )
    if "mooncake-wheel/" in text or "scripts/build_wheel.sh" in text:
        reminders.append(
            "Packaging surface changed. Revalidate wheel or install behavior and compare the path against the packaging flow in .github/workflows/ci.yml."
        )
    if "docs/" in text or "README.md" in text or "CONTRIBUTING.md" in text:
        reminders.append(
            "Documentation changed. Ensure the documented workflow still matches the actual scripts, CI steps, and repository conventions."
        )
    if "CMakeLists.txt" in text or ".cpp" in text or ".h" in text:
        reminders.append(
            "C/C++ files changed. Use scripts/code_format.sh for formatting before broader validation."
        )

    message = " ".join(dict.fromkeys(reminders))
    response = {"continue": True}
    if message:
        response["systemMessage"] = message

    json.dump(response, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())