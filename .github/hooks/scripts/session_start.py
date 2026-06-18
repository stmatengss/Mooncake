#!/usr/bin/env python3

import json
import sys


def main() -> int:
    try:
        json.load(sys.stdin)
    except json.JSONDecodeError:
        pass

    message = (
        "Mooncake harness active. Start from the nearest module boundary: "
        "transfer engine (`mooncake-transfer-engine/`), store (`mooncake-store/`), "
        "or bindings (`mooncake-integration/`). Use `scripts/code_format.sh` for C/C++ "
        "formatting, `ctest` for focused validation, and `scripts/run_tests.sh` when a "
        "change crosses module boundaries or affects Python integration."
    )

    json.dump({"continue": True, "systemMessage": message}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())