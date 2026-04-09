#!/usr/bin/env python3

import json
import re
import sys
from typing import Any


DENY_PATTERNS = [
    (re.compile(r"\bgit\s+reset\s+--hard\b"), "Refuse destructive git reset in workspace hooks."),
    (re.compile(r"\bgit\s+checkout\s+--(?:\s|$)"), "Refuse destructive checkout restore in workspace hooks."),
    (re.compile(r"\brm\s+-rf\s+(/|~|\$HOME)(?:\s|$)"), "Refuse destructive recursive deletion outside controlled build paths."),
]

ASK_PATTERNS = [
    (
        re.compile(r"\bclang-format\b"),
        "Prefer scripts/code_format.sh so formatting stays scoped to changed files or explicit all-file mode.",
    ),
    (
        re.compile(r"\brm\s+-rf\s+.*\bbuild[\w/-]*"),
        "Build cleanup is high impact. Confirm the cleanup is intentional before proceeding.",
    ),
    (
        re.compile(r"\bfind\b.*\|\s*xargs\s+clang-format\b"),
        "Broad formatting sweeps should go through scripts/code_format.sh for consistency.",
    ),
]


def find_command(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("command", "tool_input", "toolInput", "input", "arguments", "params"):
            if key in payload:
                value = payload[key]
                if isinstance(value, str):
                    return value
                nested = find_command(value)
                if nested:
                    return nested
        for value in payload.values():
            nested = find_command(value)
            if nested:
                return nested
    elif isinstance(payload, list):
        for item in payload:
            nested = find_command(item)
            if nested:
                return nested
    return ""


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        payload = {}

    command = find_command(payload)
    if not command:
        json.dump({"continue": True}, sys.stdout)
        return 0

    for pattern, reason in DENY_PATTERNS:
        if pattern.search(command):
            json.dump(
                {
                    "continue": False,
                    "stopReason": reason,
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    },
                },
                sys.stdout,
            )
            return 2

    for pattern, reason in ASK_PATTERNS:
        if pattern.search(command):
            json.dump(
                {
                    "continue": True,
                    "systemMessage": reason,
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "ask",
                        "permissionDecisionReason": reason,
                    },
                },
                sys.stdout,
            )
            return 0

    json.dump({"continue": True}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())