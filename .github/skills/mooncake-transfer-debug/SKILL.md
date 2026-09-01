---
name: mooncake-transfer-debug
description: 'Debug Mooncake transfer engine behavior. Use for TCP or RDMA transport issues, metadata connectivity, memory registration, handshake failures, topology selection, and transfer tests.'
argument-hint: 'Describe whether the issue is TCP or RDMA, and include the failing command or error string.'
---

# Mooncake Transfer Debug

Use this skill for transfer-engine scoped work in `mooncake-transfer-engine/` and transfer-related Python tests.

## When To Use

- Transfer tests fail
- Metadata connectivity breaks transfer initialization
- RDMA device selection or handshake fails
- Memory registration or buffer lifecycle looks wrong
- You need the safest local reproduction path

## Procedure

1. Decide whether the task is local TCP validation or true RDMA validation.
2. For local repro, prefer `MC_FORCE_TCP=true` and the existing transfer tests in `mooncake-wheel/tests/`.
3. Verify metadata server address and transport-specific environment variables before changing code.
4. If RDMA is involved, inspect device availability, active ports, GID configuration, and memory limits before assuming a code bug.
5. Validate with the narrowest relevant test before escalating to cross-module integration runs.

## Useful References

- Transfer API patterns: [mooncake-api skill](../../../.claude/skills/mooncake-api/SKILL.md)
- Troubleshooting checklist: [mooncake-troubleshoot skill](../../../.claude/skills/mooncake-troubleshoot/SKILL.md)
- Local integration path: [scripts/run_tests.sh](../../../scripts/run_tests.sh)
- Project-level transport variables: [CLAUDE.md](../../../CLAUDE.md)