---
name: mooncake-store-debug
description: 'Debug Mooncake store behavior. Use for master setup, metadata service issues, lease TTL mismatches, segment allocation, eviction, object lifecycle bugs, and distributed object store tests.'
argument-hint: 'Describe the failing store path, error code, or test case.'
---

# Mooncake Store Debug

Use this skill for work centered on `mooncake-store/`, store-backed Python APIs, and master-client flows.

## When To Use

- `mooncake_master` startup or connectivity fails
- Object store tests fail
- Lease expiry or TTL behavior looks wrong
- Segment allocation, replication, or eviction is involved
- Metadata server and store control flow are inconsistent

## Procedure

1. Confirm whether the failure is in master startup, metadata coordination, object lifecycle, or client access.
2. Check lease TTL assumptions on both the master side and the tests.
3. Reproduce with the smallest relevant store test before running the full suite.
4. If the change affects both store and bindings, hand off to binding-sync validation next.
5. If the change affects end-to-end semantics, include [scripts/run_tests.sh](../../../scripts/run_tests.sh) in validation.

## Useful References

- Architecture and master flags: [CLAUDE.md](../../../CLAUDE.md)
- Existing runtime diagnostics: [mooncake-troubleshoot skill](../../../.claude/skills/mooncake-troubleshoot/SKILL.md)
- End-to-end store tests: [scripts/run_tests.sh](../../../scripts/run_tests.sh)