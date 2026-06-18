---
name: mooncake-docs-workflow
description: 'Handle Mooncake documentation and workflow updates. Use for README or docs changes, PR-title and RFC conventions, contributor workflow updates, and keeping docs aligned with scripts or CI.'
argument-hint: 'Describe whether the change is documentation-only, workflow-only, or tied to a code path.'
---

# Mooncake Docs And Workflow

Use this skill when the task is about developer-facing process, documentation, or workflow alignment.

## When To Use

- `README.md`, `docs/`, or contributor docs changed
- Team workflow or CI process changed
- You need to check PR-title conventions or RFC expectations
- Script behavior changed and docs need to be updated to match

## Procedure

1. Decide whether the change is pure documentation, workflow documentation, or code-adjacent behavior.
2. Keep docs aligned with existing scripts and CI instead of documenting hypothetical flows.
3. For workflow guidance, defer to [CLAUDE.md](../../../CLAUDE.md) and [ci.yml](../../../.github/workflows/ci.yml) as the operational source of truth.
4. If a behavior changed in code or scripts, update the matching docs in the same change when appropriate.
5. For larger architecture changes, preserve the RFC expectation already documented in the repository.

## Useful References

- Repository guidance: [CLAUDE.md](../../../CLAUDE.md)
- CI workflow: [ci.yml](../../../.github/workflows/ci.yml)
- End-to-end test orchestration: [scripts/run_tests.sh](../../../scripts/run_tests.sh)