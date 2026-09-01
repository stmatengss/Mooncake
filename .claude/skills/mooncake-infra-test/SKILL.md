---
name: mooncake-infra-test
description: 'Run and debug Mooncake local infrastructure and test orchestration. Use for metadata server setup, local CI reproduction, ctest selection, scripts/run_tests.sh flows, and workflow-script consistency checks.'
argument-hint: 'Describe whether you need focused local repro, CI parity, or environment diagnostics.'
---

# Mooncake Infra And Test

Use this skill for local infrastructure setup and test orchestration across Mooncake modules.

## When To Use

- You need to reproduce CI locally
- Metadata server or local service orchestration is part of the failure
- You are choosing between `ctest`, targeted Python tests, and `scripts/run_tests.sh`
- Scripts or GitHub workflows changed

## Procedure

1. Decide whether the task is module-local validation or workflow-level reproduction.
2. For focused C++ or integration checks, start from the build directory and use `ctest`.
3. For end-to-end Python-backed behavior, use [scripts/run_tests.sh](../../../scripts/run_tests.sh).
4. When scripts or workflows changed, compare the intended local path against [ci.yml](../../../.github/workflows/ci.yml).
5. If environment setup is part of the issue, verify metadata server, `LD_LIBRARY_PATH`, and relevant `MC_*` variables before changing code.

## Useful References

- End-to-end script: [scripts/run_tests.sh](../../../scripts/run_tests.sh)
- CI contract: [ci.yml](../../../.github/workflows/ci.yml)
- Existing diagnostics: [mooncake-troubleshoot skill](../mooncake-troubleshoot/SKILL.md)