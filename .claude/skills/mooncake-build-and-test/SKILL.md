---
name: mooncake-build-and-test
description: 'Build and validate Mooncake changes. Use when modifying CMake, C++ source, packaging, test scripts, or when you need the right local build and test path for Mooncake.'
argument-hint: 'Describe the module you changed and whether you need focused or end-to-end validation.'
---

# Mooncake Build And Test

Use this skill to choose the smallest correct validation path for Mooncake.

## When To Use

- C++ source or headers changed
- `CMakeLists.txt` changed
- Packaging or wheel logic changed
- CI or local test orchestration changed
- You need to decide between `ctest` and `scripts/run_tests.sh`

## Procedure

1. Identify the narrowest module boundary touched: transfer engine, store, integration, wheel, or scripts.
2. If the change is C++-only and module-local, prefer the build directory plus `ctest`.
3. If the change crosses C++ and Python boundaries, include install or wheel validation.
4. If the change touches scripts, workflows, or bindings, compare the local path with `.github/workflows/ci.yml`.
5. If the change spans multiple modules or affects Python integration behavior, use [scripts/run_tests.sh](../../../scripts/run_tests.sh).

## Entry Points

- Focused C++ and integration validation: build directory `ctest`
- Broad Python integration validation: [scripts/run_tests.sh](../../../scripts/run_tests.sh)
- C/C++ formatting: [scripts/code_format.sh](../../../scripts/code_format.sh)
- CI source of truth: [ci.yml](../../../.github/workflows/ci.yml)

## Notes

- Prefer `MC_FORCE_TCP=true` for local, single-node validation unless RDMA is the subject of the task.
- Keep `DEFAULT_KV_LEASE_TTL` aligned with the master startup flags in local tests.