# Mooncake Workspace Guidelines

Mooncake is a high-performance C++ and Python codebase for KV-cache-centric LLM serving. Prefer small, module-aware changes that respect the separation between transfer engine, store, Python bindings, and packaging.

## Architecture

- Treat `mooncake-transfer-engine/`, `mooncake-store/`, and `mooncake-integration/` as separate modules with explicit interfaces.
- When a C++ API changes in `mooncake-transfer-engine/` or `mooncake-store/`, check whether `mooncake-integration/` bindings must change too.
- Reuse existing scripts and documented entry points before inventing ad hoc commands.

## Build And Test

- Use `scripts/code_format.sh` for C/C++ formatting instead of broad manual `clang-format` sweeps.
- Use `scripts/run_tests.sh` for local end-to-end Python coverage when the change spans multiple modules.
- Use `ctest` from the build directory for focused C++ and integration validation.
- Treat `.github/workflows/ci.yml` as the source of truth for what CI actually builds and runs.

## Conventions

- Keep changes scoped to the module you are touching unless there is a verified interface dependency.
- Prefer `MC_FORCE_TCP=true` and local metadata server flows for local validation unless the task specifically requires RDMA.
- Keep `default_kv_lease_ttl` assumptions aligned between test orchestration and test code.
- Do not edit generated build output under `build*/` when the real fix belongs in source or scripts.

## Reference Docs

- See `CLAUDE.md` for architecture, build flags, environment variables, and operational constraints.
- See `.claude/skills/mooncake-api/SKILL.md` for Python API usage patterns.
- See `.claude/skills/mooncake-troubleshoot/SKILL.md` for deployment and runtime diagnostics.