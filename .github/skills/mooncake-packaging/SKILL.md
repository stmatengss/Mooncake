---
name: mooncake-packaging
description: 'Build and validate Mooncake packaging artifacts. Use for wheel assembly, bundled binaries, install/import checks, build_wheel.sh changes, and release-facing packaging issues.'
argument-hint: 'Describe the packaging surface that changed: wheel layout, shared objects, binaries, or install behavior.'
---

# Mooncake Packaging

Use this skill for work centered on `mooncake-wheel/` and packaging scripts.

## When To Use

- `mooncake-wheel/` changed
- `scripts/build_wheel.sh` changed
- Wheel contents or import behavior changed
- Shared object bundling or binary installation changed

## Procedure

1. Identify whether the issue is caused by packaging logic, binding outputs, or upstream build artifacts.
2. Use [scripts/build_wheel.sh](../../../scripts/build_wheel.sh) as the packaging source of truth.
3. Check whether the expected artifacts match the CI packaging path in [ci.yml](../../../.github/workflows/ci.yml).
4. Validate install or import behavior after rebuilding the affected artifacts.
5. If the root cause is in source or bindings, fix that layer before changing packaging glue.

## Useful References

- Packaging script: [scripts/build_wheel.sh](../../../scripts/build_wheel.sh)
- CI packaging flow: [ci.yml](../../../.github/workflows/ci.yml)
- Binding-related follow-up: [mooncake-binding-sync](../mooncake-binding-sync/SKILL.md)