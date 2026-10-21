---
name: mooncake-binding-sync
description: 'Keep Mooncake C++ and Python bindings in sync. Use when editing mooncake-integration, changing exposed C++ APIs, updating allocators, or debugging Python import and interface mismatches.'
argument-hint: 'Describe the API or binding surface that changed.'
---

# Mooncake Binding Sync

Use this skill when a change crosses the boundary between C++ source and Python-visible APIs.

## When To Use

- `mooncake-integration/` changed
- A C++ public API changed in transfer engine or store
- Python import or symbol exposure changed
- Allocator bindings or backend capability detection changed

## Procedure

1. Identify the source module that owns the API: transfer engine, store, or allocator.
2. Check whether the Python-visible name, signature, or behavior must change in `mooncake-integration/`.
3. Rebuild the affected targets before drawing conclusions from Python behavior.
4. Use targeted Python import or API tests before broader end-to-end suites.
5. Escalate to packaging validation only if the change affects wheel contents or install-time behavior.

## Useful References

- Architecture and binding guidance: [CLAUDE.md](../../../CLAUDE.md)
- Python API examples: [mooncake-api skill](../mooncake-api/SKILL.md)
- Build and validation path: [mooncake-build-and-test](../mooncake-build-and-test/SKILL.md)