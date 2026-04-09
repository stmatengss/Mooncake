# Agent Harness Guide

This document describes the shared agent harness that lives in the Mooncake repository. The goal is to give contributors a predictable way to use memory, skills, and hooks when working across Mooncake's C++ and Python modules.

## Why This Exists

Mooncake spans several tightly-coupled modules:

- `mooncake-transfer-engine/`
- `mooncake-store/`
- `mooncake-integration/`
- `mooncake-wheel/`
- `scripts/` and `.github/workflows/`

Without a shared harness, agent-assisted development tends to drift between modules, duplicate workflow knowledge, or skip project-specific validation paths. The harness keeps common guidance versioned with the repository.

## Shared Layers

The current harness is split into three shared layers.

### 1. Workspace Instructions

Global rules live in [/.github/copilot-instructions.md](../../../.github/copilot-instructions.md). These cover repository-wide behavior that should apply to almost every task, including:

- module boundaries
- preferred validation entry points
- formatting expectations
- high-level constraints such as not editing generated build outputs

### 2. File Instructions

Module-scoped instructions live in [/.github/instructions](../../../.github/instructions). These files add context only when the active task touches matching paths.

Current modules:

- transfer engine
- store
- integration
- packaging
- workflow and docs

This keeps the global instruction surface small while still giving module-specific guidance when it is relevant.

### 3. Skills

Reusable task workflows live in [/.github/skills](../../../.github/skills). These are designed for recurring tasks such as:

- build and test selection
- transfer-engine debugging
- store debugging
- binding synchronization
- packaging validation
- infrastructure and CI reproduction
- docs and workflow alignment

Existing `.claude/skills` content remains as a compatibility layer, while `.github/skills` is the shared workspace-facing layer. The repository now mirrors the shared core skills into `.claude/skills/` as well, so Claude-based workflows can discover the same task-level capabilities.

## Hooks

Shared hooks live in [/.github/hooks](../../../.github/hooks). They are intentionally small and auditable.

For Claude compatibility, the same shared hook commands are also wired through [/.claude/settings.json](../../../.claude/settings.json). Personal machine-specific allowances remain in `.claude/settings.local.json`.

The current hook set does three things:

1. `SessionStart` injects a short reminder about Mooncake module boundaries and preferred validation commands.
2. `PreToolUse` blocks or asks for confirmation on destructive or broad-impact commands.
3. `PostToolUse` reminds the agent about likely follow-up checks when code, packaging, scripts, or docs changed.

Hooks are not meant to replace skills. They enforce or remind. Skills carry the actual task workflow.

## Memory Model

Mooncake uses a mixed memory model.

- Repository-scoped facts are stored as repo memory and should capture stable, verified knowledge such as module boundaries, build or test entry points, and synchronization rules.
- Personal preferences and machine-specific habits should stay local and should not be committed as shared repository policy.

Examples of good shared memory:

- `scripts/code_format.sh` is the preferred C/C++ formatting entry point.
- `scripts/run_tests.sh` is the broad Python-backed validation path.
- `mooncake-integration/` must stay aligned with public C++ API changes.

Examples of bad shared memory:

- one-off failures on a single machine
- temporary workaround commands
- stale environment assumptions that are not documented elsewhere

## Recommended Workflow

When using the harness for development work, follow this order:

1. Start from the nearest module boundary.
2. Let file instructions narrow the active context.
3. Use the smallest relevant skill for the task.
4. Validate with the narrowest correct path first.
5. Escalate to broader checks only if the change crosses modules or affects Python-facing behavior.

In practice this usually means:

- `ctest` for focused C++ or integration validation
- [scripts/run_tests.sh](../../../scripts/run_tests.sh) for broader Python-backed coverage
- [scripts/code_format.sh](../../../scripts/code_format.sh) for C/C++ formatting instead of ad hoc formatting sweeps
- [/.github/workflows/ci.yml](../../../.github/workflows/ci.yml) as the CI source of truth

## What To Extend Next

If the harness needs to grow, prefer the following order:

1. add or refine file instructions when a module needs more precise context
2. add a skill when a workflow repeats often enough to deserve a dedicated path
3. add repo memory only for stable, verified knowledge
4. add or tighten hooks only when deterministic guardrails are actually needed

Avoid putting everything into workspace instructions. That increases noise and makes the harness harder to maintain.

## Local Versus Shared Customization

The repository contains shared, team-facing customization under `.github/`. Local machine-specific automation should stay outside shared policy unless the team explicitly agrees to standardize it.

In particular:

- shared policy belongs in `.github/`
- shared Claude compatibility glue belongs in `.claude/settings.json` and mirrored `.claude/skills/`
- local machine permissions or personal shortcuts should stay local
- build artifacts and generated files should not become part of the harness itself

## Source Of Truth

The harness should stay aligned with the repository's existing operational guidance. The most important source documents are:

- [CLAUDE.md](../../../CLAUDE.md)
- [scripts/run_tests.sh](../../../scripts/run_tests.sh)
- [scripts/code_format.sh](../../../scripts/code_format.sh)
- [/.github/workflows/ci.yml](../../../.github/workflows/ci.yml)

If the harness and these documents disagree, fix the mismatch instead of letting both versions drift.