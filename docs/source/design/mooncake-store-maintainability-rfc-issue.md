# [RFC] Simplify Mooncake Store for Maintainability

This file is a short issue-ready draft derived from the longer RFC in
docs/source/design/mooncake-store-maintainability-rfc.md.

## Suggested issue title

[RFC]: Simplify Mooncake Store for Maintainability

## Changes proposed

I'd like to start a discussion around a maintainability-focused cleanup of
Mooncake Store.

The goal is not to redesign the Store or change its core control-plane /
data-plane split. The goal is to reduce code duplication, make routine changes
less error-prone, and shrink a few maintenance hotspots that currently require
editing multiple files for one logical change.

The main problems this RFC is trying to address are:

- the Store RPC surface is duplicated across wrapper declarations, client-side
  RPC name mapping, and server-side handler registration
- master configuration is too flat and requires repeated field-copy logic
- Client currently owns too many responsibilities in one implementation unit
- some compatibility paths and replica lifecycle states are more complex than
  the common paths need
- tests repeat large amounts of setup logic, which makes refactors noisier than
  they need to be

## Why now

Mooncake Store has grown to support HA, snapshot and restore, offload, task
execution, embedded and standalone client modes, and more. The feature set is
useful, but the cost of maintaining glue code has also gone up.

Today, relatively small changes such as adding a new master RPC or adjusting a
master config flow can require touching several files that all encode the same
information in slightly different forms.

Before adding more features on top, it seems worth aligning on a cleanup plan
that makes the codebase easier to evolve.

## What I propose

I think the cleanup can be done incrementally in four workstreams.

### 1. Unify the Store RPC method registry

Instead of manually maintaining the Store RPC list in multiple places, define
one canonical method registry and derive the repetitive glue from it.

Expected outcome:

- fewer manual sync points when adding or renaming RPCs
- smaller glue-heavy files
- lower review risk for RPC-only changes

Main files involved:

- mooncake-store/include/rpc_service.h
- mooncake-store/src/rpc_service.cpp
- mooncake-store/src/master_client.cpp
- new internal registry header for Store RPC metadata

### 2. Group master configuration by responsibility

Split the current flat MasterServiceConfig into coherent sub-configs such as
lease, eviction, snapshot, HA, storage, task, and CXL settings.

Expected outcome:

- less repeated config-copy code
- clearer boundaries inside MasterService
- cleaner test setup

Main files involved:

- mooncake-store/include/master_config.h
- mooncake-store/src/master.cpp
- mooncake-store/src/master_service.cpp
- related master tests and helpers

### 3. Split Client into facade plus internal runtime components

Keep Client as the public surface, but move background loops and runtime
ownership into a few internal components, for example:

- master connection management
- background runtime and polling loops
- segment mount registry
- hot cache lifecycle
- task execution coordination

Expected outcome:

- a smaller and easier-to-review client implementation
- clearer startup and shutdown ownership
- easier targeted testing of runtime behavior

Main files involved:

- mooncake-store/include/client_service.h
- mooncake-store/src/client_service.cpp
- mooncake-store/src/real_client.cpp
- mooncake-store/src/dummy_client.cpp
- a small set of new internal runtime files

### 4. Localize compatibility logic and clean up tests

Centralize deprecated compatibility handling, especially around ReplicateConfig,
and reduce test duplication by introducing a few named setup helpers.

Expected outcome:

- fewer branching paths in common code
- easier follow-up refactors
- shorter and more intention-revealing tests

Main files involved:

- mooncake-store/include/replica.h
- mooncake-store/src/master_service.cpp
- mooncake-store/tests/test_server_helpers.h
- mooncake-store/tests/master_service_test.cpp
- other Store integration and snapshot tests

## What is not being proposed

This RFC is not proposing:

- a Store architecture rewrite
- a change to the control-plane vs data-plane split
- allocator backend replacement
- a redesign of HA or snapshot semantics
- a one-shot public API break

## Proposed rollout

I think this should land in phases rather than one large PR.

1. Start with RPC glue deduplication.
2. Then group master config and reduce config-copy logic.
3. Then split Client runtime responsibilities.
4. Finally, clean up compatibility paths and repeated test setup.

That ordering should let us remove low-risk duplication first, and only then
move into the more sensitive lifecycle refactors.

## Expected benefits

- lower maintenance cost for normal feature work
- fewer files touched for one logical change
- smaller review surface for RPC and config changes
- better boundaries for future work in Store without changing user-facing
  semantics

## Questions for feedback

I'd especially like feedback on these points before turning this into
implementation work:

1. Does the overall direction make sense, or is there a better simplification
   target to start with?
2. For the RPC registry, would the project prefer a macro-based approach, a
   constexpr descriptor table, or something else?
3. For master config, which parts should stay directly exposed to CLI-level
   assembly, and which parts should become internal-only groupings?
4. For Client splitting, which internal boundary is the safest first cut:
   connection management, background loops, or segment lifecycle?
5. Is ReplicaStatus simplification worth pursuing, or should this RFC stop at
   localizing compatibility handling?

## Initial recommendation

If there is rough agreement on the direction, my recommendation would be to
start with the two lowest-risk and highest-return pieces:

- unify the Store RPC method registry
- decompose MasterServiceConfig into coherent sub-configs

Those two changes should reduce a large amount of repeated glue while setting up
cleaner boundaries for the later Client runtime work.