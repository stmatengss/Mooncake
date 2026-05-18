# RFC: Simplify Mooncake Store for Maintainability

## Status

Draft for GitHub RFC discussion.

For a shorter GitHub issue-ready draft, see
docs/source/design/mooncake-store-maintainability-rfc-issue.md.

## Summary

This RFC proposes a maintainability-focused simplification of Mooncake Store.
The goal is to reduce code volume, remove repeated glue code, and make the
component easier to evolve without changing its core architecture:

- Master remains the control plane for metadata, allocation, eviction, and task
  orchestration.
- Clients continue to own the data plane through direct transfer paths.
- Public Store semantics such as Get, Put, Remove, MountSegment, and task APIs
  remain stable unless explicitly called out.

The proposal does not attempt to redesign the whole Store. It instead targets a
small number of high-cost maintenance hotspots that currently create most of the
code duplication and change amplification:

- duplicated RPC binding and forwarding code
- oversized configuration objects and repeated config conversion logic
- a monolithic Client runtime that mixes control plane, background loops, and
  segment lifecycle management
- compatibility layers and state models that keep old paths alive longer than
  necessary
- tests whose shape mirrors production code duplication

## Motivation

Mooncake Store has grown quickly to support:

- default and HA master modes
- snapshot and restore
- offload and local disk paths
- task scheduling and execution
- embedded, real, dummy, and C-facing client surfaces

This feature growth is valuable, but it has also created several code patterns
that are expensive to maintain.

### Current pain points

1. RPC surface duplication is spread across multiple files.

   The same Store RPC method list is effectively encoded in at least three
   places:

   - wrapper declarations in mooncake-store/include/rpc_service.h
   - client-side name traits in mooncake-store/src/master_client.cpp
   - server-side handler registration in mooncake-store/src/rpc_service.cpp

   Adding or renaming one RPC requires touching several files and keeping them
   manually synchronized.

2. Configuration is too flat and too repetitive.

   mooncake-store/include/master_config.h currently contains one large builder,
   one large runtime config, and conversion logic from WrappedMasterServiceConfig
   to MasterServiceConfig. This creates a high-field-count structure with weak
   module boundaries.

3. Client owns too many responsibilities.

   mooncake-store/src/client_service.cpp is more than 3000 lines and combines:

   - control plane interactions with master
   - transfer-engine setup and transport choices
   - task polling and task execution coordination
   - leader monitoring
   - storage heartbeat
   - segment mount and unmount lifecycle
   - hot cache lifecycle

   This makes the class difficult to reason about and risky to modify.

4. Compatibility and lifecycle logic is more complex than necessary.

   Some data structures and states preserve older behavior in ways that now
   increase branching and documentation cost. ReplicateConfig and ReplicaStatus
   are the main examples.

5. Tests duplicate construction and setup patterns.

   Large test files repeatedly build nearly identical master and client setups.
   This makes tests harder to read and also slows down refactors because helper
   patterns are not centralized.

## Goals

- Reduce change amplification for Store RPC evolution.
- Reduce the amount of hand-written forwarding, registration, and config-copy
  code.
- Split oversized classes along stable responsibilities.
- Keep behavior stable during the refactor.
- Create a phased migration plan that can land incrementally.

## Non-goals

- Redesigning Store architecture or changing the control-plane vs data-plane
  split.
- Replacing allocator backends or redefining allocation policy.
- Reworking HA election, snapshot backend semantics, or Transfer Engine APIs.
- Introducing user-visible API breaks in one step.

## Design principles

1. Remove repeated glue before changing behavior.
2. Prefer single-source registries over duplicated method lists.
3. Split by operational responsibility, not by arbitrary file size alone.
4. Preserve public APIs first, then deprecate carefully.
5. Let tests validate the migration path instead of rewriting everything at
   once.

## Proposal

### Workstream A: Unify the Store RPC method registry

#### Problem

The Store RPC method list is repeated across declarations, name lookup traits,
and server registration code.

#### Proposed change

Introduce one canonical RPC method registry, likely through an X-macro or a
small compile-time method list. Use it to derive:

- wrapper method declarations
- wrapper forwarding implementations
- client-side RPC name mapping
- server-side handler registration

#### Expected result

- fewer manual sync points when adding or renaming RPCs
- smaller master_client.cpp and rpc_service.cpp
- lower review risk for RPC-only changes

#### File-level change list

- mooncake-store/include/rpc_service.h
  - replace the repeated wrapper declarations with generated declarations from
    a single method list
- mooncake-store/src/rpc_service.cpp
  - replace repeated register_handler calls with registry-driven expansion
  - optionally replace repeated forwarding bodies with generated forwarding
- mooncake-store/src/master_client.cpp
  - replace manual RpcNameTraits specializations with a generated table or
    macro expansion
- proposed new file: mooncake-store/include/rpc_method_registry.h
  - canonical Store RPC method list

#### Risks

- macro-driven code can become opaque if overused
- debugging symbol names must remain readable

#### Mitigation

- keep the method registry small and explicit
- generate only repetitive glue, not business logic

### Workstream B: Decompose master configuration into coherent sub-configs

#### Problem

MasterServiceConfig mixes HA, snapshot, eviction, allocation, storage, quota,
task scheduling, and CXL settings in one flat object. The builder mirrors this
flatness, and WrappedMasterServiceConfig conversion repeats field assignment.

#### Proposed change

Split MasterServiceConfig into nested sub-configs while preserving an outer
top-level config object for compatibility. Candidate groups:

- LeaseConfig
- EvictionConfig
- SnapshotConfig
- HaConfig
- StorageConfig
- TaskConfig
- CxlConfig

Keep public construction backward-compatible in the first phase, then migrate
callers to scoped builders or helper constructors.

#### Expected result

- less code repetition in config conversion
- clearer ownership boundaries inside MasterService
- cleaner test configuration setup

#### File-level change list

- mooncake-store/include/master_config.h
  - introduce grouped config structs
  - shrink the flat builder surface
  - reduce duplicated aliases for backward compatibility over time
- mooncake-store/src/master_service.cpp
  - consume grouped config sections instead of a single flat blob
- mooncake-store/src/master.cpp
  - keep CLI or runtime config assembly in one place, but write into grouped
    sub-configs
- mooncake-store/include/master_service.h
  - update constructor documentation to reflect grouped config usage
- mooncake-store/tests/master_service_test.cpp
  - migrate repetitive config creation to helpers
- mooncake-store/tests/master_service_ssd_test.cpp
  - migrate repetitive config creation to helpers
- mooncake-store/tests/test_server_helpers.h
  - add shared helper constructors for common master configurations

#### Risks

- broad mechanical edits may create noisy diffs
- config serialization or snapshot restore assumptions may be implicit

#### Mitigation

- land sub-config introduction first without removing old fields immediately
- gate old-to-new conversion behind tests

### Workstream C: Split Client into facade plus runtime components

#### Problem

Client currently mixes API surface, connection management, segment lifecycle,
background loops, and task execution orchestration in one large implementation.

#### Proposed change

Keep Client as the public facade, but extract runtime responsibilities into a
small set of internal components:

- MasterConnectionManager
- ClientBackgroundRuntime
- SegmentMountRegistry
- HotCacheController
- TaskExecutionCoordinator

The public API remains on Client. Internally, Client becomes a coordinator
rather than the implementation site of every loop and lifecycle transition.

#### Expected result

- smaller and easier-to-review client_service.cpp
- clearer shutdown order and ownership boundaries
- easier isolated testing of background runtime behavior

#### File-level change list

- mooncake-store/include/client_service.h
  - keep public facade, reduce direct runtime member exposure
- mooncake-store/src/client_service.cpp
  - move background loop logic and lifecycle helpers into dedicated files
- proposed new file: mooncake-store/include/master_connection_manager.h
- proposed new file: mooncake-store/src/master_connection_manager.cpp
- proposed new file: mooncake-store/include/client_background_runtime.h
- proposed new file: mooncake-store/src/client_background_runtime.cpp
- proposed new file: mooncake-store/include/segment_mount_registry.h
- proposed new file: mooncake-store/src/segment_mount_registry.cpp
- proposed new file: mooncake-store/include/hot_cache_controller.h
- proposed new file: mooncake-store/src/hot_cache_controller.cpp
- proposed new file: mooncake-store/include/task_execution_coordinator.h
- proposed new file: mooncake-store/src/task_execution_coordinator.cpp
- mooncake-store/src/real_client.cpp
  - adapt construction to the new Client internals without changing external
    behavior
- mooncake-store/src/dummy_client.cpp
  - keep forwarding surface stable while Client internals change

#### Risks

- shutdown ordering bugs
- hidden dependencies between background loops and shared state

#### Mitigation

- move one responsibility at a time
- add focused tests for startup and teardown order before large extraction

### Workstream D: Simplify replica semantics and compatibility surface

#### Problem

Replica-related structures are carrying more compatibility and state detail than
the most common paths need.

Examples:

- ReplicateConfig keeps both preferred_segments and deprecated preferred_segment
- ReplicaStatus has more phases than most call paths differentiate on

#### Proposed change

1. Centralize backward-compatibility translation for ReplicateConfig into one
   helper.
2. Audit ReplicaStatus transitions and collapse states only if tests show the
   intermediate states are not externally meaningful.
3. Keep wire compatibility first, remove deprecated input only after callers are
   migrated.

#### Expected result

- fewer branching paths in allocation and replica lifecycle code
- cleaner documentation and fewer compatibility branches

#### File-level change list

- mooncake-store/include/replica.h
  - centralize compatibility handling for preferred_segment
  - document state transition invariants explicitly
- mooncake-store/src/master_service.cpp
  - simplify replica state transition checks where possible
- mooncake-store/src/client_service.cpp
  - remove duplicate compatibility normalization logic if present
- mooncake-store/include/types.h
  - verify any shared RPC-visible structs remain compatible during transition
- docs/source/design/mooncake-store.md
  - update documentation after semantic cleanup is completed

#### Risks

- state simplification can accidentally weaken safety checks

#### Mitigation

- treat this as a late-phase cleanup after RPC and Client refactors stabilize

### Workstream E: Rationalize Store test fixtures and setup helpers

#### Problem

Tests currently mirror production duplication. Repeated configuration and master
or client setup logic make the suite bulky and harder to evolve.

#### Proposed change

Add a small set of named helpers for common scenarios instead of building every
scenario inline. Candidate helper families:

- in-process master with memory-only storage
- master with SSD offload enabled
- snapshot-enabled master
- client with embedded mode defaults
- client with task execution enabled

#### Expected result

- shorter tests
- lower migration cost for config or lifecycle refactors
- fewer brittle test-specific config fragments

#### File-level change list

- mooncake-store/tests/test_server_helpers.h
  - centralize common master construction
- mooncake-store/tests/master_service_test.cpp
  - replace repeated inline builders with helpers
- mooncake-store/tests/master_service_ssd_test.cpp
  - replace repeated inline builders with helpers
- mooncake-store/tests/client_integration_test.cpp
  - move common client setup into helper utilities
- mooncake-store/tests/cxl_client_integration_test.cpp
  - move common client setup into helper utilities
- mooncake-store/tests/ha/snapshot/master_service_test_for_snapshot_base.h
  - centralize snapshot master construction
- mooncake-store/tests/ha/snapshot/snapshot_child_process_test.cpp
  - remove repeated config assembly where possible

#### Risks

- over-abstracted test helpers can obscure scenario-specific intent

#### Mitigation

- keep helpers scenario-based, not generic factories with too many arguments

## Phased implementation plan

### Phase 0: Guardrails and inventory

Scope:

- document invariants for RPC behavior, config defaults, and client shutdown
  behavior
- add or identify coverage for master-client RPC registration, config defaults,
  and client teardown paths

Primary files:

- mooncake-store/tests/master_service_test.cpp
- mooncake-store/tests/client_integration_test.cpp
- mooncake-store/tests/test_server_helpers.h

Exit criteria:

- key behavior is covered before structural changes start

### Phase 1: Remove RPC glue duplication

Scope:

- introduce rpc_method_registry.h
- generate wrapper declarations, name mapping, and handler registration from the
  same source

Primary files:

- mooncake-store/include/rpc_service.h
- mooncake-store/src/rpc_service.cpp
- mooncake-store/src/master_client.cpp
- mooncake-store/include/rpc_method_registry.h

Exit criteria:

- adding one new RPC requires editing only the canonical method list and any
  real business logic implementation

### Phase 2: Group config and shrink config-copy logic

Scope:

- introduce grouped sub-configs
- keep compatibility shims temporarily
- migrate tests to use shared config helpers

Primary files:

- mooncake-store/include/master_config.h
- mooncake-store/src/master.cpp
- mooncake-store/src/master_service.cpp
- mooncake-store/tests/test_server_helpers.h
- mooncake-store/tests/master_service_test.cpp

Exit criteria:

- config responsibilities are grouped and repeated field-copy code is reduced

### Phase 3: Split Client runtime responsibilities

Scope:

- extract background runtime and segment lifecycle helpers
- keep public Client API stable

Primary files:

- mooncake-store/include/client_service.h
- mooncake-store/src/client_service.cpp
- mooncake-store/src/real_client.cpp
- mooncake-store/src/dummy_client.cpp
- new internal runtime files introduced in this RFC

Exit criteria:

- client_service.cpp no longer owns all background loops directly
- startup and shutdown sequencing is easier to test independently

### Phase 4: Cleanup replica compatibility and test shape

Scope:

- centralize ReplicateConfig compatibility handling
- audit ReplicaStatus transitions
- replace repeated inline test setup with named fixtures

Primary files:

- mooncake-store/include/replica.h
- mooncake-store/src/master_service.cpp
- mooncake-store/tests/*

Exit criteria:

- compatibility handling is localized
- tests no longer duplicate large setup fragments for common scenarios

## File-level execution checklist

This checklist is intentionally concrete and is meant to guide implementation
work item by work item.

### Core RPC layer

- mooncake-store/include/rpc_service.h
- mooncake-store/src/rpc_service.cpp
- mooncake-store/src/master_client.cpp
- new: mooncake-store/include/rpc_method_registry.h

### Master config and startup

- mooncake-store/include/master_config.h
- mooncake-store/include/master_service.h
- mooncake-store/src/master.cpp
- mooncake-store/src/master_service.cpp

### Client runtime split

- mooncake-store/include/client_service.h
- mooncake-store/src/client_service.cpp
- mooncake-store/src/real_client.cpp
- mooncake-store/src/dummy_client.cpp
- mooncake-store/src/store_c.cpp
- new: mooncake-store/include/master_connection_manager.h
- new: mooncake-store/src/master_connection_manager.cpp
- new: mooncake-store/include/client_background_runtime.h
- new: mooncake-store/src/client_background_runtime.cpp
- new: mooncake-store/include/segment_mount_registry.h
- new: mooncake-store/src/segment_mount_registry.cpp
- new: mooncake-store/include/hot_cache_controller.h
- new: mooncake-store/src/hot_cache_controller.cpp
- new: mooncake-store/include/task_execution_coordinator.h
- new: mooncake-store/src/task_execution_coordinator.cpp

### Replica and shared types

- mooncake-store/include/replica.h
- mooncake-store/include/types.h
- mooncake-store/include/rpc_types.h
- mooncake-store/src/master_service.cpp

### Tests and fixtures

- mooncake-store/tests/test_server_helpers.h
- mooncake-store/tests/master_service_test.cpp
- mooncake-store/tests/master_service_ssd_test.cpp
- mooncake-store/tests/client_integration_test.cpp
- mooncake-store/tests/cxl_client_integration_test.cpp
- mooncake-store/tests/ha/snapshot/master_service_test_for_snapshot_base.h
- mooncake-store/tests/ha/snapshot/snapshot_child_process_test.cpp

### Documentation updates after implementation

- docs/source/design/mooncake-store.md
- docs/source/deployment/mooncake-store-deployment-guide.md
- docs/source/http-api-reference/http-service.md
- docs/source/python-api-reference/mooncake-store.md

## Compatibility and migration

- Keep Store public APIs stable during Phase 1 to Phase 3.
- Preserve existing RPC names during the registry refactor.
- Keep old config entry points temporarily while new grouped config types are
  introduced.
- Keep deprecated ReplicateConfig inputs readable until all callers are updated.

## Validation plan

Minimum validation expected for each phase:

- existing Store unit tests continue to pass
- master-client RPC integration tests continue to pass
- snapshot-related tests pass unchanged unless the phase directly modifies
  snapshot code paths
- embedded, real, and dummy client flows still initialize and shut down cleanly

Recommended validation focus by phase:

- Phase 1: RPC method registration and invocation parity
- Phase 2: config default parity and startup behavior parity
- Phase 3: startup, shutdown, and background loop lifecycle ordering
- Phase 4: replica transition correctness and compatibility behavior

## Alternatives considered

### Alternative 1: Leave structure as-is and only update docs

Rejected because the main problem is not documentation quality alone. The code
currently requires repeated edits for routine feature work.

### Alternative 2: Full rewrite of Store control plane internals

Rejected because it is too risky and unnecessary. The highest-value problems are
localized to glue duplication, config shape, and oversized runtime ownership.

### Alternative 3: Defer all cleanup until a larger architecture change

Rejected because current duplication already increases the cost of normal
feature delivery.

## Open questions

1. Should the canonical RPC registry be implemented with X-macros, constexpr
   descriptors, or a smaller generated-code step?
2. Which config groups should remain constructible from CLI flags directly, and
   which should be internal-only?
3. Should RealClient and DummyClient remain fully separate top-level types after
   Client runtime extraction, or should part of their transport/setup logic be
   unified later?
4. Is ReplicaStatus simplification worth doing at all, or is compatibility
   normalization alone enough?

## Recommendation

Start with Workstream A and Workstream B.

These two workstreams have the best ratio of risk to return:

- they reduce a large amount of repeated glue code
- they create better boundaries for later Client extraction work
- they can be landed incrementally without changing Store semantics

Only after those are stable should the project move to Client runtime splitting
and replica compatibility cleanup.