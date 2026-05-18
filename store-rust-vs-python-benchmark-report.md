# Mooncake Store Rust API vs Python API Benchmark Report

## Overview

This report summarizes a local benchmark comparison between the Mooncake Store Rust API and the Python API on the main branch.

The goal was to:

1. Validate that the Rust Store API can be exercised against a live Mooncake master and metadata service.
2. Measure comparable put/get performance for the Rust and Python interfaces.
3. Document branch-specific caveats observed during validation.

All benchmark runs were executed in an isolated main worktree so the current working branch and uncommitted changes in the primary workspace were not affected.

## Executive Summary

The Rust API worked correctly in end-to-end live validation after its native link dependencies were supplied explicitly.

For medium and large payloads, the Rust API outperformed the Python API, especially on put throughput:

- At 64 KiB, Rust put throughput was about 45.1% higher than Python.
- At 1 MiB, Rust put throughput was about 37.8% higher than Python.
- Rust get throughput was also higher for 64 KiB and 1 MiB payloads, though with a smaller advantage.

For very small payloads, the picture was mixed:

- At 4 KiB, Rust put was slightly faster than Python.
- At 4 KiB, Rust get was slower than Python.

An important implementation caveat was also identified on main:

- The Rust crate itself builds successfully.
- The default Rust example build path does not link successfully unless a larger set of native dependencies is supplied than what is currently declared in the Rust build script.

## Scope

This comparison focused on standard synchronous store operations:

- put
- get

It did not cover:

- zero-copy APIs
- batch APIs
- tensor APIs
- multi-client contention
- RDMA paths
- persistent storage paths

The benchmark was intentionally limited to a TCP-only local setup so that Rust and Python were measured under the same runtime conditions.

## Test Environment

### Branch and workspace strategy

- Source branch under test: main
- Validation method: isolated git worktree based on main
- Reason: the primary workspace contained unrelated uncommitted changes and was not suitable for an in-place branch switch

### Service configuration

The following live services were used for both Rust and Python runs:

- Mooncake master service: 127.0.0.1:50051
- Built-in HTTP metadata server: http://127.0.0.1:8080/metadata
- Metrics port: 9103
- Transport protocol: tcp
- Local hostname: localhost

### Store client configuration

The same client-side settings were used for both language bindings:

- global_segment_size: 512 MiB
- local_buffer_size: 128 MiB
- protocol: tcp
- device_name: empty
- master_server_addr: 127.0.0.1:50051

## Validation Performed

### Rust API validation on main

The following checks were performed for the Rust interface:

1. Rust library build on main
2. Rust example build on main
3. Live end-to-end execution against a running Mooncake master and metadata server

Observed outcome:

- The Rust library build succeeded.
- The Rust example reached successful live round-trip operation after native link dependencies were provided explicitly.
- The example completed successful create, setup, put, get, size query, and existence checks.

One remove call for example-key returned error code -706 during the basic usage run. This is consistent with lease-related behavior and did not affect the successful validation of the main put/get path.

### Python API validation

The Python API was validated through a direct live setup using the installed mooncake.store binding and then used for the benchmark runs.

## Important Main Branch Caveat

On main, the Rust build script in mooncake-store/rust/build.rs does not declare all native libraries required for successful end-to-end linking of the example path.

The default script declares only a subset of dependencies such as:

- mooncake_store
- transfer_engine
- stdc++
- glog
- gflags
- pthread
- xxhash

However, successful linking in this environment additionally required native dependencies including:

- asio
- base
- cachelib_memory_allocator
- numa
- ibverbs
- jsoncpp
- curl
- etcd_wrapper
- hiredis
- liburing

This means the Rust binding on main is functionally usable, but its default standalone example build path is not currently self-sufficient.

## Benchmark Methodology

### Operation model

Only standard put and get operations were measured.

- Setup time was excluded.
- Warmup iterations were executed before timed iterations.
- Payload content was deterministic and identical between Rust and Python runs.
- Data integrity was validated on every get.

### Payload sizes and iteration counts

Three payload sizes were measured:

- 4 KiB: 200 iterations, 10 warmup iterations
- 64 KiB: 100 iterations, 8 warmup iterations
- 1 MiB: 20 iterations, 4 warmup iterations

### Measurement metrics

The following metrics were recorded:

- Average put latency per operation, in milliseconds
- Average get latency per operation, in milliseconds
- Effective put throughput, in MB/s
- Effective get throughput, in MB/s

## Raw Benchmark Results

| Payload | Python Put Latency (ms) | Rust Put Latency (ms) | Python Put Throughput (MB/s) | Rust Put Throughput (MB/s) | Python Get Latency (ms) | Rust Get Latency (ms) | Python Get Throughput (MB/s) | Rust Get Throughput (MB/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 KiB | 0.101 | 0.094 | 38.50 | 41.74 | 0.073 | 0.091 | 53.43 | 42.89 |
| 64 KiB | 0.175 | 0.120 | 357.50 | 518.87 | 0.112 | 0.109 | 559.76 | 570.99 |
| 1 MiB | 1.017 | 0.738 | 983.16 | 1354.39 | 0.509 | 0.474 | 1963.63 | 2109.14 |

## Relative Difference

The table below expresses Rust relative to Python.

- Negative latency delta means Rust was faster.
- Positive throughput delta means Rust delivered higher throughput.

| Payload | Put Latency Delta | Put Throughput Delta | Get Latency Delta | Get Throughput Delta |
|---|---:|---:|---:|---:|
| 4 KiB | -6.9% | +8.4% | +24.7% | -19.7% |
| 64 KiB | -31.4% | +45.1% | -2.7% | +2.0% |
| 1 MiB | -27.4% | +37.8% | -6.9% | +7.4% |

## Analysis

### 4 KiB payload

For the smallest tested payload, Rust showed a small advantage on put but a disadvantage on get.

- Rust put latency was 6.9% lower.
- Rust put throughput was 8.4% higher.
- Rust get latency was 24.7% higher.
- Rust get throughput was 19.7% lower.

This suggests that for very small payloads, fixed request-path overheads dominate overall performance more than raw data movement cost. In this regime, the Python path did not exhibit a clear disadvantage.

### 64 KiB payload

At 64 KiB, Rust clearly outperformed Python on put and was approximately on par on get.

- Rust put latency was 31.4% lower.
- Rust put throughput was 45.1% higher.
- Rust get latency was 2.7% lower.
- Rust get throughput was 2.0% higher.

This is the first size where the Rust path shows a strong practical advantage while maintaining comparable get behavior.

### 1 MiB payload

At 1 MiB, Rust retained a clear lead.

- Rust put latency was 27.4% lower.
- Rust put throughput was 37.8% higher.
- Rust get latency was 6.9% lower.
- Rust get throughput was 7.4% higher.

This indicates that the Rust binding is better suited for larger-value store operations where serialization and per-call overhead become a smaller fraction of end-to-end time.

## Interpretation

Based on this local TCP benchmark:

- If the workload is dominated by medium to large values, the Rust API is the stronger interface from a throughput perspective.
- If the workload is dominated by very small reads, Python remains competitive and may even perform better on get latency.
- The largest advantage observed for Rust was on put throughput, not get throughput.

In practical terms, the Rust API looks especially attractive for high-rate object writes and larger object round-trip workloads.

## Limitations

This report should be interpreted with the following constraints in mind:

1. The benchmark was local and TCP-only.
2. It did not measure RDMA, GPU, or zero-copy flows.
3. It was not a clean rebuild of every native dependency directly from the isolated main worktree.
4. The Rust example path required manual supplementation of native link dependencies because the current main branch Rust build script is incomplete for standalone example linking.
5. Results were gathered on one machine and should be treated as a local reference point, not a universal performance guarantee.

## Conclusions

The main branch Rust Store binding is functionally viable, but the out-of-the-box native link configuration for the example path is incomplete.

Once linked correctly, the Rust API demonstrates stronger performance than the Python API for medium and large payloads, with the strongest gains appearing on put throughput.

Summary of the benchmark outcome:

- Rust is slightly better on 4 KiB put, but worse on 4 KiB get.
- Rust is substantially better on 64 KiB put and modestly better on 64 KiB get.
- Rust is substantially better on 1 MiB put and moderately better on 1 MiB get.

## Recommended Follow-up

The next practical step is to fix the native dependency declarations in mooncake-store/rust/build.rs so that the Rust example and benchmark path can be built and run directly on main without manual linker flags.

After that, the benchmark should ideally be repeated under:

- a clean rebuild from main
- multi-client load
- zero-copy APIs
- larger payload ranges
- RDMA-enabled transport, if relevant to the target deployment