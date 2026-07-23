# Mooncake Store Benchmarks

This directory contains benchmark tools for Mooncake Store internals.

## CI Performance Gate

GitHub Actions runs an **independent** Store performance job
(`store-performance-test` in `.github/workflows/ci.yml`) that does not share
the ASan Debug functional build. It follows the SGLang pattern of a separate
benchmark CI lane:

1. Reuse the Release-oriented wheel from `build-flags`
2. Drive `store_kv_bench.py` with CI-sized workloads
3. Enforce conservative floors from `ci_perf_thresholds.json`
4. Publish a GitHub Actions step summary and upload JSON/log artifacts

Local reproduction (with master + metadata server already running):

```bash
python scripts/ci/run_store_perf_ci.py --output-dir /tmp/store-perf-ci-results
```

Optional: skip zero-copy cases or select a subset:

```bash
python scripts/ci/run_store_perf_ci.py --skip-zcopy
python scripts/ci/run_store_perf_ci.py --cases verify_write_plain,write_perf_plain
```

## Allocation Strategy Benchmark

`allocation_strategy_bench` evaluates Store allocation behavior across segment
counts, replica counts, allocation strategies, and workload patterns.

Build the benchmark from an existing CMake build directory:

```bash
cmake --build build --target allocation_strategy_bench -j$(nproc)
```

### Size-Class Churn Fragmentation Benchmark

The `size_class_churn` workload measures fragmentation under mixed-size
KVCache-like allocation pressure. It pre-fills the simulated cluster when
`--prefill_pct` is set, then repeatedly allocates objects from weighted size
classes. On allocation failure it randomly evicts a fraction of live objects and
retries.

When prefill is enabled, the prefill attempt cap is auto-derived from target
utilization, total cluster capacity, weighted average object size, and replica
count, with a 5000-attempt minimum for small cases.

This is an allocation-strategy-layer benchmark. It complements the existing
`dsa` workload by adding explicit fragmentation sampling and configurable
weighted size-class patterns. It is not a replacement for `allocator_bench`,
which remains the low-level `OffsetAllocator` microbenchmark.

Run a small local validation:

```bash
./build/mooncake-store/benchmarks/allocation_strategy_bench \
  --workload=size_class_churn \
  --segment_capacity=1024 \
  --num_allocations=10000 \
  --prefill_pct=70
```

Run a larger baseline:

```bash
./build/mooncake-store/benchmarks/allocation_strategy_bench \
  --workload=size_class_churn \
  --segment_capacity=1024 \
  --num_allocations=100000 \
  --prefill_pct=80
```

Supported size-class patterns:

- `kv_mixed`: 4KB at 70%, 256KB at 20%, and 3.12MB at 10%.
- `dsa_pair`: 3.12MB KV pages at 50% and 643KB indexer entries at 50%.
- `all`: run both patterns.

Key output columns:

- `Throughput`, `Avg(ns)`, `P50(ns)`, `P90(ns)`, and `P99(ns)` measure
  allocation performance.
- `Frag_avg`, `Frag_p50`, `Frag_p90`, and `Frag_p99` summarize sampled
  fragmentation ratios.
- `LargestFreeMB` shows the final largest contiguous free region.
- `Evictions` counts fail-triggered eviction rounds during measurement.
- `Full/Partial/Fail/Total` reports allocation outcomes. Only results with
  `result->size() == replica_num` count as full success; shorter replica
  results are counted as partial allocations.

Fragmentation is computed per `OffsetBufferAllocator` and then averaged by free
space:

```text
1 - largest_free_region / total_free_space
```

The weighted average avoids treating free space in different Store segments as
one mergeable region. `LargestFreeMB` still reports the final largest contiguous
free region across all segments.

The benchmark also prints a one-line `Prefill summary`, `Fragmentation summary`,
and `Size-class breakdown` after each result row, so reviewers can read the
actual prefill utilization, fragmentation, and per-size-class latency numbers
without manually deriving them from the table.
