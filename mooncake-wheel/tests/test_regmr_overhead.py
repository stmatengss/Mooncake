#!/usr/bin/env python3
"""Benchmark RDMA memory registration overhead via register_memory().

This benchmark is intentionally focused on the registration path only:
it does not allocate Mooncake-managed buffers and does not call
get_first_buffer_address().

Default settings match the shard layouts discussed in the benchmark note:

- 100 GiB total registration footprint
- shard sizes: 64, 32, 16, 8, 4, 2 MiB

Example:
    python mooncake-wheel/tests/test_regmr_overhead.py

Typical output:
    register memory 100GiB (1600 * 64MiB) 697.19ms
    register memory 100GiB (3200 * 32MiB) 1888.16ms
    allocate 9245.08ms, register memory 100GiB (12800 * 8MiB) 15102.90ms
"""

from __future__ import annotations

import argparse
import ctypes
import mmap
import os
import socket
import time
import unittest
from dataclasses import dataclass

from mooncake.engine import TransferEngine


DEFAULT_TOTAL_BYTES = 100 * 1024**3
DEFAULT_SHARD_SIZES_MIB = (64, 32, 16, 8, 4, 2)


@dataclass
class MemoryChunk:
    mapping: mmap.mmap
    address: int
    size: int


@dataclass
class BenchmarkResult:
    shard_size_bytes: int
    chunk_count: int
    total_bytes: int
    allocate_ms: float
    register_ms: float

    @property
    def shard_size_mib(self) -> int:
        return self.shard_size_bytes // 1024**2

    @property
    def total_gib(self) -> int:
        return self.total_bytes // 1024**3

    @property
    def avg_register_ms_per_chunk(self) -> float:
        return self.register_ms / self.chunk_count

    def render(self) -> str:
        if self.allocate_ms >= 1000:
            return (
                f"allocate {self.allocate_ms:.2f}ms, register memory "
                f"{self.total_gib}GiB ({self.chunk_count} * "
                f"{self.shard_size_mib}MiB) {self.register_ms:.2f}ms"
            )
        return (
            f"register memory {self.total_gib}GiB ({self.chunk_count} * "
            f"{self.shard_size_mib}MiB) {self.register_ms:.2f}ms"
        )


def _resolve_local_server_name() -> str:
    explicit = os.getenv("LOCAL_SERVER_NAME")
    if explicit:
        return explicit

    try:
        host_ip = socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        host_ip = "127.0.0.1"
    return f"{host_ip}:0"


def _parse_shard_sizes_mib(raw_value: str | None) -> tuple[int, ...]:
    if not raw_value:
        return DEFAULT_SHARD_SIZES_MIB

    shard_sizes = []
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        shard_size = int(part)
        if shard_size <= 0:
            raise ValueError("Shard size must be positive")
        shard_sizes.append(shard_size)

    if not shard_sizes:
        raise ValueError("At least one shard size must be provided")
    return tuple(shard_sizes)


def _allocate_chunk(size: int) -> MemoryChunk:
    mapping = mmap.mmap(-1, size, access=mmap.ACCESS_WRITE)
    address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
    return MemoryChunk(mapping=mapping, address=address, size=size)


def _cleanup_chunks(
    engine: TransferEngine,
    chunks: list[MemoryChunk],
    registered: int,
) -> None:
    for index in range(registered - 1, -1, -1):
        engine.unregister_memory(chunks[index].address)

    for chunk in reversed(chunks):
        chunk.mapping.close()


def _benchmark_one_layout(
    engine: TransferEngine,
    total_bytes: int,
    shard_size_bytes: int,
) -> BenchmarkResult:
    chunk_count = (total_bytes + shard_size_bytes - 1) // shard_size_bytes
    chunks: list[MemoryChunk] = []
    registered = 0

    allocate_start = time.perf_counter()
    for index in range(chunk_count):
        remaining = total_bytes - index * shard_size_bytes
        chunk_size = min(shard_size_bytes, remaining)
        chunks.append(_allocate_chunk(chunk_size))
    allocate_ms = (time.perf_counter() - allocate_start) * 1000

    try:
        register_start = time.perf_counter()
        for chunk in chunks:
            ret = engine.register_memory(chunk.address, chunk.size)
            if ret != 0:
                raise RuntimeError(
                    "register_memory failed for "
                    f"{chunk.size // 1024**2}MiB chunk with code {ret}"
                )
            registered += 1
        register_ms = (time.perf_counter() - register_start) * 1000
    finally:
        _cleanup_chunks(engine, chunks, registered)

    return BenchmarkResult(
        shard_size_bytes=shard_size_bytes,
        chunk_count=chunk_count,
        total_bytes=total_bytes,
        allocate_ms=allocate_ms,
        register_ms=register_ms,
    )


def run_regmr_benchmark(
    total_bytes: int,
    shard_sizes_mib: tuple[int, ...],
    metadata_server: str,
    protocol: str,
    device_name: str,
) -> list[BenchmarkResult]:
    engine = TransferEngine()
    local_server_name = _resolve_local_server_name()
    ret = engine.initialize(local_server_name, metadata_server, protocol, device_name)
    if ret != 0:
        raise RuntimeError(
            "TransferEngine initialize failed with code "
            f"{ret}, protocol={protocol}, metadata_server={metadata_server}"
        )

    results = []
    for shard_size_mib in shard_sizes_mib:
        shard_size_bytes = shard_size_mib * 1024**2
        result = _benchmark_one_layout(engine, total_bytes, shard_size_bytes)
        print(result.render())
        print(
            "  avg register per chunk: "
            f"{result.avg_register_ms_per_chunk:.4f}ms"
        )
        results.append(result)
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark RDMA memory registration overhead via register_memory()"
    )
    parser.add_argument(
        "--total-gib",
        type=int,
        default=int(os.getenv("REGMR_TOTAL_GIB", DEFAULT_TOTAL_BYTES // 1024**3)),
        help="Total registration footprint in GiB",
    )
    parser.add_argument(
        "--shard-sizes-mib",
        default=os.getenv("REGMR_SHARD_SIZES_MIB", "64,32,16,8,4,2"),
        help="Comma-separated shard sizes in MiB",
    )
    parser.add_argument(
        "--metadata-server",
        default=os.getenv("MC_METADATA_SERVER", "P2PHANDSHAKE"),
        help="Metadata server passed to TransferEngine.initialize",
    )
    parser.add_argument(
        "--protocol",
        default=os.getenv("PROTOCOL", "rdma"),
        help="Transport protocol, expected to be rdma for this benchmark",
    )
    parser.add_argument(
        "--device-name",
        default=os.getenv("DEVICE_NAME", ""),
        help="Optional RDMA device filter",
    )
    return parser


class TestRegMrOverhead(unittest.TestCase):
    def test_rdma_register_memory_benchmark(self):
        if os.getenv("RUN_REGMR_BENCHMARK") != "1":
            self.skipTest("Set RUN_REGMR_BENCHMARK=1 to run the RDMA benchmark")

        total_gib = int(os.getenv("REGMR_TOTAL_GIB", "100"))
        shard_sizes_mib = _parse_shard_sizes_mib(
            os.getenv("REGMR_SHARD_SIZES_MIB", "64,32,16,8,4,2")
        )
        results = run_regmr_benchmark(
            total_bytes=total_gib * 1024**3,
            shard_sizes_mib=shard_sizes_mib,
            metadata_server=os.getenv("MC_METADATA_SERVER", "P2PHANDSHAKE"),
            protocol=os.getenv("PROTOCOL", "rdma"),
            device_name=os.getenv("DEVICE_NAME", ""),
        )
        self.assertEqual(len(results), len(shard_sizes_mib))


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    shard_sizes_mib = _parse_shard_sizes_mib(args.shard_sizes_mib)
    total_bytes = args.total_gib * 1024**3

    print("RDMA register_memory benchmark")
    print(f"total footprint: {args.total_gib}GiB")
    print(f"shard sizes: {', '.join(str(size) for size in shard_sizes_mib)} MiB")
    print(f"metadata server: {args.metadata_server}")
    print(f"protocol: {args.protocol}")
    if args.device_name:
        print(f"device filter: {args.device_name}")
    print()

    run_regmr_benchmark(
        total_bytes=total_bytes,
        shard_sizes_mib=shard_sizes_mib,
        metadata_server=args.metadata_server,
        protocol=args.protocol,
        device_name=args.device_name,
    )


if __name__ == "__main__":
    main()
