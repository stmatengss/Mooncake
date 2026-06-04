// Copyright (c) Mooncake Project
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <ylt/util/tl/expected.hpp>

#include "chunk_descriptor.h"
#include "types.h"  // ErrorCode

namespace mooncake {

/// View into a resolved chunk. Pointer validity is guaranteed only while the
/// caller holds the corresponding ref (acquired via Resolve or IncRef).
struct ChunkRef {
    ChunkDescriptor descriptor;
    const uint8_t* kv_data;
    size_t kv_size;
    const uint8_t* meta_data;
    size_t meta_size;
};

struct ChunkLookupResult {
    bool exists;
    uint64_t kv_blob_size;
    uint64_t metadata_blob_size;
    uint32_t replica_count;  // Always 1 in Phase 1 (single in-memory copy).
};

struct RegistryMetrics {
    uint64_t total_chunks;
    uint64_t total_bytes;
    uint64_t hits;
    uint64_t misses;
    uint64_t dedup_savings_bytes;  // Cumulative bytes saved by idempotent put.
    uint64_t evictions;
};

struct EvictionStats {
    size_t evicted_chunks;
    size_t freed_bytes;
};

/// In-process content-addressable KV chunk registry.
///
/// Phase 1 (this class):
///   - Heap-owned storage (std::vector<uint8_t>); no Replica integration.
///   - Single-process, mutex-guarded; no RPC.
///   - LRU eviction over zero-refcount entries.
///   - Idempotent RegisterChunk: re-register increases refcount, reuses storage.
///
/// Deferred:
///   - Storage_backend / Replica integration (Phase 2).
///   - coro_rpc service registration (Phase 1.5).
///   - Cross-process metric aggregation.
class ChunkRegistry {
   public:
    /// @param capacity_bytes Soft cap on total stored KV bytes. Exceeding this
    ///                        triggers LRU eviction during RegisterChunk.
    explicit ChunkRegistry(size_t capacity_bytes);
    ~ChunkRegistry();

    ChunkRegistry(const ChunkRegistry&) = delete;
    ChunkRegistry& operator=(const ChunkRegistry&) = delete;

    /// Insert chunk. Idempotent: if descriptor.content_hash already exists,
    /// refcount is incremented and the supplied blob is discarded.
    /// Returns ErrorCode::INVALID_PARAMS if descriptor.kv_blob_size != kv_size.
    tl::expected<void, ErrorCode> RegisterChunk(const ChunkDescriptor& desc,
                                                const void* kv_data,
                                                size_t kv_size,
                                                const void* meta_data,
                                                size_t meta_size);

    /// Look up by content_hash. Increments refcount on success; caller MUST
    /// pair with DecRef when done with the returned view.
    tl::expected<ChunkRef, ErrorCode> ResolveChunk(uint64_t content_hash);

    /// Batch existence query without taking refs.
    std::vector<ChunkLookupResult> LookupChunks(
        const std::vector<uint64_t>& content_hashes);

    void IncRef(uint64_t content_hash);
    void DecRef(uint64_t content_hash);

    int GetRefCount(uint64_t content_hash) const;

    /// Evict zero-refcount entries until total_bytes <= target_bytes (LRU).
    EvictionStats RunEviction(size_t target_bytes);

    RegistryMetrics GetMetrics() const;

   private:
    struct Entry {
        ChunkDescriptor descriptor;
        std::vector<uint8_t> kv_data;
        std::vector<uint8_t> meta_data;
        int ref_count;
        // Iterator into lru_list_ pointing at this entry's node, for O(1)
        // promotion/eviction.
        std::list<uint64_t>::iterator lru_it;
    };

    mutable std::mutex mu_;
    const size_t capacity_bytes_;

    std::unordered_map<uint64_t, std::unique_ptr<Entry>> entries_;
    // Front = most-recently-used. Eviction pops from back.
    std::list<uint64_t> lru_list_;

    RegistryMetrics metrics_{};

    // Helper: assumes mu_ held. Touch entry to MRU.
    void TouchLocked(Entry& e);
    // Helper: assumes mu_ held. Evict while over capacity.
    void EvictWhileOverCapacityLocked();
    // Helper: assumes mu_ held. Walks LRU from oldest toward newest,
    // evicting unpinned entries until total_bytes <= target_bytes (or no
    // more evictable entries remain). Returns stats for the pass.
    EvictionStats EvictDownToLocked(size_t target_bytes);
};

}  // namespace mooncake
