// Copyright (c) Mooncake Project
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ylt/util/tl/expected.hpp>

#include "chunk_descriptor.h"
#include "chunk_registry.h"  // ChunkLookupResult
#include "mutex.h"
#include "types.h"

namespace mooncake {

enum class ChunkStatus : uint8_t {
    PROCESSING = 0,
    COMPLETE = 1,
};

struct ChunkMetadata {
    ChunkDescriptor descriptor;
    std::string store_key;
    uint32_t ref_count = 0;
    ChunkStatus status = ChunkStatus::PROCESSING;
    std::chrono::steady_clock::time_point last_access;
};

/// Aggregate metrics for chunks tracked by ChunkMetadataTable.
/// Intentionally separate from RegistryMetrics (which includes evictions
/// tracked by ChunkRegistry).
struct ChunkRegistryMetrics {
    uint64_t total_chunks = 0;
    uint64_t total_bytes = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t dedup_savings_bytes = 0;
};

/// 1024-shard concurrent hash map that stores chunk metadata indexed by
/// content_hash.
///
/// Each public method acquires only the shard lock for the target hash,
/// so operations on different shards proceed in parallel.
class ChunkMetadataTable {
   public:
    static constexpr uint32_t kNumShards = 1024;

    /// Look up metadata by content_hash.  Returns CHUNK_NOT_FOUND on miss.
    tl::expected<ChunkMetadata, ErrorCode> Lookup(uint64_t content_hash);

    /// Insert new metadata or increment refcount if already COMPLETE.
    /// Returns (metadata, true) when the entry was already COMPLETE (dedup hit),
    /// (metadata, false) when a new entry was created or an existing entry is
    /// still PROCESSING.
    tl::expected<std::pair<ChunkMetadata, bool>, ErrorCode> InsertOrIncRef(
        uint64_t content_hash, const ChunkDescriptor& desc,
        const std::string& store_key);

    /// Transition a PROCESSING entry to COMPLETE.
    tl::expected<void, ErrorCode> MarkComplete(uint64_t content_hash);

    /// Increment refcount of an existing entry.
    tl::expected<void, ErrorCode> IncRef(uint64_t content_hash);

    /// Decrement refcount and return the new count.
    tl::expected<uint32_t, ErrorCode> DecRefAndGetCount(
        uint64_t content_hash);

    /// Erase an entry entirely.
    tl::expected<void, ErrorCode> Remove(uint64_t content_hash);

    /// Batch existence check.  Returns one ChunkLookupResult per input hash.
    /// Only COMPLETE entries are reported as existing.
    std::vector<ChunkLookupResult> BatchLookup(
        const std::vector<uint64_t>& hashes);

    /// Snapshot of aggregate metrics (iterates all shards).
    ChunkRegistryMetrics GetMetrics() const;

   private:
    struct Shard {
        mutable Mutex mutex;
        std::unordered_map<uint64_t, ChunkMetadata> entries;
    };
    std::array<Shard, kNumShards> shards_;

    Shard& GetShard(uint64_t hash) { return shards_[hash % kNumShards]; }
    const Shard& GetShard(uint64_t hash) const {
        return shards_[hash % kNumShards];
    }

    mutable std::atomic<uint64_t> hits_{0};
    mutable std::atomic<uint64_t> misses_{0};
    std::atomic<uint64_t> dedup_savings_bytes_{0};
};

}  // namespace mooncake
