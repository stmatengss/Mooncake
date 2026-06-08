// Copyright (c) Mooncake Project
#include "chunk_metadata_table.h"

namespace mooncake {

tl::expected<ChunkMetadata, ErrorCode> ChunkMetadataTable::Lookup(
    uint64_t content_hash) {
    auto& shard = GetShard(content_hash);
    MutexLocker lock(&shard.mutex);

    auto it = shard.entries.find(content_hash);
    if (it == shard.entries.end()) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return tl::unexpected(ErrorCode::CHUNK_NOT_FOUND);
    }
    hits_.fetch_add(1, std::memory_order_relaxed);
    it->second.last_access = std::chrono::steady_clock::now();
    return it->second;
}

tl::expected<std::pair<ChunkMetadata, bool>, ErrorCode>
ChunkMetadataTable::InsertOrIncRef(uint64_t content_hash,
                                   const ChunkDescriptor& desc,
                                   const std::string& store_key) {
    auto& shard = GetShard(content_hash);
    MutexLocker lock(&shard.mutex);

    auto it = shard.entries.find(content_hash);
    if (it != shard.entries.end()) {
        auto& meta = it->second;
        if (meta.status == ChunkStatus::COMPLETE) {
            meta.ref_count++;
            meta.last_access = std::chrono::steady_clock::now();
            dedup_savings_bytes_.fetch_add(meta.descriptor.kv_blob_size,
                                          std::memory_order_relaxed);
            return std::make_pair(meta, true);
        }
        // PROCESSING -- don't treat as exists (Scheme A)
        return std::make_pair(meta, false);
    }

    ChunkMetadata meta;
    meta.descriptor = desc;
    meta.store_key = store_key;
    meta.ref_count = 1;
    meta.status = ChunkStatus::PROCESSING;
    meta.last_access = std::chrono::steady_clock::now();

    auto [inserted_it, ok] =
        shard.entries.emplace(content_hash, std::move(meta));
    return std::make_pair(inserted_it->second, false);
}

tl::expected<void, ErrorCode> ChunkMetadataTable::MarkComplete(
    uint64_t content_hash) {
    auto& shard = GetShard(content_hash);
    MutexLocker lock(&shard.mutex);

    auto it = shard.entries.find(content_hash);
    if (it == shard.entries.end()) {
        return tl::unexpected(ErrorCode::CHUNK_NOT_FOUND);
    }
    it->second.status = ChunkStatus::COMPLETE;
    return {};
}

tl::expected<void, ErrorCode> ChunkMetadataTable::IncRef(
    uint64_t content_hash) {
    auto& shard = GetShard(content_hash);
    MutexLocker lock(&shard.mutex);

    auto it = shard.entries.find(content_hash);
    if (it == shard.entries.end()) {
        return tl::unexpected(ErrorCode::CHUNK_NOT_FOUND);
    }
    it->second.ref_count++;
    return {};
}

tl::expected<uint32_t, ErrorCode> ChunkMetadataTable::DecRefAndGetCount(
    uint64_t content_hash) {
    auto& shard = GetShard(content_hash);
    MutexLocker lock(&shard.mutex);

    auto it = shard.entries.find(content_hash);
    if (it == shard.entries.end()) {
        return tl::unexpected(ErrorCode::CHUNK_NOT_FOUND);
    }
    if (it->second.ref_count > 0) {
        it->second.ref_count--;
    }
    return it->second.ref_count;
}

tl::expected<void, ErrorCode> ChunkMetadataTable::Remove(
    uint64_t content_hash) {
    auto& shard = GetShard(content_hash);
    MutexLocker lock(&shard.mutex);

    auto it = shard.entries.find(content_hash);
    if (it == shard.entries.end()) {
        return tl::unexpected(ErrorCode::CHUNK_NOT_FOUND);
    }
    shard.entries.erase(it);
    return {};
}

std::vector<ChunkLookupResult> ChunkMetadataTable::BatchLookup(
    const std::vector<uint64_t>& hashes) {
    std::vector<ChunkLookupResult> results;
    results.reserve(hashes.size());

    for (auto hash : hashes) {
        ChunkLookupResult result{};
        auto& shard = GetShard(hash);
        MutexLocker lock(&shard.mutex);

        auto it = shard.entries.find(hash);
        if (it != shard.entries.end() &&
            it->second.status == ChunkStatus::COMPLETE) {
            result.exists = true;
            result.kv_blob_size = it->second.descriptor.kv_blob_size;
            result.metadata_blob_size =
                it->second.descriptor.metadata_blob_size;
            result.replica_count = 1;
        }
        results.push_back(result);
    }
    return results;
}

ChunkRegistryMetrics ChunkMetadataTable::GetMetrics() const {
    ChunkRegistryMetrics metrics;
    metrics.hits = hits_.load(std::memory_order_relaxed);
    metrics.misses = misses_.load(std::memory_order_relaxed);
    metrics.dedup_savings_bytes =
        dedup_savings_bytes_.load(std::memory_order_relaxed);

    for (uint32_t i = 0; i < kNumShards; i++) {
        const auto& shard = shards_[i];
        MutexLocker lock(&shard.mutex);
        for (const auto& [hash, meta] : shard.entries) {
            if (meta.status == ChunkStatus::COMPLETE) {
                metrics.total_chunks++;
                metrics.total_bytes += meta.descriptor.kv_blob_size;
            }
        }
    }
    return metrics;
}

}  // namespace mooncake
