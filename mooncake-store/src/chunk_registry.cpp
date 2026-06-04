// Copyright (c) Mooncake Project
#include "chunk_registry.h"

#include <cstring>

namespace mooncake {

ChunkRegistry::ChunkRegistry(size_t capacity_bytes)
    : capacity_bytes_(capacity_bytes) {}

ChunkRegistry::~ChunkRegistry() = default;

tl::expected<void, ErrorCode> ChunkRegistry::RegisterChunk(
    const ChunkDescriptor& desc, const void* kv_data, size_t kv_size,
    const void* meta_data, size_t meta_size) {
    if (desc.kv_blob_size != kv_size) {
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    if (desc.metadata_blob_size != meta_size) {
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    if (kv_size == 0) {
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }

    std::lock_guard<std::mutex> lock(mu_);

    auto it = entries_.find(desc.content_hash);
    if (it != entries_.end()) {
        // Idempotent: bump refcount and account dedup savings.
        ++it->second->ref_count;
        metrics_.dedup_savings_bytes += kv_size;
        TouchLocked(*it->second);
        return {};
    }

    auto entry = std::make_unique<Entry>();
    entry->descriptor = desc;
    entry->kv_data.assign(static_cast<const uint8_t*>(kv_data),
                          static_cast<const uint8_t*>(kv_data) + kv_size);
    if (meta_size > 0 && meta_data != nullptr) {
        entry->meta_data.assign(
            static_cast<const uint8_t*>(meta_data),
            static_cast<const uint8_t*>(meta_data) + meta_size);
    }
    entry->ref_count = 1;
    lru_list_.push_front(desc.content_hash);
    entry->lru_it = lru_list_.begin();

    metrics_.total_chunks += 1;
    metrics_.total_bytes += kv_size + meta_size;
    entries_.emplace(desc.content_hash, std::move(entry));

    EvictWhileOverCapacityLocked();
    return {};
}

tl::expected<ChunkRef, ErrorCode> ChunkRegistry::ResolveChunk(
    uint64_t content_hash) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = entries_.find(content_hash);
    if (it == entries_.end()) {
        ++metrics_.misses;
        return tl::unexpected(ErrorCode::OBJECT_NOT_FOUND);
    }
    ++metrics_.hits;
    ++it->second->ref_count;
    TouchLocked(*it->second);

    ChunkRef ref;
    ref.descriptor = it->second->descriptor;
    ref.kv_data = it->second->kv_data.data();
    ref.kv_size = it->second->kv_data.size();
    ref.meta_data =
        it->second->meta_data.empty() ? nullptr : it->second->meta_data.data();
    ref.meta_size = it->second->meta_data.size();
    return ref;
}

std::vector<ChunkLookupResult> ChunkRegistry::LookupChunks(
    const std::vector<uint64_t>& content_hashes) {
    std::vector<ChunkLookupResult> out;
    out.reserve(content_hashes.size());
    std::lock_guard<std::mutex> lock(mu_);
    for (uint64_t h : content_hashes) {
        auto it = entries_.find(h);
        ChunkLookupResult r{};
        if (it == entries_.end()) {
            r.exists = false;
        } else {
            r.exists = true;
            r.kv_blob_size = it->second->descriptor.kv_blob_size;
            r.metadata_blob_size = it->second->descriptor.metadata_blob_size;
            r.replica_count = 1;
        }
        out.push_back(r);
    }
    return out;
}

void ChunkRegistry::IncRef(uint64_t content_hash) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = entries_.find(content_hash);
    if (it != entries_.end()) {
        ++it->second->ref_count;
    }
}

void ChunkRegistry::DecRef(uint64_t content_hash) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = entries_.find(content_hash);
    if (it != entries_.end() && it->second->ref_count > 0) {
        --it->second->ref_count;
    }
}

int ChunkRegistry::GetRefCount(uint64_t content_hash) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = entries_.find(content_hash);
    return it == entries_.end() ? 0 : it->second->ref_count;
}

EvictionStats ChunkRegistry::RunEviction(size_t target_bytes) {
    EvictionStats stats{};
    std::lock_guard<std::mutex> lock(mu_);
    while (metrics_.total_bytes > target_bytes && !lru_list_.empty()) {
        uint64_t victim_hash = lru_list_.back();
        auto it = entries_.find(victim_hash);
        if (it == entries_.end()) {
            lru_list_.pop_back();
            continue;
        }
        if (it->second->ref_count > 0) {
            // Cannot evict pinned entry; abort eviction pass.
            break;
        }
        size_t freed =
            it->second->kv_data.size() + it->second->meta_data.size();
        lru_list_.pop_back();
        metrics_.total_chunks -= 1;
        metrics_.total_bytes -= freed;
        metrics_.evictions += 1;
        entries_.erase(it);
        stats.evicted_chunks += 1;
        stats.freed_bytes += freed;
    }
    return stats;
}

RegistryMetrics ChunkRegistry::GetMetrics() const {
    std::lock_guard<std::mutex> lock(mu_);
    return metrics_;
}

void ChunkRegistry::TouchLocked(Entry& e) {
    lru_list_.erase(e.lru_it);
    lru_list_.push_front(e.descriptor.content_hash);
    e.lru_it = lru_list_.begin();
}

void ChunkRegistry::EvictWhileOverCapacityLocked() {
    while (metrics_.total_bytes > capacity_bytes_ && !lru_list_.empty()) {
        uint64_t victim_hash = lru_list_.back();
        auto it = entries_.find(victim_hash);
        if (it == entries_.end()) {
            lru_list_.pop_back();
            continue;
        }
        if (it->second->ref_count > 0) break;
        size_t freed =
            it->second->kv_data.size() + it->second->meta_data.size();
        lru_list_.pop_back();
        metrics_.total_chunks -= 1;
        metrics_.total_bytes -= freed;
        metrics_.evictions += 1;
        entries_.erase(it);
    }
}

}  // namespace mooncake
