// Copyright (c) Mooncake Project
#include "chunk_client.h"

#include <cstring>
#include <utility>

namespace mooncake {

ChunkClient::ChunkClient(std::shared_ptr<ChunkRegistry> registry)
    : registry_(std::move(registry)) {}

ChunkClient::~ChunkClient() = default;

int64_t ChunkClient::put_chunk_from(const ChunkDescriptor& desc,
                                    const void* kv_data, size_t kv_size,
                                    const void* meta_data, size_t meta_size) {
    auto r =
        registry_->RegisterChunk(desc, kv_data, kv_size, meta_data, meta_size);
    if (!r.has_value()) {
        // ErrorCode values are already negative (see types.h §254).
        return static_cast<int64_t>(r.error());
    }
    return static_cast<int64_t>(kv_size);
}

int64_t ChunkClient::get_chunk_into(uint64_t content_hash, void* kv_data,
                                    size_t kv_size, void* meta_data,
                                    size_t meta_size,
                                    ChunkDescriptor* out_desc) {
    auto r = registry_->ResolveChunk(content_hash);
    if (!r.has_value()) {
        return static_cast<int64_t>(r.error());
    }
    if (kv_size < r->kv_size) {
        // Caller buffer too small. Release the ref we just took.
        registry_->DecRef(content_hash);
        return static_cast<int64_t>(ErrorCode::INVALID_PARAMS);
    }
    std::memcpy(kv_data, r->kv_data, r->kv_size);
    if (meta_data != nullptr && meta_size >= r->meta_size && r->meta_size > 0) {
        std::memcpy(meta_data, r->meta_data, r->meta_size);
    }
    if (out_desc != nullptr) {
        *out_desc = r->descriptor;
    }
    int64_t copied = static_cast<int64_t>(r->kv_size);
    // Release the ref taken by ResolveChunk; in Phase 1 the data is heap
    // copied so the caller no longer needs to hold it.
    registry_->DecRef(content_hash);
    return copied;
}

std::vector<ChunkLookupResult> ChunkClient::lookup_chunks(
    const std::vector<uint64_t>& hashes) {
    return registry_->LookupChunks(hashes);
}

}  // namespace mooncake
