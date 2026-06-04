// Copyright (c) Mooncake Project
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "chunk_descriptor.h"
#include "chunk_registry.h"

namespace mooncake {

/// Caller-side facade over ChunkRegistry.
///
/// Phase 1: thin wrapper that talks to a local in-process ChunkRegistry.
/// Provides the API surface that Phase 1.5 will expose via coro_rpc and
/// Phase 2 will plug into LMCache / SGLang HiCache.
///
/// Method names follow the spec §6.1 exactly so the future RPC binding does
/// not need to introduce new names.
class ChunkClient {
   public:
    explicit ChunkClient(std::shared_ptr<ChunkRegistry> registry);
    ~ChunkClient();

    ChunkClient(const ChunkClient&) = delete;
    ChunkClient& operator=(const ChunkClient&) = delete;

    /// Write a chunk. Returns kv_size on success, negative ErrorCode on
    /// failure. Idempotent: writing an existing content_hash bumps refcount.
    int64_t put_chunk_from(const ChunkDescriptor& desc, const void* kv_data,
                           size_t kv_size, const void* meta_data,
                           size_t meta_size);

    /// Read a chunk into caller-provided buffer.
    /// @param out_desc If non-null, populated with the stored descriptor on
    ///                 success. Caller MUST validate rope_theta_id matches
    ///                 the active inference model before using the KV.
    /// @return bytes copied (kv only) on success, negative ErrorCode on failure.
    ///         Returns ErrorCode::INVALID_PARAMS if kv_size is smaller than
    ///         the stored blob, OR if meta_data is non-null but meta_size is
    ///         smaller than the stored metadata. Pass meta_data == nullptr to
    ///         skip metadata retrieval entirely.
    int64_t get_chunk_into(uint64_t content_hash, void* kv_data, size_t kv_size,
                           void* meta_data, size_t meta_size,
                           ChunkDescriptor* out_desc);

    /// Batch existence query (cheap; no refs taken).
    std::vector<ChunkLookupResult> lookup_chunks(
        const std::vector<uint64_t>& hashes);

   private:
    std::shared_ptr<ChunkRegistry> registry_;
};

}  // namespace mooncake
