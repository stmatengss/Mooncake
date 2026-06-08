#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <ylt/reflection/user_reflect_macro.hpp>

#include "chunk_descriptor.h"
#include "replica.h"

namespace mooncake {

struct ChunkDescriptorWire {
    uint64_t content_hash = 0;
    uint32_t token_count = 0;
    uint64_t model_id = 0;
    uint16_t tokenizer_version = 0;
    uint16_t layer_count = 0;
    uint16_t head_count_kv = 0;
    uint16_t head_dim = 0;
    uint16_t kv_dtype = 0;
    uint16_t layout = 0;
    bool stored_pre_rope = false;
    uint64_t source_position = 0;
    uint32_t rope_theta_id = 0;
    uint32_t hkvd_count = 0;
    uint32_t kv_blob_size = 0;
    uint32_t metadata_blob_size = 0;
    uint32_t schema_version = 0;

    ChunkDescriptorWire() = default;

    explicit ChunkDescriptorWire(const ChunkDescriptor& d)
        : content_hash(d.content_hash),
          token_count(d.token_count),
          model_id(d.model_id),
          tokenizer_version(d.tokenizer_version),
          layer_count(d.layer_count),
          head_count_kv(d.head_count_kv),
          head_dim(d.head_dim),
          kv_dtype(static_cast<uint16_t>(d.kv_dtype)),
          layout(static_cast<uint16_t>(d.layout)),
          stored_pre_rope(d.stored_pre_rope),
          source_position(d.source_position),
          rope_theta_id(d.rope_theta_id),
          hkvd_count(d.hkvd_count),
          kv_blob_size(d.kv_blob_size),
          metadata_blob_size(d.metadata_blob_size),
          schema_version(d.schema_version) {}

    ChunkDescriptor ToDescriptor() const {
        ChunkDescriptor d{};
        d.content_hash = content_hash;
        d.token_count = token_count;
        d.model_id = model_id;
        d.tokenizer_version = tokenizer_version;
        d.layer_count = layer_count;
        d.head_count_kv = head_count_kv;
        d.head_dim = head_dim;
        d.kv_dtype = static_cast<KVDtype>(kv_dtype);
        d.layout = static_cast<KVLayout>(layout);
        d.stored_pre_rope = stored_pre_rope;
        d.source_position = source_position;
        d.rope_theta_id = rope_theta_id;
        d.hkvd_count = hkvd_count;
        d.kv_blob_size = kv_blob_size;
        d.metadata_blob_size = metadata_blob_size;
        d.schema_version = schema_version;
        return d;
    }
};
YLT_REFL(ChunkDescriptorWire,
    content_hash, token_count, model_id, tokenizer_version,
    layer_count, head_count_kv, head_dim, kv_dtype, layout,
    stored_pre_rope, source_position, rope_theta_id,
    hkvd_count, kv_blob_size, metadata_blob_size, schema_version);

struct PutChunkStartResponse {
    bool already_exists = false;
    std::vector<Replica::Descriptor> replicas;
    uint64_t lease_ttl_ms = 0;
};
YLT_REFL(PutChunkStartResponse, already_exists, replicas, lease_ttl_ms);

struct ResolveChunkResponse {
    ChunkDescriptorWire descriptor;
    std::vector<Replica::Descriptor> replicas;
    uint64_t lease_ttl_ms = 0;
};
YLT_REFL(ResolveChunkResponse, descriptor, replicas, lease_ttl_ms);

struct ChunkLookupResultWire {
    bool exists = false;
    uint32_t kv_blob_size = 0;
    uint32_t metadata_blob_size = 0;
    uint32_t replica_count = 0;
};
YLT_REFL(ChunkLookupResultWire, exists, kv_blob_size, metadata_blob_size, replica_count);

struct ChunkRegistryMetricsWire {
    uint64_t total_chunks = 0;
    uint64_t total_bytes = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t dedup_savings_bytes = 0;
};
YLT_REFL(ChunkRegistryMetricsWire, total_chunks, total_bytes, hits, misses, dedup_savings_bytes);

}  // namespace mooncake
