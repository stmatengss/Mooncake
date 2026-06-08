// Copyright (c) Mooncake Project
#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "chunk_descriptor.h"

namespace mooncake {

/// Parameters that uniquely identify a chunk's KV semantics.
///
/// All members are inputs to the canonical hash. Two chunks with identical
/// ChunkHashInputs are byte-equivalent KV caches that can be safely
/// substituted for one another (subject to RoPE position correction by the
/// consumer, see spec §5.3).
struct ChunkHashInputs {
    uint64_t model_id;
    uint16_t tokenizer_version;
    KVDtype kv_dtype;
    KVLayout layout;
    bool stored_pre_rope;
    uint32_t rope_theta_id;
    std::vector<uint32_t> token_ids;
};

/// Compute the content hash for a chunk.
///
/// Implementation: xxHash64 over a packed little-endian byte stream of the
/// fields in deterministic order. Field ordering is part of the on-wire
/// contract; changing it requires bumping kChunkDescriptorSchemaVersion.
uint64_t ComputeChunkContentHash(const ChunkHashInputs& inputs);

/// Stable ID for a RoPE configuration.
///
/// Different theta / dim / max_position_embeddings combos yield different
/// IDs. This protects against the Irminsul-style silent-theta bug: consumers
/// MUST validate rope_theta_id before reusing a cached chunk.
uint32_t ComputeRopeThetaId(double theta_base, uint32_t head_dim,
                            uint32_t max_position_embeddings);

/// Stable ID for a model+revision pair.
///
/// Computed via FNV-1a over "{model_name}@{revision}". Designed to be cheap
/// and deterministic across processes.
uint64_t ComputeModelId(std::string_view model_name,
                        std::string_view revision);

}  // namespace mooncake
