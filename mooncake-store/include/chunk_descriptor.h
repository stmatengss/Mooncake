// Copyright (c) Mooncake Project
#pragma once

#include <cstdint>
#include <type_traits>

namespace mooncake {

/// On-wire layout for the KV blob inside a chunk.
enum class KVLayout : uint16_t {
    kPageFirst = 0,
    kLayerFirst = 1,
    kPageFirstDirect = 2,
};

/// Numeric dtype of stored K and V tensors.
enum class KVDtype : uint16_t {
    kBF16 = 0,
    kFP16 = 1,
    kFP8E4M3 = 2,
    kFP8E5M2 = 3,
};

/// Self-describing header for one KV chunk stored in ChunkRegistry.
///
/// Layout is intentionally trivially-copyable + standard-layout so it can be
/// passed across process / RDMA boundaries without serialization. Total size
/// is fixed at 64 bytes (one cache line); changes that break this must bump
/// schema_version.
///
/// Note: kv_blob_size and metadata_blob_size are uint32_t (max 4 GiB per
/// blob) to keep the descriptor at one cache line. A single KV chunk is
/// bounded by token_count <= 512 and per-token KV size, so 4 GiB is far
/// beyond physical limits.
///
/// See docs/superpowers/specs/2026-06-04-mooncake-chunk-registry-design.md §5.1
struct ChunkDescriptor {
    // ---- Identity ----                                       offset / size
    uint64_t content_hash;          // xxHash64(canonical(...))    0  / 8
    uint64_t model_id;              // FNV-1a("{org}/{model}@{rev}") 8  / 8

    // ---- Model / format ----
    uint16_t tokenizer_version;     //                              16 / 2
    uint16_t layer_count;           //                              18 / 2
    uint16_t head_count_kv;         //                              20 / 2
    uint16_t head_dim;              //                              22 / 2
    KVDtype kv_dtype;               //                              24 / 2
    KVLayout layout;                //                              26 / 2
    uint32_t token_count;           // 32 <= N <= 512               28 / 4

    // ---- Position-independent fields ----
    bool stored_pre_rope;           // true = NoPE; false = post-RoPE 32 / 1
    uint8_t _pad0[3];               //                              33 / 3
    uint32_t rope_theta_id;         // FNV-1a(theta_base, dim, max_pos) 36 / 4
    uint64_t source_position;       // valid iff stored_pre_rope==false 40 / 8

    // ---- Selective-recompute metadata (optional) ----
    uint32_t hkvd_count;            // 0 = not computed             48 / 4
    uint32_t schema_version;        // 1 in Phase 1                 52 / 4

    // ---- Size validation ----
    uint32_t kv_blob_size;          // bytes (max 4 GiB)            56 / 4
    uint32_t metadata_blob_size;    // bytes (max 4 GiB)            60 / 4
};

static_assert(sizeof(ChunkDescriptor) == 64,
              "ChunkDescriptor must remain one cache line; bump schema_version "
              "if a member must be added.");
static_assert(std::is_trivially_copyable_v<ChunkDescriptor>);
static_assert(std::is_standard_layout_v<ChunkDescriptor>);

constexpr uint32_t kChunkDescriptorSchemaVersion = 1;

}  // namespace mooncake
