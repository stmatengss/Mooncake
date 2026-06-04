// Copyright (c) Mooncake Project
#include "chunk_canonical_hash.h"

#include <xxhash.h>

#include <cstring>
#include <vector>

namespace mooncake {

namespace {

constexpr uint64_t kFnv1aBasis64 = 14695981039346656037ull;
constexpr uint64_t kFnv1aPrime64 = 1099511628211ull;
constexpr uint32_t kFnv1aBasis32 = 2166136261u;
constexpr uint32_t kFnv1aPrime32 = 16777619u;

uint64_t Fnv1a64(const void* data, size_t len) {
    uint64_t h = kFnv1aBasis64;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= kFnv1aPrime64;
    }
    return h;
}

uint32_t Fnv1a32(const void* data, size_t len) {
    uint32_t h = kFnv1aBasis32;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= kFnv1aPrime32;
    }
    return h;
}

template <class T>
void AppendLE(std::vector<uint8_t>& buf, T value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const size_t off = buf.size();
    buf.resize(off + sizeof(T));
    std::memcpy(buf.data() + off, &value, sizeof(T));
}

}  // namespace

uint64_t ComputeChunkContentHash(const ChunkHashInputs& inputs) {
    // Canonical byte stream — field order is part of the wire contract.
    std::vector<uint8_t> buf;
    buf.reserve(40 + inputs.token_ids.size() * 4);

    AppendLE<uint64_t>(buf, inputs.model_id);
    AppendLE<uint16_t>(buf, inputs.tokenizer_version);
    AppendLE<uint16_t>(buf, static_cast<uint16_t>(inputs.kv_dtype));
    AppendLE<uint16_t>(buf, static_cast<uint16_t>(inputs.layout));
    AppendLE<uint8_t>(buf, inputs.stored_pre_rope ? 1 : 0);
    AppendLE<uint8_t>(buf, 0);  // pad to align next field deterministically
    AppendLE<uint32_t>(buf, inputs.rope_theta_id);
    AppendLE<uint32_t>(buf, static_cast<uint32_t>(inputs.token_ids.size()));
    for (uint32_t tid : inputs.token_ids) {
        AppendLE<uint32_t>(buf, tid);
    }

    return XXH64(buf.data(), buf.size(), /*seed=*/0);
}

uint32_t ComputeRopeThetaId(double theta_base, uint32_t head_dim,
                            uint32_t max_position_embeddings) {
    uint8_t buf[sizeof(double) + 2 * sizeof(uint32_t)];
    std::memcpy(buf, &theta_base, sizeof(double));
    std::memcpy(buf + sizeof(double), &head_dim, sizeof(uint32_t));
    std::memcpy(buf + sizeof(double) + sizeof(uint32_t),
                &max_position_embeddings, sizeof(uint32_t));
    return Fnv1a32(buf, sizeof(buf));
}

uint64_t ComputeModelId(std::string_view model_name, std::string_view revision) {
    std::vector<uint8_t> buf;
    buf.reserve(model_name.size() + 1 + revision.size());
    for (char c : model_name) buf.push_back(static_cast<uint8_t>(c));
    buf.push_back('@');
    for (char c : revision) buf.push_back(static_cast<uint8_t>(c));
    uint64_t h = Fnv1a64(buf.data(), buf.size());
    return h == 0 ? 1 : h;  // never collide with the "unset" sentinel
}

}  // namespace mooncake
