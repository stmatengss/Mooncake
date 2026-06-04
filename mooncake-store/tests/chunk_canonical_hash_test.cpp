#include "chunk_canonical_hash.h"

#include <gtest/gtest.h>

#include <vector>

namespace mooncake {
namespace {

ChunkHashInputs MakeInputs() {
    ChunkHashInputs in;
    in.model_id = 0xDEADBEEFCAFEBABEull;
    in.tokenizer_version = 7;
    in.kv_dtype = KVDtype::BF16;
    in.layout = KVLayout::PAGE_FIRST;
    in.stored_pre_rope = true;
    in.rope_theta_id = 0x12345678u;
    in.token_ids = {1u, 2u, 3u, 4u, 5u};
    return in;
}

TEST(ChunkCanonicalHashTest, IsDeterministic) {
    auto a = MakeInputs();
    auto b = MakeInputs();
    EXPECT_EQ(ComputeChunkContentHash(a), ComputeChunkContentHash(b));
}

TEST(ChunkCanonicalHashTest, DiffersForDifferentModelId) {
    auto a = MakeInputs();
    auto b = MakeInputs();
    b.model_id = 0;
    EXPECT_NE(ComputeChunkContentHash(a), ComputeChunkContentHash(b));
}

TEST(ChunkCanonicalHashTest, DiffersForDifferentTokenIds) {
    auto a = MakeInputs();
    auto b = MakeInputs();
    b.token_ids.back() = 99u;
    EXPECT_NE(ComputeChunkContentHash(a), ComputeChunkContentHash(b));
}

TEST(ChunkCanonicalHashTest, DiffersForDifferentDtype) {
    auto a = MakeInputs();
    auto b = MakeInputs();
    b.kv_dtype = KVDtype::FP8_E4M3;
    EXPECT_NE(ComputeChunkContentHash(a), ComputeChunkContentHash(b));
}

TEST(ChunkCanonicalHashTest, DiffersForPreVsPostRope) {
    auto a = MakeInputs();
    auto b = MakeInputs();
    b.stored_pre_rope = false;
    EXPECT_NE(ComputeChunkContentHash(a), ComputeChunkContentHash(b));
}

TEST(ChunkCanonicalHashTest, EmptyTokenIdsStillHashes) {
    auto a = MakeInputs();
    a.token_ids.clear();
    auto h = ComputeChunkContentHash(a);
    EXPECT_NE(h, 0u);
}

TEST(ChunkCanonicalHashTest, ComputeRopeThetaIdIsStable) {
    auto a = ComputeRopeThetaId(10000.0, 128, 32768);
    auto b = ComputeRopeThetaId(10000.0, 128, 32768);
    auto c = ComputeRopeThetaId(50000.0, 128, 32768);
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(ChunkCanonicalHashTest, ComputeModelIdIsStable) {
    auto a = ComputeModelId("moonshotai/Kimi-K2", "rev-abc");
    auto b = ComputeModelId("moonshotai/Kimi-K2", "rev-abc");
    auto c = ComputeModelId("moonshotai/Kimi-K2", "rev-def");
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
    EXPECT_NE(a, 0u);
}

TEST(ChunkCanonicalHashTest, ContentHashIsStableFrozenValue) {
    // Golden test. If this fails, the canonical encoding changed —
    // bump kChunkDescriptorSchemaVersion AND update this value.
    auto inputs = MakeInputs();
    uint64_t h = ComputeChunkContentHash(inputs);
    EXPECT_EQ(h, /*FROZEN=*/0x8b84cd4f57f896c9ULL);
}

TEST(ChunkCanonicalHashTest, RopeThetaIdIsStableFrozenValue) {
    uint32_t id = ComputeRopeThetaId(10000.0, 128, 32768);
    EXPECT_EQ(id, /*FROZEN=*/0x392bf036u);
}

TEST(ChunkCanonicalHashTest, ModelIdIsStableFrozenValue) {
    uint64_t id = ComputeModelId("moonshotai/Kimi-K2", "rev-abc");
    EXPECT_EQ(id, /*FROZEN=*/0xfd911cbfe38d3ae9ULL);
}

}  // namespace
}  // namespace mooncake
