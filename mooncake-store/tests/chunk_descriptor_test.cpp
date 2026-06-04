#include "chunk_descriptor.h"

#include <gtest/gtest.h>

#include <cstring>

namespace mooncake {
namespace {

TEST(ChunkDescriptorTest, DefaultValuesAreZero) {
    ChunkDescriptor d{};
    EXPECT_EQ(d.content_hash, 0u);
    EXPECT_EQ(d.token_count, 0u);
    EXPECT_EQ(d.layer_count, 0u);
    EXPECT_EQ(d.kv_blob_size, 0u);
    EXPECT_EQ(d.schema_version, 0u);
    EXPECT_FALSE(d.stored_pre_rope);
}

TEST(ChunkDescriptorTest, FitsInOneCacheLine) {
    // Spec §5.1: ChunkDescriptor MUST be 64 bytes to fit a cache line.
    EXPECT_EQ(sizeof(ChunkDescriptor), 64u);
}

TEST(ChunkDescriptorTest, MemcpyRoundTripPreservesEveryField) {
    ChunkDescriptor src{};
    src.content_hash = 0xDEADBEEFCAFEBABEull;
    src.model_id = 0x1234567890ABCDEFull;
    src.tokenizer_version = 7;
    src.layer_count = 32;
    src.head_count_kv = 8;
    src.head_dim = 128;
    src.kv_dtype = KVDtype::FP8_E4M3;
    src.layout = KVLayout::PAGE_FIRST_DIRECT;
    src.token_count = 256;
    src.stored_pre_rope = true;
    src.rope_theta_id = 0xABCD1234u;
    src.source_position = 4096;
    src.hkvd_count = 42;
    src.schema_version = kChunkDescriptorSchemaVersion;
    src.kv_blob_size = 1u << 20;
    src.metadata_blob_size = 84;

    uint8_t buffer[sizeof(ChunkDescriptor)];
    std::memcpy(buffer, &src, sizeof(src));
    ChunkDescriptor dst{};
    std::memcpy(&dst, buffer, sizeof(dst));

    EXPECT_EQ(dst.content_hash, src.content_hash);
    EXPECT_EQ(dst.model_id, src.model_id);
    EXPECT_EQ(dst.tokenizer_version, src.tokenizer_version);
    EXPECT_EQ(dst.layer_count, src.layer_count);
    EXPECT_EQ(dst.head_count_kv, src.head_count_kv);
    EXPECT_EQ(dst.head_dim, src.head_dim);
    EXPECT_EQ(dst.kv_dtype, src.kv_dtype);
    EXPECT_EQ(dst.layout, src.layout);
    EXPECT_EQ(dst.token_count, src.token_count);
    EXPECT_EQ(dst.stored_pre_rope, src.stored_pre_rope);
    EXPECT_EQ(dst.rope_theta_id, src.rope_theta_id);
    EXPECT_EQ(dst.source_position, src.source_position);
    EXPECT_EQ(dst.hkvd_count, src.hkvd_count);
    EXPECT_EQ(dst.schema_version, src.schema_version);
    EXPECT_EQ(dst.kv_blob_size, src.kv_blob_size);
    EXPECT_EQ(dst.metadata_blob_size, src.metadata_blob_size);
}

TEST(ChunkDescriptorTest, KVLayoutEnumHasExpectedValues) {
    EXPECT_EQ(static_cast<int>(KVLayout::PAGE_FIRST), 0);
    EXPECT_EQ(static_cast<int>(KVLayout::LAYER_FIRST), 1);
    EXPECT_EQ(static_cast<int>(KVLayout::PAGE_FIRST_DIRECT), 2);
}

TEST(ChunkDescriptorTest, KVDtypeEnumHasExpectedValues) {
    EXPECT_EQ(static_cast<int>(KVDtype::BF16), 0);
    EXPECT_EQ(static_cast<int>(KVDtype::FP16), 1);
    EXPECT_EQ(static_cast<int>(KVDtype::FP8_E4M3), 2);
    EXPECT_EQ(static_cast<int>(KVDtype::FP8_E5M2), 3);
}

}  // namespace
}  // namespace mooncake
