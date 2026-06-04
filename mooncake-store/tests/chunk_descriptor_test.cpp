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

TEST(ChunkDescriptorTest, IsTriviallyCopyable) {
    // For zero-copy across boundaries.
    EXPECT_TRUE(std::is_trivially_copyable_v<ChunkDescriptor>);
    EXPECT_TRUE(std::is_standard_layout_v<ChunkDescriptor>);
}

TEST(ChunkDescriptorTest, KVLayoutEnumHasExpectedValues) {
    EXPECT_EQ(static_cast<int>(KVLayout::kPageFirst), 0);
    EXPECT_EQ(static_cast<int>(KVLayout::kLayerFirst), 1);
    EXPECT_EQ(static_cast<int>(KVLayout::kPageFirstDirect), 2);
}

TEST(ChunkDescriptorTest, KVDtypeEnumHasExpectedValues) {
    EXPECT_EQ(static_cast<int>(KVDtype::kBF16), 0);
    EXPECT_EQ(static_cast<int>(KVDtype::kFP16), 1);
    EXPECT_EQ(static_cast<int>(KVDtype::kFP8E4M3), 2);
    EXPECT_EQ(static_cast<int>(KVDtype::kFP8E5M2), 3);
}

}  // namespace
}  // namespace mooncake
