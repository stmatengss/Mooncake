#include "chunk_registry.h"

#include <gtest/gtest.h>

#include <vector>

namespace mooncake {
namespace {

std::vector<uint8_t> MakeBlob(size_t size, uint8_t fill) {
    return std::vector<uint8_t>(size, fill);
}

ChunkDescriptor MakeDescriptor(uint64_t hash, uint64_t kv_size) {
    ChunkDescriptor d{};
    d.content_hash = hash;
    d.model_id = 0xABCDull;
    d.tokenizer_version = 1;
    d.layer_count = 32;
    d.head_count_kv = 1;
    d.head_dim = 128;
    d.kv_dtype = KVDtype::BF16;
    d.layout = KVLayout::PAGE_FIRST;
    d.token_count = 256;
    d.stored_pre_rope = true;
    d.rope_theta_id = 0x12345678u;
    d.source_position = 0;
    d.hkvd_count = 0;
    d.schema_version = kChunkDescriptorSchemaVersion;
    d.kv_blob_size = static_cast<uint32_t>(kv_size);
    d.metadata_blob_size = 0;
    return d;
}

TEST(ChunkRegistryTest, RegisterAndResolveRoundTrip) {
    ChunkRegistry reg(/*capacity_bytes=*/1 << 20);
    auto blob = MakeBlob(1024, 0xAB);
    auto desc = MakeDescriptor(/*hash=*/42, /*kv_size=*/1024);

    auto reg_result = reg.RegisterChunk(desc, blob.data(), blob.size(),
                                        /*meta=*/nullptr, /*meta_size=*/0);
    ASSERT_TRUE(reg_result.has_value()) << "register failed";

    auto resolved = reg.ResolveChunk(42);
    ASSERT_TRUE(resolved.has_value());
    EXPECT_EQ(resolved->descriptor.content_hash, 42u);
    EXPECT_EQ(resolved->descriptor.kv_blob_size, 1024u);
    EXPECT_EQ(resolved->kv_size, 1024u);
    EXPECT_EQ(std::memcmp(resolved->kv_data, blob.data(), 1024), 0);
}

TEST(ChunkRegistryTest, ResolveMissReturnsError) {
    ChunkRegistry reg(/*capacity_bytes=*/1 << 20);
    auto resolved = reg.ResolveChunk(999);
    EXPECT_FALSE(resolved.has_value());
}

TEST(ChunkRegistryTest, IdempotentRegisterDoesNotDuplicate) {
    ChunkRegistry reg(/*capacity_bytes=*/1 << 20);
    auto blob = MakeBlob(2048, 0x11);
    auto desc = MakeDescriptor(/*hash=*/7, /*kv_size=*/2048);

    auto r1 = reg.RegisterChunk(desc, blob.data(), blob.size(), nullptr, 0);
    auto r2 = reg.RegisterChunk(desc, blob.data(), blob.size(), nullptr, 0);
    ASSERT_TRUE(r1.has_value());
    ASSERT_TRUE(r2.has_value());

    auto m = reg.GetMetrics();
    EXPECT_EQ(m.total_chunks, 1u);
    EXPECT_EQ(m.total_bytes, 2048u);
    EXPECT_EQ(m.dedup_savings_bytes, 2048u);
}

TEST(ChunkRegistryTest, RefcountIsTrackedAcrossRegisters) {
    ChunkRegistry reg(/*capacity_bytes=*/1 << 20);
    auto blob = MakeBlob(512, 0x22);
    auto desc = MakeDescriptor(/*hash=*/3, /*kv_size=*/512);

    reg.RegisterChunk(desc, blob.data(), blob.size(), nullptr, 0);
    reg.RegisterChunk(desc, blob.data(), blob.size(), nullptr, 0);
    reg.RegisterChunk(desc, blob.data(), blob.size(), nullptr, 0);

    EXPECT_EQ(reg.GetRefCount(3), 3);
    reg.DecRef(3);
    EXPECT_EQ(reg.GetRefCount(3), 2);
}

TEST(ChunkRegistryTest, RegisterWithMismatchedBlobSizeFails) {
    ChunkRegistry reg(/*capacity_bytes=*/1 << 20);
    auto blob = MakeBlob(100, 0x33);
    auto desc = MakeDescriptor(/*hash=*/8, /*kv_size=*/200);  // desc says 200

    auto r = reg.RegisterChunk(desc, blob.data(), blob.size(), nullptr, 0);
    EXPECT_FALSE(r.has_value());
}

TEST(ChunkRegistryTest, LookupReportsExistence) {
    ChunkRegistry reg(/*capacity_bytes=*/1 << 20);
    auto blob = MakeBlob(64, 0x44);
    auto desc = MakeDescriptor(/*hash=*/100, /*kv_size=*/64);
    reg.RegisterChunk(desc, blob.data(), blob.size(), nullptr, 0);

    auto results = reg.LookupChunks({100, 101, 102});
    ASSERT_EQ(results.size(), 3u);
    EXPECT_TRUE(results[0].exists);
    EXPECT_EQ(results[0].kv_blob_size, 64u);
    EXPECT_FALSE(results[1].exists);
    EXPECT_FALSE(results[2].exists);
}

}  // namespace
}  // namespace mooncake
