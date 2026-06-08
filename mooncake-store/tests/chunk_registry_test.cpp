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

// ---- Eviction tests (Phase 1 task 4) ----

namespace mooncake {
namespace {

TEST(ChunkRegistryEvictionTest, EvictsOverCapacityRespectingLru) {
    ChunkRegistry reg(/*capacity_bytes=*/2048);
    auto blob_a = MakeBlob(1024, 0xAA);
    auto blob_b = MakeBlob(1024, 0xBB);
    auto blob_c = MakeBlob(1024, 0xCC);

    auto da = MakeDescriptor(1, 1024);
    auto db = MakeDescriptor(2, 1024);
    auto dc = MakeDescriptor(3, 1024);

    reg.RegisterChunk(da, blob_a.data(), 1024, nullptr, 0);
    reg.DecRef(1);  // unpin so it is evictable
    reg.RegisterChunk(db, blob_b.data(), 1024, nullptr, 0);
    reg.DecRef(2);

    // Now register C — total would be 3072 > 2048; oldest (a) must evict.
    reg.RegisterChunk(dc, blob_c.data(), 1024, nullptr, 0);

    EXPECT_FALSE(reg.ResolveChunk(1).has_value());  // evicted
    EXPECT_TRUE(reg.ResolveChunk(2).has_value());
    EXPECT_TRUE(reg.ResolveChunk(3).has_value());

    auto m = reg.GetMetrics();
    EXPECT_EQ(m.evictions, 1u);
}

TEST(ChunkRegistryEvictionTest, PinnedEntriesAreNotEvicted) {
    ChunkRegistry reg(/*capacity_bytes=*/2048);
    auto blob = MakeBlob(1024, 0x77);
    auto da = MakeDescriptor(10, 1024);
    auto db = MakeDescriptor(11, 1024);
    auto dc = MakeDescriptor(12, 1024);

    // A stays pinned (ref=1).
    reg.RegisterChunk(da, blob.data(), 1024, nullptr, 0);
    // B unpinned.
    reg.RegisterChunk(db, blob.data(), 1024, nullptr, 0);
    reg.DecRef(11);
    // C pushes us over capacity; oldest unpinned (B) should go.
    // But A is older AND pinned — eviction must skip it.
    reg.RegisterChunk(dc, blob.data(), 1024, nullptr, 0);

    EXPECT_TRUE(reg.ResolveChunk(10).has_value())
        << "A is pinned, must not be evicted";
    EXPECT_FALSE(reg.ResolveChunk(11).has_value())
        << "B was unpinned and oldest; must be evicted";
    EXPECT_TRUE(reg.ResolveChunk(12).has_value());
}

TEST(ChunkRegistryEvictionTest, RunEvictionRespectsTarget) {
    ChunkRegistry reg(/*capacity_bytes=*/1 << 20);
    for (int i = 0; i < 4; ++i) {
        auto blob = MakeBlob(1024, static_cast<uint8_t>(i));
        auto d = MakeDescriptor(static_cast<uint64_t>(100 + i), 1024);
        reg.RegisterChunk(d, blob.data(), 1024, nullptr, 0);
        reg.DecRef(100 + i);
    }
    EXPECT_EQ(reg.GetMetrics().total_bytes, 4096u);

    auto stats = reg.RunEviction(/*target_bytes=*/2048);
    EXPECT_GE(stats.evicted_chunks, 2u);
    EXPECT_LE(reg.GetMetrics().total_bytes, 2048u);
}

}  // namespace
}  // namespace mooncake
