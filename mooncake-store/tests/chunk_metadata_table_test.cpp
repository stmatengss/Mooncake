// Copyright (c) Mooncake Project
#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "chunk_metadata_table.h"

namespace mooncake {
namespace testing {

class ChunkMetadataTableTest : public ::testing::Test {
   protected:
    ChunkMetadataTable table_;

    ChunkDescriptor MakeDescriptor(uint64_t hash,
                                   uint32_t token_count = 256,
                                   uint32_t kv_size = 1048576) {
        ChunkDescriptor d{};
        d.content_hash = hash;
        d.token_count = token_count;
        d.model_id = 0xfd911cbfe38d3ae9ULL;
        d.tokenizer_version = 1;
        d.layer_count = 32;
        d.head_count_kv = 8;
        d.head_dim = 128;
        d.kv_dtype = KVDtype::BF16;
        d.layout = KVLayout::PAGE_FIRST;
        d.stored_pre_rope = true;
        d.source_position = 0;
        d.rope_theta_id = 0x392bf036;
        d.hkvd_count = 0;
        d.kv_blob_size = kv_size;
        d.metadata_blob_size = 0;
        d.schema_version = 1;
        return d;
    }
};

TEST_F(ChunkMetadataTableTest, InsertAndLookup) {
    auto desc = MakeDescriptor(0x1234);
    std::string store_key = "chunk:0000000000001234";

    auto insert_result = table_.InsertOrIncRef(0x1234, desc, store_key);
    ASSERT_TRUE(insert_result.has_value());
    auto [meta, already_existed] = insert_result.value();
    EXPECT_FALSE(already_existed);
    EXPECT_EQ(meta.ref_count, 1u);
    EXPECT_EQ(meta.status, ChunkStatus::PROCESSING);
    EXPECT_EQ(meta.store_key, store_key);

    auto lookup = table_.Lookup(0x1234);
    ASSERT_TRUE(lookup.has_value());
    EXPECT_EQ(lookup->descriptor.content_hash, 0x1234u);
    EXPECT_EQ(lookup->ref_count, 1u);
}

TEST_F(ChunkMetadataTableTest, InsertIdempotentAfterComplete) {
    auto desc = MakeDescriptor(0xAAAA);
    std::string store_key = "chunk:000000000000aaaa";

    table_.InsertOrIncRef(0xAAAA, desc, store_key);
    table_.MarkComplete(0xAAAA);

    auto result = table_.InsertOrIncRef(0xAAAA, desc, store_key);
    ASSERT_TRUE(result.has_value());
    auto [meta, already_existed] = result.value();
    EXPECT_TRUE(already_existed);
    EXPECT_EQ(meta.ref_count, 2u);
    EXPECT_EQ(meta.status, ChunkStatus::COMPLETE);
}

TEST_F(ChunkMetadataTableTest, InsertWhileProcessingReturnsNotExisted) {
    auto desc = MakeDescriptor(0xBBBB);
    std::string store_key = "chunk:000000000000bbbb";

    table_.InsertOrIncRef(0xBBBB, desc, store_key);

    auto result = table_.InsertOrIncRef(0xBBBB, desc, store_key);
    ASSERT_TRUE(result.has_value());
    auto [meta, already_existed] = result.value();
    EXPECT_FALSE(already_existed);
}

TEST_F(ChunkMetadataTableTest, MarkComplete) {
    auto desc = MakeDescriptor(0x5555);
    table_.InsertOrIncRef(0x5555, desc, "chunk:0000000000005555");

    auto result = table_.MarkComplete(0x5555);
    ASSERT_TRUE(result.has_value());

    auto lookup = table_.Lookup(0x5555);
    ASSERT_TRUE(lookup.has_value());
    EXPECT_EQ(lookup->status, ChunkStatus::COMPLETE);
}

TEST_F(ChunkMetadataTableTest, MarkCompleteNotFound) {
    auto result = table_.MarkComplete(0xDEAD);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::CHUNK_NOT_FOUND);
}

TEST_F(ChunkMetadataTableTest, RefCounting) {
    auto desc = MakeDescriptor(0x7777);
    table_.InsertOrIncRef(0x7777, desc, "chunk:0000000000007777");
    table_.MarkComplete(0x7777);

    table_.IncRef(0x7777);
    auto lookup = table_.Lookup(0x7777);
    EXPECT_EQ(lookup->ref_count, 2u);

    auto dec_result = table_.DecRefAndGetCount(0x7777);
    ASSERT_TRUE(dec_result.has_value());
    EXPECT_EQ(dec_result.value(), 1u);

    dec_result = table_.DecRefAndGetCount(0x7777);
    ASSERT_TRUE(dec_result.has_value());
    EXPECT_EQ(dec_result.value(), 0u);
}

TEST_F(ChunkMetadataTableTest, DecRefDoesNotUnderflow) {
    auto desc = MakeDescriptor(0x8888);
    table_.InsertOrIncRef(0x8888, desc, "chunk:0000000000008888");

    auto dec1 = table_.DecRefAndGetCount(0x8888);
    ASSERT_TRUE(dec1.has_value());
    EXPECT_EQ(dec1.value(), 0u);

    // Already at zero -- should stay at zero.
    auto dec2 = table_.DecRefAndGetCount(0x8888);
    ASSERT_TRUE(dec2.has_value());
    EXPECT_EQ(dec2.value(), 0u);
}

TEST_F(ChunkMetadataTableTest, Remove) {
    auto desc = MakeDescriptor(0x9999);
    table_.InsertOrIncRef(0x9999, desc, "chunk:0000000000009999");

    auto result = table_.Remove(0x9999);
    ASSERT_TRUE(result.has_value());

    auto lookup = table_.Lookup(0x9999);
    EXPECT_FALSE(lookup.has_value());
    EXPECT_EQ(lookup.error(), ErrorCode::CHUNK_NOT_FOUND);
}

TEST_F(ChunkMetadataTableTest, RemoveNotFound) {
    auto result = table_.Remove(0xBADBAD);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::CHUNK_NOT_FOUND);
}

TEST_F(ChunkMetadataTableTest, BatchLookup) {
    auto desc1 = MakeDescriptor(0x1111);
    auto desc2 = MakeDescriptor(0x2222);
    table_.InsertOrIncRef(0x1111, desc1, "chunk:0000000000001111");
    table_.MarkComplete(0x1111);
    table_.InsertOrIncRef(0x2222, desc2, "chunk:0000000000002222");
    table_.MarkComplete(0x2222);

    auto results = table_.BatchLookup({0x1111, 0x2222, 0x3333});
    ASSERT_EQ(results.size(), 3u);
    EXPECT_TRUE(results[0].exists);
    EXPECT_EQ(results[0].kv_blob_size, 1048576u);
    EXPECT_TRUE(results[1].exists);
    EXPECT_FALSE(results[2].exists);
}

TEST_F(ChunkMetadataTableTest, BatchLookupSkipsProcessing) {
    auto desc = MakeDescriptor(0xABCD);
    table_.InsertOrIncRef(0xABCD, desc, "chunk:000000000000abcd");
    // Not marked complete -- should not show as existing.

    auto results = table_.BatchLookup({0xABCD});
    ASSERT_EQ(results.size(), 1u);
    EXPECT_FALSE(results[0].exists);
}

TEST_F(ChunkMetadataTableTest, GetMetrics) {
    auto desc = MakeDescriptor(0xFACE, 256, 2097152);
    table_.InsertOrIncRef(0xFACE, desc, "chunk:000000000000face");
    table_.MarkComplete(0xFACE);

    auto metrics = table_.GetMetrics();
    EXPECT_EQ(metrics.total_chunks, 1u);
    EXPECT_EQ(metrics.total_bytes, 2097152u);
}

TEST_F(ChunkMetadataTableTest, GetMetricsCountsOnlyComplete) {
    auto desc1 = MakeDescriptor(0xA001, 256, 100);
    auto desc2 = MakeDescriptor(0xA002, 256, 200);
    table_.InsertOrIncRef(0xA001, desc1, "k1");
    table_.InsertOrIncRef(0xA002, desc2, "k2");
    table_.MarkComplete(0xA001);
    // 0xA002 is still PROCESSING.

    auto metrics = table_.GetMetrics();
    EXPECT_EQ(metrics.total_chunks, 1u);
    EXPECT_EQ(metrics.total_bytes, 100u);
}

TEST_F(ChunkMetadataTableTest, MetricsTrackHitsAndMisses) {
    auto desc = MakeDescriptor(0xF00D);
    table_.InsertOrIncRef(0xF00D, desc, "chunk:000000000000f00d");

    table_.Lookup(0xF00D);   // hit
    table_.Lookup(0xF00D);   // hit
    table_.Lookup(0xBEEF);   // miss

    auto metrics = table_.GetMetrics();
    EXPECT_EQ(metrics.hits, 2u);
    EXPECT_EQ(metrics.misses, 1u);
}

TEST_F(ChunkMetadataTableTest, ConcurrentInsertSameHash) {
    const uint64_t hash = 0xC0C0C0C0;
    auto desc = MakeDescriptor(hash);
    std::string store_key = "chunk:00000000c0c0c0c0";

    std::atomic<int> insert_count{0};
    std::atomic<int> existed_count{0};

    auto worker = [&]() {
        auto result = table_.InsertOrIncRef(hash, desc, store_key);
        if (result.has_value()) {
            insert_count++;
            if (result->second) existed_count++;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < 10; i++) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(insert_count.load(), 10);
    auto lookup = table_.Lookup(hash);
    ASSERT_TRUE(lookup.has_value());
}

TEST_F(ChunkMetadataTableTest, ConcurrentDifferentHashes) {
    // Insert many different hashes in parallel to exercise multiple shards.
    constexpr int kCount = 1000;
    std::vector<std::thread> threads;
    threads.reserve(kCount);

    for (int i = 0; i < kCount; i++) {
        threads.emplace_back([&, i]() {
            uint64_t hash = static_cast<uint64_t>(i) + 1;
            auto desc = MakeDescriptor(hash);
            auto result =
                table_.InsertOrIncRef(hash, desc, "key:" + std::to_string(i));
            EXPECT_TRUE(result.has_value());
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    // Verify all entries are present.
    for (int i = 0; i < kCount; i++) {
        uint64_t hash = static_cast<uint64_t>(i) + 1;
        auto lookup = table_.Lookup(hash);
        ASSERT_TRUE(lookup.has_value())
            << "Missing entry for hash " << hash;
    }
}

TEST_F(ChunkMetadataTableTest, LookupNotFound) {
    auto result = table_.Lookup(0xDEADBEEF);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::CHUNK_NOT_FOUND);
}

TEST_F(ChunkMetadataTableTest, IncRefNotFound) {
    auto result = table_.IncRef(0xDEADBEEF);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::CHUNK_NOT_FOUND);
}

TEST_F(ChunkMetadataTableTest, DecRefNotFound) {
    auto result = table_.DecRefAndGetCount(0xDEADBEEF);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::CHUNK_NOT_FOUND);
}

}  // namespace testing
}  // namespace mooncake
