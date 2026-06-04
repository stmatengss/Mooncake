#include "chunk_client.h"

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <vector>

#include "chunk_canonical_hash.h"
#include "chunk_registry.h"

namespace mooncake {
namespace {

ChunkHashInputs MakeInputs(std::vector<uint32_t> tokens) {
    ChunkHashInputs in;
    in.model_id = ComputeModelId("test/model", "v1");
    in.tokenizer_version = 1;
    in.kv_dtype = KVDtype::BF16;
    in.layout = KVLayout::PAGE_FIRST;
    in.stored_pre_rope = true;
    in.rope_theta_id = ComputeRopeThetaId(10000.0, 128, 8192);
    in.token_ids = std::move(tokens);
    return in;
}

ChunkDescriptor MakeDescriptorFromInputs(const ChunkHashInputs& in,
                                         uint64_t kv_size) {
    ChunkDescriptor d{};
    d.content_hash = ComputeChunkContentHash(in);
    d.model_id = in.model_id;
    d.tokenizer_version = in.tokenizer_version;
    d.layer_count = 32;
    d.head_count_kv = 1;
    d.head_dim = 128;
    d.kv_dtype = in.kv_dtype;
    d.layout = in.layout;
    d.token_count = static_cast<uint32_t>(in.token_ids.size());
    d.stored_pre_rope = in.stored_pre_rope;
    d.rope_theta_id = in.rope_theta_id;
    d.source_position = 0;
    d.hkvd_count = 0;
    d.schema_version = kChunkDescriptorSchemaVersion;
    d.kv_blob_size = static_cast<uint32_t>(kv_size);
    d.metadata_blob_size = 0;
    return d;
}

class ChunkClientTest : public ::testing::Test {
   protected:
    void SetUp() override {
        registry_ = std::make_shared<ChunkRegistry>(/*capacity=*/1 << 20);
        client_ = std::make_unique<ChunkClient>(registry_);
    }
    std::shared_ptr<ChunkRegistry> registry_;
    std::unique_ptr<ChunkClient> client_;
};

TEST_F(ChunkClientTest, PutThenGetRoundTrip) {
    auto in = MakeInputs({10, 20, 30});
    std::vector<uint8_t> kv(512, 0x5A);
    auto desc = MakeDescriptorFromInputs(in, 512);

    int64_t put_rc =
        client_->put_chunk_from(desc, kv.data(), 512, nullptr, 0);
    ASSERT_GE(put_rc, 0);

    std::vector<uint8_t> dst(512);
    ChunkDescriptor out_desc{};
    int64_t get_rc = client_->get_chunk_into(desc.content_hash, dst.data(),
                                             512, nullptr, 0, &out_desc);
    ASSERT_GE(get_rc, 0);
    EXPECT_EQ(get_rc, 512);
    EXPECT_EQ(out_desc.content_hash, desc.content_hash);
    EXPECT_EQ(out_desc.rope_theta_id, desc.rope_theta_id);
    EXPECT_EQ(std::memcmp(dst.data(), kv.data(), 512), 0);
}

TEST_F(ChunkClientTest, GetMissReturnsNegative) {
    std::vector<uint8_t> dst(64);
    int64_t rc =
        client_->get_chunk_into(0xDEAD, dst.data(), 64, nullptr, 0, nullptr);
    EXPECT_LT(rc, 0);
}

TEST_F(ChunkClientTest, GetWithUndersizedBufferReturnsNegative) {
    auto in = MakeInputs({1, 2});
    std::vector<uint8_t> kv(1024, 0x33);
    auto desc = MakeDescriptorFromInputs(in, 1024);
    client_->put_chunk_from(desc, kv.data(), 1024, nullptr, 0);

    std::vector<uint8_t> small(512);
    int64_t rc = client_->get_chunk_into(desc.content_hash, small.data(), 512,
                                         nullptr, 0, nullptr);
    EXPECT_LT(rc, 0);
}

TEST_F(ChunkClientTest, BatchLookupMixesHitsAndMisses) {
    auto in_a = MakeInputs({1, 2, 3});
    auto in_b = MakeInputs({4, 5, 6});
    std::vector<uint8_t> kv(128, 0x77);

    auto desc_a = MakeDescriptorFromInputs(in_a, 128);
    auto desc_b = MakeDescriptorFromInputs(in_b, 128);
    client_->put_chunk_from(desc_a, kv.data(), 128, nullptr, 0);
    client_->put_chunk_from(desc_b, kv.data(), 128, nullptr, 0);

    auto results = client_->lookup_chunks(
        {desc_a.content_hash, /*miss=*/12345ull, desc_b.content_hash});
    ASSERT_EQ(results.size(), 3u);
    EXPECT_TRUE(results[0].exists);
    EXPECT_FALSE(results[1].exists);
    EXPECT_TRUE(results[2].exists);
}

TEST_F(ChunkClientTest, CrossProducerDedupViaContentHash) {
    // Two independent producers compute the same canonical inputs ->
    // identical content_hash -> registry stores only one copy.
    auto in = MakeInputs({99, 100});
    std::vector<uint8_t> kv(256, 0x42);
    auto desc1 = MakeDescriptorFromInputs(in, 256);
    auto desc2 = MakeDescriptorFromInputs(in, 256);  // identical
    ASSERT_EQ(desc1.content_hash, desc2.content_hash);

    client_->put_chunk_from(desc1, kv.data(), 256, nullptr, 0);
    client_->put_chunk_from(desc2, kv.data(), 256, nullptr, 0);

    auto m = registry_->GetMetrics();
    EXPECT_EQ(m.total_chunks, 1u);
    EXPECT_EQ(m.dedup_savings_bytes, 256u);
}

}  // namespace
}  // namespace mooncake
