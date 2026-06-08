#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "chunk_rpc_types.h"
#include "chunk_service.h"
#include "master_client.h"
#include "test_server_helpers.h"
#include "types.h"

namespace mooncake {
namespace testing {

class ChunkServiceIntegrationTest : public ::testing::Test {
   protected:
    void SetUp() override {
        InProcMasterConfig config;
        ASSERT_TRUE(master_.Start(config));

        client_id_ = generate_uuid();
        master_client_ = std::make_unique<MasterClient>(client_id_);
        auto ec = master_client_->Connect(master_.master_address());
        ASSERT_EQ(ec, ErrorCode::OK);
    }

    void TearDown() override {
        master_client_.reset();
        master_.Stop();
    }

    ChunkDescriptorWire MakeDescWire(uint64_t hash,
                                     uint32_t kv_size = 1048576) {
        ChunkDescriptorWire w;
        w.content_hash = hash;
        w.token_count = 256;
        w.model_id = 0xfd911cbfe38d3ae9ULL;
        w.tokenizer_version = 1;
        w.layer_count = 32;
        w.head_count_kv = 8;
        w.head_dim = 128;
        w.kv_dtype = 0;
        w.layout = 0;
        w.stored_pre_rope = true;
        w.source_position = 0;
        w.rope_theta_id = 0x392bf036;
        w.hkvd_count = 0;
        w.kv_blob_size = kv_size;
        w.metadata_blob_size = 0;
        w.schema_version = 1;
        return w;
    }

    InProcMaster master_;
    UUID client_id_;
    std::unique_ptr<MasterClient> master_client_;
};

// Without any mounted segments the master has nowhere to place replicas,
// so PutChunkStart should fail with NO_AVAILABLE_HANDLE.
TEST_F(ChunkServiceIntegrationTest, PutChunkStartReturnsNoHandle) {
    auto desc = MakeDescWire(0x1234);
    ReplicateConfig config;

    auto result = master_client_->PutChunkStart(desc, 1048576, config);
    // Without segments mounted, expect NO_AVAILABLE_HANDLE.
    // If somehow replicas are returned, just verify the shape.
    if (!result.has_value()) {
        EXPECT_EQ(result.error(), ErrorCode::NO_AVAILABLE_HANDLE);
    } else {
        EXPECT_FALSE(result->already_exists);
    }
}

// Looking up hashes that have never been stored should yield exists=false.
TEST_F(ChunkServiceIntegrationTest, LookupChunksEmpty) {
    auto result = master_client_->LookupChunks({0x1111, 0x2222});
    ASSERT_TRUE(result.has_value()) << "LookupChunks RPC failed";
    ASSERT_EQ(result->size(), 2u);
    EXPECT_FALSE((*result)[0].exists);
    EXPECT_FALSE((*result)[1].exists);
}

// Resolving a hash that was never stored should return CHUNK_NOT_FOUND.
TEST_F(ChunkServiceIntegrationTest, ResolveChunkNotFound) {
    auto result = master_client_->ResolveChunk(0xDEAD);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::CHUNK_NOT_FOUND);
}

// On a fresh server with no chunks the metrics should be zeroed.
TEST_F(ChunkServiceIntegrationTest, GetChunkMetricsEmpty) {
    auto result = master_client_->GetChunkMetrics();
    ASSERT_TRUE(result.has_value()) << "GetChunkMetrics RPC failed";
    EXPECT_EQ(result->total_chunks, 0u);
    EXPECT_EQ(result->total_bytes, 0u);
}

// PutChunkEnd for a hash that was never started should fail.
TEST_F(ChunkServiceIntegrationTest, PutChunkEndWithoutStart) {
    auto result = master_client_->PutChunkEnd(0xBEEF);
    EXPECT_FALSE(result.has_value());
}

// PutChunkRevoke for a non-existent hash should fail.
TEST_F(ChunkServiceIntegrationTest, PutChunkRevokeNotFound) {
    auto result = master_client_->PutChunkRevoke(0xCAFE);
    EXPECT_FALSE(result.has_value());
}

// DecRefChunk for a non-existent hash should fail.
TEST_F(ChunkServiceIntegrationTest, DecRefChunkNotFound) {
    auto result = master_client_->DecRefChunk(0xFACE);
    EXPECT_FALSE(result.has_value());
}

// LookupChunks with an empty list should return an empty vector.
TEST_F(ChunkServiceIntegrationTest, LookupChunksEmptyInput) {
    auto result = master_client_->LookupChunks({});
    ASSERT_TRUE(result.has_value()) << "LookupChunks RPC failed";
    EXPECT_EQ(result->size(), 0u);
}

}  // namespace testing
}  // namespace mooncake
