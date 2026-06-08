#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <ylt/util/tl/expected.hpp>

#include "chunk_metadata_table.h"
#include "chunk_rpc_types.h"
#include "rpc_service.h"
#include "types.h"

namespace mooncake {

class WrappedChunkService {
   public:
    explicit WrappedChunkService(WrappedMasterService& master_service);

    tl::expected<PutChunkStartResponse, ErrorCode>
    PutChunkStart(const UUID& client_id,
                  const ChunkDescriptorWire& desc_wire,
                  uint64_t kv_size,
                  const ReplicateConfig& config);

    tl::expected<void, ErrorCode>
    PutChunkEnd(const UUID& client_id, uint64_t content_hash);

    tl::expected<void, ErrorCode>
    PutChunkRevoke(const UUID& client_id, uint64_t content_hash);

    tl::expected<ResolveChunkResponse, ErrorCode>
    ResolveChunk(uint64_t content_hash);

    tl::expected<std::vector<ChunkLookupResultWire>, ErrorCode>
    LookupChunks(const std::vector<uint64_t>& hashes);

    tl::expected<void, ErrorCode>
    IncRefChunk(uint64_t content_hash);

    tl::expected<void, ErrorCode>
    DecRefChunk(uint64_t content_hash);

    tl::expected<ChunkRegistryMetricsWire, ErrorCode>
    GetChunkMetrics();

   private:
    WrappedMasterService& master_service_;
    ChunkMetadataTable metadata_table_;

    static std::string MakeStoreKey(uint64_t content_hash);
};

void RegisterChunkService(coro_rpc::coro_rpc_server& server,
                          WrappedChunkService& chunk_service);

}  // namespace mooncake
