// Copyright (c) Mooncake Project
#include "chunk_service.h"

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "allocator.h"  // ReplicaType
#include "chunk_registry.h"  // ChunkLookupResult
#include "utils/scoped_vlog_timer.h"

namespace mooncake {

WrappedChunkService::WrappedChunkService(WrappedMasterService& master_service)
    : master_service_(master_service) {}

std::string WrappedChunkService::MakeStoreKey(uint64_t content_hash) {
    std::ostringstream oss;
    oss << "__chunk__/" << std::hex << content_hash;
    return oss.str();
}

tl::expected<PutChunkStartResponse, ErrorCode>
WrappedChunkService::PutChunkStart(const UUID& client_id,
                                   const ChunkDescriptorWire& desc_wire,
                                   uint64_t kv_size,
                                   const ReplicateConfig& config) {
    ScopedVLogTimer timer(1, "PutChunkStart");
    timer.LogRequest("client_id=", client_id,
                     ", content_hash=", desc_wire.content_hash,
                     ", kv_size=", kv_size);

    const uint64_t content_hash = desc_wire.content_hash;
    const std::string store_key = MakeStoreKey(content_hash);
    const ChunkDescriptor desc = desc_wire.ToDescriptor();

    auto insert_result =
        metadata_table_.InsertOrIncRef(content_hash, desc, store_key);
    if (!insert_result.has_value()) {
        LOG(ERROR) << "PutChunkStart: InsertOrIncRef failed for hash="
                   << content_hash << ": " << toString(insert_result.error());
        return tl::unexpected(insert_result.error());
    }

    const auto& [meta, dedup_hit] = insert_result.value();
    if (dedup_hit) {
        // Dedup hit: chunk is already COMPLETE with data stored.
        VLOG(1) << "PutChunkStart: dedup hit for hash=" << content_hash;
        PutChunkStartResponse response;
        response.already_exists = true;
        timer.LogResponse("already_exists=true");
        return response;
    }

    // New chunk or still PROCESSING -- allocate replicas via master service.
    auto put_result =
        master_service_.PutStart(client_id, store_key, kv_size, config);
    if (!put_result.has_value()) {
        // Allocation failed -- clean up metadata entry.
        LOG(ERROR) << "PutChunkStart: PutStart failed for hash=" << content_hash
                   << ": " << toString(put_result.error());
        metadata_table_.Remove(content_hash);
        return tl::unexpected(put_result.error());
    }

    PutChunkStartResponse response;
    response.already_exists = false;
    response.replicas = std::move(put_result.value());
    timer.LogResponse("already_exists=false, replicas=",
                      response.replicas.size());
    return response;
}

tl::expected<void, ErrorCode> WrappedChunkService::PutChunkEnd(
    const UUID& client_id, uint64_t content_hash) {
    ScopedVLogTimer timer(1, "PutChunkEnd");
    timer.LogRequest("client_id=", client_id,
                     ", content_hash=", content_hash);

    auto lookup = metadata_table_.Lookup(content_hash);
    if (!lookup.has_value()) {
        LOG(ERROR) << "PutChunkEnd: chunk not found for hash=" << content_hash;
        return tl::unexpected(lookup.error());
    }

    const auto& meta = lookup.value();
    if (meta.status == ChunkStatus::COMPLETE) {
        // Scheme A race: another writer already completed this chunk.
        VLOG(1) << "PutChunkEnd: chunk already COMPLETE for hash="
                << content_hash;
        timer.LogResponse("already_complete=true");
        return {};
    }

    // Finalize the object in master service (memory replicas).
    auto end_result = master_service_.PutEnd(client_id, meta.store_key,
                                             ReplicaType::MEMORY);
    if (!end_result.has_value()) {
        LOG(ERROR) << "PutChunkEnd: PutEnd failed for hash=" << content_hash
                   << ": " << toString(end_result.error());
        return tl::unexpected(end_result.error());
    }

    // Mark metadata as COMPLETE.
    auto mark_result = metadata_table_.MarkComplete(content_hash);
    if (!mark_result.has_value()) {
        LOG(WARNING) << "PutChunkEnd: MarkComplete failed for hash="
                     << content_hash << " (may have been removed concurrently)";
    }

    timer.LogResponse("success=true");
    return {};
}

tl::expected<void, ErrorCode> WrappedChunkService::PutChunkRevoke(
    const UUID& client_id, uint64_t content_hash) {
    ScopedVLogTimer timer(1, "PutChunkRevoke");
    timer.LogRequest("client_id=", client_id,
                     ", content_hash=", content_hash);

    auto lookup = metadata_table_.Lookup(content_hash);
    if (!lookup.has_value()) {
        LOG(ERROR) << "PutChunkRevoke: chunk not found for hash="
                   << content_hash;
        return tl::unexpected(lookup.error());
    }

    const std::string store_key = lookup.value().store_key;

    auto revoke_result = master_service_.PutRevoke(client_id, store_key);
    if (!revoke_result.has_value()) {
        LOG(ERROR) << "PutChunkRevoke: PutRevoke failed for hash="
                   << content_hash << ": " << toString(revoke_result.error());
        return tl::unexpected(revoke_result.error());
    }

    // Remove metadata entry.
    metadata_table_.Remove(content_hash);

    timer.LogResponse("success=true");
    return {};
}

tl::expected<ResolveChunkResponse, ErrorCode>
WrappedChunkService::ResolveChunk(uint64_t content_hash) {
    ScopedVLogTimer timer(1, "ResolveChunk");
    timer.LogRequest("content_hash=", content_hash);

    auto lookup = metadata_table_.Lookup(content_hash);
    if (!lookup.has_value()) {
        VLOG(1) << "ResolveChunk: chunk not found for hash=" << content_hash;
        return tl::unexpected(ErrorCode::CHUNK_NOT_FOUND);
    }

    const auto& meta = lookup.value();
    if (meta.status != ChunkStatus::COMPLETE) {
        VLOG(1) << "ResolveChunk: chunk not COMPLETE for hash=" << content_hash;
        return tl::unexpected(ErrorCode::CHUNK_STATUS_INVALID);
    }

    // Fetch replica locations from master service.
    auto replica_result = master_service_.GetReplicaList(meta.store_key);
    if (!replica_result.has_value()) {
        if (replica_result.error() == ErrorCode::OBJECT_NOT_FOUND) {
            // Orphaned metadata -- the underlying store object is gone.
            LOG(WARNING) << "ResolveChunk: orphaned metadata for hash="
                         << content_hash << ", removing";
            metadata_table_.Remove(content_hash);
        }
        return tl::unexpected(replica_result.error());
    }

    // Pin the chunk for the duration of the read (caller must DecRef later).
    auto inc_result = metadata_table_.IncRef(content_hash);
    if (!inc_result.has_value()) {
        LOG(WARNING) << "ResolveChunk: IncRef failed for hash=" << content_hash
                     << " (may have been removed concurrently)";
    }

    ResolveChunkResponse response;
    response.descriptor = ChunkDescriptorWire(meta.descriptor);
    response.replicas = std::move(replica_result.value().replicas);
    response.lease_ttl_ms = replica_result.value().lease_ttl_ms;

    timer.LogResponse("replicas=", response.replicas.size());
    return response;
}

tl::expected<std::vector<ChunkLookupResultWire>, ErrorCode>
WrappedChunkService::LookupChunks(const std::vector<uint64_t>& hashes) {
    ScopedVLogTimer timer(1, "LookupChunks");
    timer.LogRequest("hashes_count=", hashes.size());

    auto results = metadata_table_.BatchLookup(hashes);

    std::vector<ChunkLookupResultWire> wire_results;
    wire_results.reserve(results.size());
    for (const auto& r : results) {
        ChunkLookupResultWire w;
        w.exists = r.exists;
        w.kv_blob_size = static_cast<uint32_t>(r.kv_blob_size);
        w.metadata_blob_size = static_cast<uint32_t>(r.metadata_blob_size);
        w.replica_count = r.replica_count;
        wire_results.push_back(w);
    }

    timer.LogResponse("results=", wire_results.size());
    return wire_results;
}

tl::expected<void, ErrorCode> WrappedChunkService::IncRefChunk(
    uint64_t content_hash) {
    ScopedVLogTimer timer(1, "IncRefChunk");
    timer.LogRequest("content_hash=", content_hash);

    auto result = metadata_table_.IncRef(content_hash);
    timer.LogResponseExpected(result);
    return result;
}

tl::expected<void, ErrorCode> WrappedChunkService::DecRefChunk(
    uint64_t content_hash) {
    ScopedVLogTimer timer(1, "DecRefChunk");
    timer.LogRequest("content_hash=", content_hash);

    auto dec_result = metadata_table_.DecRefAndGetCount(content_hash);
    if (!dec_result.has_value()) {
        timer.LogResponseExpected(dec_result);
        return tl::unexpected(dec_result.error());
    }

    const uint32_t new_count = dec_result.value();
    if (new_count == 0) {
        // Refcount dropped to zero -- remove the backing store object
        // and the metadata entry.
        const std::string store_key = MakeStoreKey(content_hash);
        auto remove_result = master_service_.Remove(store_key, /*force=*/true);
        if (!remove_result.has_value()) {
            LOG(WARNING) << "DecRefChunk: Remove failed for hash="
                         << content_hash
                         << ": " << toString(remove_result.error());
        }
        metadata_table_.Remove(content_hash);
        VLOG(1) << "DecRefChunk: refcount=0, removed hash=" << content_hash;
    }

    timer.LogResponse("new_count=", new_count);
    return {};
}

tl::expected<ChunkRegistryMetricsWire, ErrorCode>
WrappedChunkService::GetChunkMetrics() {
    ScopedVLogTimer timer(1, "GetChunkMetrics");
    timer.LogRequest("action=get_metrics");

    auto metrics = metadata_table_.GetMetrics();

    ChunkRegistryMetricsWire wire;
    wire.total_chunks = metrics.total_chunks;
    wire.total_bytes = metrics.total_bytes;
    wire.hits = metrics.hits;
    wire.misses = metrics.misses;
    wire.dedup_savings_bytes = metrics.dedup_savings_bytes;

    timer.LogResponse("total_chunks=", wire.total_chunks,
                      ", total_bytes=", wire.total_bytes);
    return wire;
}

void RegisterChunkService(coro_rpc::coro_rpc_server& server,
                          WrappedChunkService& chunk_service) {
    server.register_handler<&mooncake::WrappedChunkService::PutChunkStart>(
        &chunk_service);
    server.register_handler<&mooncake::WrappedChunkService::PutChunkEnd>(
        &chunk_service);
    server.register_handler<&mooncake::WrappedChunkService::PutChunkRevoke>(
        &chunk_service);
    server.register_handler<&mooncake::WrappedChunkService::ResolveChunk>(
        &chunk_service);
    server.register_handler<&mooncake::WrappedChunkService::LookupChunks>(
        &chunk_service);
    server.register_handler<&mooncake::WrappedChunkService::IncRefChunk>(
        &chunk_service);
    server.register_handler<&mooncake::WrappedChunkService::DecRefChunk>(
        &chunk_service);
    server.register_handler<&mooncake::WrappedChunkService::GetChunkMetrics>(
        &chunk_service);
}

}  // namespace mooncake
