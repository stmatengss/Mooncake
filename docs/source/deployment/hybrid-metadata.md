# Hybrid Metadata Deployment Guide

This guide explains how to deploy Mooncake Transfer Engine nodes that use **hybrid metadata discovery**: Central metadata storage (etcd, HTTP, or Redis) plus P2P handshake fallback for mixed clusters.

## When to Use Hybrid

Use hybrid mode when your cluster contains both:

- **Central-metadata nodes** — registered in etcd/HTTP/Redis under a stable hostname (e.g. `node-a`)
- **P2P-only nodes** — discovered via direct `ip:port` metadata exchange

Hybrid nodes participate in both worlds: they publish to Central **and** run a local P2P metadata RPC daemon.

| Mode | Central publish | P2P exchange | Typical use |
|------|-----------------|--------------|-------------|
| `central` | Yes | No | Static clusters with metadata server |
| `p2p` | No | Yes | Dynamic ports, no metadata server |
| `hybrid` | Yes | Yes | Mixed / rolling migration |

## Configuration

### Environment variables (Legacy Engine)

```bash
# Option A: connection string suffix (recommended)
export MC_METADATA_SERVER="http://10.0.0.1:8080/metadata+P2PHANDSHAKE"

# Option B: explicit discovery mode
export MC_METADATA_SERVER="http://10.0.0.1:8080/metadata"
export MC_METADATA_DISCOVERY_MODE=hybrid

# Use hostname as segment name (not ip:port)
export MC_LOCAL_SERVER_NAME="node-b"
```

### Python / Mooncake Store

```json
{
  "local_hostname": "node-b",
  "metadata_server": "http://10.0.0.1:8080/metadata",
  "metadata_discovery_mode": "hybrid",
  "master_server_address": "10.0.0.1:50051",
  "protocol": "rdma",
  "device_name": "mlx5_0"
}
```

`MooncakeConfig.apply_transfer_engine_env()` sets `MC_METADATA_SERVER` and `MC_METADATA_DISCOVERY_MODE` before store initialization.

### TENT Engine (`MC_USE_TENT=1`)

```json
{
  "metadata_type": "hybrid",
  "metadata_servers": "http://10.0.0.1:8080/metadata",
  "local_segment_name": "node-b",
  "rpc_server_hostname": "10.0.0.2"
}
```

Or via Legacy `TransferEngine::init()` — hybrid is inferred from `+P2PHANDSHAKE` in the connection string.

## Deployment Topology

```text
                    ┌─────────────────────┐
                    │  Metadata Server    │
                    │  (etcd / HTTP /     │
                    │   Redis)            │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
    ┌────▼────┐          ┌─────▼─────┐         ┌─────▼─────┐
    │ Central │          │  Hybrid   │         │   P2P     │
    │  node   │          │   node    │         │   node    │
    │         │          │ Central + │         │ ip:port   │
    │ hostname│          │ P2P RPC   │         │ only      │
    └─────────┘          └───────────┘         └───────────┘
```

**Lookup order on hybrid initiator:**

1. Query Central by segment name (hostname)
2. If not found and segment name is `host:port`, fall back to P2P `exchangeMetadata`

## Rolling Migration

1. Upgrade Transfer Engine to a build with hybrid support (default behavior unchanged).
2. Ensure metadata server is reachable from all nodes.
3. Convert P2P nodes to hybrid one at a time:
   ```bash
   export MC_METADATA_SERVER="etcd://10.0.0.1:2379+P2PHANDSHAKE"
   export MC_LOCAL_SERVER_NAME="$(hostname)"
   ```
4. Update initiators to open segments by **hostname** instead of `ip:port`.
5. Leave pure-central nodes unchanged.

## HTTP Metadata Server

When using HTTP as the Central backend, start the metadata server before Transfer Engine nodes:

```bash
mooncake_http_metadata_server --port 8080 --host 0.0.0.0
```

Or enable the embedded server in `mooncake_master`:

```bash
mooncake_master --enable_http_metadata_server=true \
                --http_metadata_server_port=8080
```

Hybrid nodes publish segment JSON including a `discovery` block:

```json
{
  "discovery": {
    "mode": "hybrid",
    "rpc_endpoint": "10.0.0.2:15432",
    "prefer": "central"
  }
}
```

## Verification

```bash
# Check discovery mode in logs
# Legacy: "Metadata discovery mode: hybrid"
# TENT:    "Metadata Type: hybrid"

# Query Central (HTTP example)
curl "http://10.0.0.1:8080/metadata?key=mooncake%2Fram%2Fnode-b"

# Run unit tests
./metadata_hybrid_test
./tent_hybrid_segment_registry_test   # requires USE_TENT=ON USE_HTTP=ON
```

## Troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Central lookup fails | Metadata server unreachable | Check `curl` to metadata URL; unset `http_proxy` |
| P2P fallback fails | Target RPC port closed | Verify `discovery.rpc_endpoint` in segment JSON |
| Segment name mismatch | Still using `ip:port` on hybrid node | Set `MC_LOCAL_SERVER_NAME` to hostname |
| Duplicate rpc_meta | Port changed without re-publish | Restart node or call re-publish path |

See also: [Hybrid Metadata Design](../design/transfer-engine/metadata-hybrid-design.md) and the `mooncake-troubleshoot` skill.
