#!/usr/bin/env bash
# Set up Soft-RoCE (rdma_rxe) for RDMA loopback CI on GitHub-hosted runners.
# Exports RDMA_DEVICE and RDMA_BIND_IP (via stdout KEY=value lines and optional
# GITHUB_ENV) for callers that drive store_kv_bench --protocol=rdma.

set -euo pipefail

log() { printf '+ %s\n' "$*" >&2; }

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $1" >&2
    exit 1
  fi
}

raise_memlock() {
  # RDMA memory registration needs locked memory.
  if command -v prlimit >/dev/null 2>&1; then
    sudo prlimit --pid="$$" --memlock=unlimited:unlimited 2>/dev/null || true
  fi
  ulimit -l unlimited 2>/dev/null || true
  log "memlock soft=$(ulimit -l 2>/dev/null || echo unknown)"
}

install_packages() {
  if ! command -v ibv_devices >/dev/null 2>&1 || ! command -v rdma >/dev/null 2>&1; then
    log "installing rdma-core / ibverbs utilities"
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
      rdma-core ibverbs-utils libibverbs1
  fi
  require_cmd ibv_devices
  require_cmd rdma
  require_cmd ip
}

load_rxe_module() {
  if lsmod 2>/dev/null | grep -qE '^rdma_rxe|^rxe'; then
    log "rdma_rxe module already loaded"
    return 0
  fi
  if ! command -v modprobe >/dev/null 2>&1; then
    echo "ERROR: modprobe not available; cannot load Soft-RoCE (rdma_rxe)" >&2
    exit 1
  fi
  log "loading rdma_rxe module"
  if ! sudo modprobe rdma_rxe 2>/dev/null; then
    # Older trees expose the module as "rxe".
    sudo modprobe rxe
  fi
}

pick_netdev() {
  local netdev
  netdev="$(ip -o route show default 2>/dev/null | awk '{print $5; exit}')"
  if [ -z "${netdev}" ]; then
    netdev="$(ip -o -4 addr show scope global | awk '{print $2; exit}')"
  fi
  if [ -z "${netdev}" ]; then
    echo "ERROR: could not find a usable netdev for Soft-RoCE" >&2
    ip -br link >&2 || true
    exit 1
  fi
  printf '%s\n' "${netdev}"
}

netdev_ipv4() {
  local netdev=$1
  ip -o -4 addr show dev "${netdev}" | awk '{print $4}' | cut -d/ -f1 | head -n1
}

existing_rxe_device() {
  ibv_devices 2>/dev/null | awk 'NR > 1 && $1 ~ /^rxe/ {print $1; exit}'
}

ensure_rxe_link() {
  local netdev=$1
  local existing
  existing="$(existing_rxe_device || true)"
  if [ -n "${existing}" ]; then
    log "reusing existing Soft-RoCE device ${existing}"
    printf '%s\n' "${existing}"
    return 0
  fi

  local link_name="rxe0"
  log "creating Soft-RoCE link ${link_name} on netdev ${netdev}"
  if ! sudo rdma link add "${link_name}" type rxe netdev "${netdev}"; then
    # Some rdma-core builds prefer rxe_<netdev>.
    link_name="rxe_${netdev}"
    sudo rdma link add "${link_name}" type rxe netdev "${netdev}"
  fi
  # ibv device name usually matches the link name (rxe0 / rxe_eth0).
  local device
  device="$(existing_rxe_device || true)"
  if [ -z "${device}" ]; then
    device="${link_name}"
  fi
  printf '%s\n' "${device}"
}

emit_exports() {
  local device=$1 bind_ip=$2
  # Machine-readable lines on stdout; human logs go to stderr via log().
  printf 'RDMA_DEVICE=%s\n' "${device}"
  printf 'RDMA_BIND_IP=%s\n' "${bind_ip}"
  if [ -n "${GITHUB_ENV:-}" ]; then
    {
      printf 'RDMA_DEVICE=%s\n' "${device}"
      printf 'RDMA_BIND_IP=%s\n' "${bind_ip}"
    } >>"${GITHUB_ENV}"
  fi
}

main() {
  install_packages
  raise_memlock
  load_rxe_module

  local netdev device bind_ip
  netdev="$(pick_netdev)"
  log "using netdev ${netdev}"
  device="$(ensure_rxe_link "${netdev}")"
  bind_ip="$(netdev_ipv4 "${netdev}")"
  if [ -z "${bind_ip}" ]; then
    echo "ERROR: no IPv4 address on ${netdev}" >&2
    exit 1
  fi

  log "ibv_devices:"
  ibv_devices || true
  log "rdma link:"
  rdma link show || true

  emit_exports "${device}" "${bind_ip}"
  log "Soft-RoCE ready: device=${device} bind_ip=${bind_ip}"
}

main "$@"
