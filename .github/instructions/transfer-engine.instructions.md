---
applyTo: "mooncake-transfer-engine/**,mooncake-wheel/tests/transfer_engine_*.py"
description: "Transfer-engine module instructions for transport, metadata, and memory-registration work. Use when editing transfer engine source or transfer-engine Python tests."
---

# Transfer Engine Module Instructions

- Treat transport selection, metadata connectivity, and memory registration as separate failure domains.
- Prefer local TCP validation with `MC_FORCE_TCP=true` unless the task is explicitly about RDMA.
- Check whether a public API change also needs a matching update in `mooncake-integration/`.
- Reuse the existing transfer tests before adding new reproduction scripts.