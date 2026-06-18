---
applyTo: "mooncake-store/**,mooncake-wheel/tests/test_*store*.py,mooncake-wheel/tests/test_dummy_client.py,mooncake-wheel/tests/test_multi_dummy_clients.py"
description: "Store module instructions for master-client flow, lease TTL, replication, eviction, and distributed object-store tests."
---

# Store Module Instructions

- Keep master flags and client-side test assumptions aligned, especially `default_kv_lease_ttl` and metadata server address.
- Separate failures in master startup, metadata coordination, object lifecycle, replication, and eviction before changing code.
- If store-facing APIs change, check whether `mooncake-integration/` needs binding updates.
- Prefer focused store validation first, then escalate to `scripts/run_tests.sh` if behavior crosses modules.