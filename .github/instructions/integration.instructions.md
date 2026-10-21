---
applyTo: "mooncake-integration/**,mooncake-wheel/mooncake/**"
description: "Integration-layer instructions for pybind bindings, allocators, Python surface changes, and C++ to Python API synchronization."
---

# Integration Module Instructions

- Treat `mooncake-integration/` as an interface layer that must stay in sync with C++ source modules.
- When a binding changes, verify whether the underlying C++ symbol, signature, or behavior changed too.
- Rebuild affected targets before deciding whether a Python behavior difference is a code bug.
- Escalate to packaging validation if the change affects install-time imports or wheel contents.