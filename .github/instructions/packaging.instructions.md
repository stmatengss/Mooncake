---
applyTo: "mooncake-wheel/**,scripts/build_wheel.sh,setup.py,pyproject.toml"
description: "Packaging instructions for wheel assembly, binary bundling, Python install validation, and release-facing packaging changes."
---

# Packaging Module Instructions

- Prefer existing packaging entry points, especially `scripts/build_wheel.sh`, over ad hoc wheel assembly commands.
- Treat shared objects, CLI binaries, and Python package contents as one packaging surface.
- Packaging changes should be checked against `.github/workflows/ci.yml` so local steps stay aligned with CI.
- If a packaging issue originates in source or bindings, fix the source of truth rather than editing generated outputs.