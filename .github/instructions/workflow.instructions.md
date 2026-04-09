---
applyTo: "scripts/**,.github/workflows/**,README.md,docs/**,CONTRIBUTING.md"
description: "Workflow instructions for build scripts, CI, documentation, and team process changes such as RFC or PR-title conventions."
---

# Workflow And Automation Instructions

- Treat `.github/workflows/ci.yml` as the canonical CI contract when editing scripts or local validation flows.
- Keep local scripts and workflow assumptions in sync, especially metadata server setup, install steps, and test entry points.
- For larger architectural changes, preserve the RFC expectation documented in `CLAUDE.md`.
- Prefer documenting the real workflow over adding one-off troubleshooting steps into scripts.