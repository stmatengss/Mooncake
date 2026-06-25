#!/usr/bin/env bash
# Submit sgl-project/sglang#23457 fix as an upstream PR via stmatengss/sglang fork.
# Requires: git, gh (logged in as a user with push access to stmatengss/sglang)
set -euo pipefail

FORK="${FORK:-stmatengss/sglang}"
BRANCH="cursor/fix-mooncake-hicache-multinode-local-hostname-d2df"
MOONCAKE_BRANCH="cursor/sglang-23457-hicache-multinode-doc-d2df"
WORKDIR="${WORKDIR:-$(mktemp -d)}"

echo "Working in: $WORKDIR"
cd "$WORKDIR"

git clone --branch "$MOONCAKE_BRANCH" --depth 1 https://github.com/stmatengss/Mooncake.git mooncake
git clone --depth 1 https://github.com/sgl-project/sglang.git sglang
cd sglang
git checkout -b "$BRANCH"
git am ../mooncake/contrib/sglang-patches/23457-fix-mooncake-local-hostname.patch

if git remote get-url fork >/dev/null 2>&1; then
  git remote set-url fork "https://github.com/${FORK}.git"
else
  git remote add fork "https://github.com/${FORK}.git"
fi

git push -u fork "$BRANCH"

gh pr create \
  --repo sgl-project/sglang \
  --base main \
  --head "${FORK%%/*}:${BRANCH}" \
  --title "fix(hicache): resolve Mooncake local_hostname per node for runtime attach" \
  --body "Fixes sgl-project/sglang#23457

When Mooncake HiCache storage is attached at runtime, extra_config is broadcast to all ranks.
\`load_from_extra_config()\` previously fell back to the fixed default \`localhost\` when
\`local_hostname\` was omitted, and a head-node \`local_hostname\` in the attach payload was
applied to every rank.

Introduce \`_resolve_local_hostname()\` so \`MOONCAKE_LOCAL_HOSTNAME\` / \`LOCAL_HOSTNAME\`
from the current process take precedence over shared extra_config."

echo "Done."
