# SGLang upstream patches

Patches intended for [sgl-project/sglang](https://github.com/sgl-project/sglang).

## 23457 — Mooncake HiCache multi-node runtime attach `local_hostname`

**Issue:** https://github.com/sgl-project/sglang/issues/23457

**Patch:** `23457-fix-mooncake-local-hostname.patch`

Apply and open a PR against `sgl-project/sglang` `main`:

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout -b cursor/fix-mooncake-hicache-multinode-local-hostname-d2df
git am /path/to/mooncake/contrib/sglang-patches/23457-fix-mooncake-local-hostname.patch
git remote add fork git@github.com:stmatengss/sglang.git   # or your fork
git push -u fork cursor/fix-mooncake-hicache-multinode-local-hostname-d2df
gh pr create --repo sgl-project/sglang --base main \
  --head stmatengss:cursor/fix-mooncake-hicache-multinode-local-hostname-d2df \
  --title "fix(hicache): resolve Mooncake local_hostname per node for runtime attach" \
  --body "Fixes sgl-project/sglang#23457"
```
