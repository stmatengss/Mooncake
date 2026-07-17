# SGLang #23457 attachment bundle

Upstream issue: https://github.com/sgl-project/sglang/issues/23457

This directory contains the SGLang-side fix for multi-node Mooncake HiCache runtime attach
`local_hostname` resolution, packaged for submission to `sgl-project/sglang`.

## Contents

| File | Description |
|------|-------------|
| `23457-fix-mooncake-local-hostname.patch` | Git patch (`git am`) against `sgl-project/sglang` `main` |
| `attachments/mooncake_store.py` | Modified SGLang source (reference copy) |
| `attachments/mooncake_store.README.md` | Modified Mooncake HiCache README in SGLang tree |
| `attachments/test_mooncake_store_config.py` | New unit tests |
| `../submit-23457-pr.sh` | One-shot script to push fork branch and open upstream PR |

## Upstream paths

When applied to SGLang, files map to:

- `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`
- `python/sglang/srt/mem_cache/storage/mooncake_store/README.md`
- `test/registered/unit/mem_cache/test_mooncake_store_config.py`

## Quick apply

```bash
git am contrib/sglang-patches/23457/23457-fix-mooncake-local-hostname.patch
```

Or run from Mooncake repo root:

```bash
./contrib/sglang-patches/submit-23457-pr.sh
```
