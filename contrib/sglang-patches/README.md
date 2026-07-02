# SGLang upstream patches

Patches intended for [sgl-project/sglang](https://github.com/sgl-project/sglang), attached here for tracking and manual upstream submission.

## Bundles

| Issue | Directory | Summary |
|-------|-----------|---------|
| [#23457](https://github.com/sgl-project/sglang/issues/23457) | [`23457/`](23457/) | Mooncake HiCache multi-node runtime attach `local_hostname` fix |

Each bundle includes:

- `.patch` file for `git am`
- `attachments/` with modified upstream source copies
- `MANIFEST.md` with file mapping and apply instructions

## Submit upstream PR

From Mooncake repo root:

```bash
./contrib/sglang-patches/submit-23457-pr.sh
```

Requires `gh` logged in with push access to your `sglang` fork.
