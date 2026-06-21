# AGENTS.md

## Cursor Cloud specific instructions

Mooncake is a CMake-based C++ project with Python (pybind11) / Rust / Go components.
The product is a KVCache library: the **Transfer Engine** (data movement) and the
**Mooncake Store** (`mooncake_master` + built-in HTTP metadata server) plus the
`mooncake.store` / `mooncake.engine` Python extensions. Canonical build/run docs:
`README.md`, `scripts/run_tests.sh`, and the `mooncake-ci-local` skill.

The startup update script only runs `git submodule update --init --recursive`.
Everything below (system packages, GCC default, Go, `yalantinglibs`, the `build/`
dir, and the `test_env` venv) is baked into the VM snapshot, so a normal rebuild
is incremental — you do not need to re-run `dependencies.sh`.

### Toolchain caveats (non-obvious)
- **Use GCC, not Clang.** This image's `cc`/`c++` alternatives were pinned to
  Clang 18, which fails to link `libstdc++` (`cannot find -lstdc++`) and breaks the
  build. They are now set to GCC 13 via `update-alternatives`. If a build suddenly
  fails at the compiler-check/link step, run
  `sudo update-alternatives --set c++ /usr/bin/g++ && sudo update-alternatives --set cc /usr/bin/gcc`.
- **Go** 1.25.9 lives in `/usr/local/go`; add it with `export PATH=$PATH:/usr/local/go/bin`.
  It is only used for the optional etcd/Go HA backends (off in the default build).
- **No RDMA NIC and no CUDA** in this VM. Always use the TCP fallback:
  `MC_FORCE_TCP=true` and `MC_METADATA_SERVER=http://127.0.0.1:8080/metadata`.

### Build (incremental)
```bash
export PATH=$PATH:/usr/local/go/bin
cd build && cmake .. && cmake --build . -j4   # ~10 min from clean; OOM-prone above -j4
```
Clean builds can intermittently fail under parallelism (transient OOM in heavy
`ylt`/`coro_rpc` TUs). Just re-run `cmake --build .` — it resumes and completes.
Default build = Transfer Engine + Store + Rust bindings; CUDA/RDMA/etcd are OFF.

### Python wheel (how the CLIs get onto PATH)
```bash
source test_env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
bash scripts/build_wheel.sh
pip install mooncake-wheel/dist/*.whl
```
`scripts/run_tests.sh` deliberately requires the **pip-installed** wheel and fails
if `mooncake_master` is found in `/usr/local/bin` (i.e. do not `make install`).

### Run the Store (dev)
```bash
source test_env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
mooncake_master --default_kv_lease_ttl=500 --enable_http_metadata_server=true
# RPC on :50051, HTTP metadata server on :8080
```

### Test
- C++: from `build/`, `MC_METADATA_SERVER=http://127.0.0.1:8080/metadata DEFAULT_KV_LEASE_TTL=500 ctest --output-on-failure`.
- Python e2e: `bash scripts/run_tests.sh` (note: it `pip install`s `torch==2.11.0+cu128`,
  a large download; override with `MOONCAKE_TEST_TORCH_SPEC`/`MOONCAKE_TEST_TORCH_INDEX_URL`).
- **Lease gotcha:** the client's `DEFAULT_KV_LEASE_TTL` must match the master's
  `--default_kv_lease_ttl`, otherwise `remove()`/delete returns `-706` (object still
  leased). The standard value used by the tests is `500`.

### Lint
- `bash scripts/code_format.sh --check` (clang-format, requires `clang-format-20`, installed).
- `pre-commit run --all-files` (ruff, codespell, clang-format-20, cmake-format).
