# Mooncake 统一 Artifact I/O 设计（KV Cache / RL / Model Weights）

## 1. 目标

将三类场景统一到一套读写接口，参数尽可能一致：
- KV cache 持久化/加载
- RL checkpoint 持久化/加载
- 模型权重持久化/加载

核心原则：
- 统一入口：`save_artifact` / `load_artifact`
- 统一路径参数：`save_path` / `load_path`
- 统一格式参数：`format`（默认 `"auto"`，优先 `safetensors`）
- 统一远端参数：`storage_options`
- 统一分片参数：`tp_rank` / `tp_size`

## 2. 统一 API 签名（Python）

```python
from typing import Any, Dict, Literal, Optional

ArtifactKind = Literal["kv_cache", "rl_checkpoint", "model_weights"]

class MooncakeArtifactIO:
    def save_artifact(
        self,
        kind: ArtifactKind,
        name: str,
        obj: Any,
        save_path: str,
        *,
        format: str = "auto",
        storage_options: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tp_rank: Optional[int] = None,
        tp_size: int = 1,
        use_p2p: str = "auto",
    ) -> Dict[str, Any]:
        ...

    def load_artifact(
        self,
        kind: ArtifactKind,
        name: str,
        load_path: str,
        *,
        format: str = "auto",
        storage_options: Optional[Dict[str, Any]] = None,
        map_location: str = "cpu",
        tp_rank: Optional[int] = None,
        tp_size: int = 1,
        use_p2p: str = "auto",
    ) -> Any:
        ...
```

## 3. `save_path` 统一语义

### 协议支持
- 本地目录：`/data/ckpt/step_100/`
- 本地文件：`/data/ckpt/model.safetensors` 或 `file:///data/ckpt/model.safetensors`
- S3：`s3://bucket/prefix/...`
- Hugging Face Hub：`hf://org_or_user/repo_name/path/...`

### 规则
- `format="auto"` 时：
  - 若 `save_path` 以 `.safetensors` 结尾，使用 `safetensors`
  - 否则默认 `safetensors`
- 多分片（`tp_size > 1`）时建议目录路径，内部写入：
  - `manifest.json`
  - `rank_00000.safetensors`、`rank_00001.safetensors` ...
- 单文件 `.safetensors` 仅适用于不分片或单 rank 输出。

## 4. 后端适配（Path Resolver）

建议使用统一解析器：
- `s3://`、`file://`、本地路径：走 `fsspec`
- `hf://`：走 `huggingface_hub`（上传/下载文件或目录）

建议约定：
- `storage_options` 对不同后端透传：
  - S3：`{"key": "...", "secret": "...", "client_kwargs": {...}}`
  - HF：`{"token": "...", "revision": "main", "repo_type": "model"}`

## 5. 三类场景到统一 API 的映射

### KV cache
```python
io.save_artifact(
    kind="kv_cache",
    name="prompt_hash_xxx",
    obj={"k": k_tensor, "v": v_tensor},
    save_path="s3://mc-cache/prompt_hash_xxx/",
    tp_rank=rank,
    tp_size=world_size,
    metadata={"model": "qwen3-8b", "layer": 32},
)
```

### RL checkpoint
```python
io.save_artifact(
    kind="rl_checkpoint",
    name="expA_step_1000",
    obj=model.state_dict(),
    save_path="hf://my-org/rl-ckpt/expA/step_1000/",
    storage_options={"token": "${HF_TOKEN}"},
)
```

### Model weights
```python
io.save_artifact(
    kind="model_weights",
    name="qwen3-32b",
    obj=model.state_dict(),
    save_path="/mnt/weights/qwen3-32b/",
    tp_rank=rank,
    tp_size=world_size,
)
```

## 6. 兼容层（避免改动现有调用方）

保留现有方法并转调统一接口：
- `save_kv_cache(...)` -> `save_artifact(kind="kv_cache", ...)`
- `save_checkpoint(...)` -> `save_artifact(kind="rl_checkpoint" 或 "model_weights", ...)`
- `load_kv_cache(...)` / `load_checkpoint(...)` 同理

这样可在不破坏历史代码的前提下，逐步收敛 API。

## 7. 推荐最小实现顺序

1. 先落地统一签名和路径解析器（本地 + S3 + `.safetensors`）
2. 再增加 `hf://` 适配器（`huggingface_hub`）
3. 最后把三类现有 API 做兼容封装并补端到端测试

## 8. 测试建议

- 参数一致性测试：三类场景都使用同一组参数名
- 路径协议测试：本地、`file://`、`s3://`、`hf://`
- 格式测试：`format=auto` 与 `.safetensors` 后缀自动识别
- 分片测试：`tp_size=1` 与 `tp_size>1` 的文件布局和加载一致性
