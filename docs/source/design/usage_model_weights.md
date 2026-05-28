# Mooncake 大模型权重持久化/加载 API 设计与示例

## 1. 背景与目标
Mooncake 支持高性能权重分发，适用于 LLM Serving 冷启动、动态扩容等场景。

建议将模型权重与 KV cache、RL checkpoint 收敛到统一 API：`save_artifact/load_artifact`。统一后只保留一组核心参数，其中 `save_path` 同时支持本地路径、`file://`、`s3://`、`hf://`，并通过 `format="auto"` 或 `.safetensors` 后缀统一格式行为。完整方案见 `docs/source/design/usage_unified_artifact_io.md`。

## 2. 推荐 API 设计（Python）

```python
class WeightManager:
    def __init__(self, metadata_server, protocol="rdma", device_name="", use_p2p=True, **kwargs):
        ...
    def save_checkpoint(self, model_id, state_dict, tp_size=1, save_path=None, format="safetensors", storage_options=None) -> list:
        ...
    def load_checkpoint(self, model_id, tp_rank=0, tp_size=1, load_path=None, map_location="cuda", storage_options=None) -> dict:
        ...
```

## 3. 典型用法示例

### 训练端保存
```python
wm.save_checkpoint(
    model_id="qwen-72b-v1",
    state_dict=model.state_dict(),
    tp_size=4,
    save_path="oss://llm-weights/qwen-72b-v1/",
    storage_options={"key": "...", "secret": "..."}
)
```

### 推理端加载分片
```python
state_dict = wm.load_checkpoint(
    model_id="qwen-72b-v1",
    tp_rank=rank,
    tp_size=world_size,
    load_path="oss://llm-weights/qwen-72b-v1/"
)
model.load_state_dict(state_dict)
```

## 4. 技术细节与结合
- **fsspec**：统一协议，支持多云/本地。
- **SafeTensors**：推荐格式，支持 mmap/零拷贝。
- **TP Shard**：自动分片与加载。
- **P2P Store**：集群内热数据优先分发。

## 5. 测试与验证建议
- 一致性验证（TP=4 保存，TP=1/2 加载比对）。
- 性能对比（P2P 热加载 vs 远端冷加载）。
- 容错与大规模测试。

## 6. 开发者备注
- 推荐实现于 `mooncake-integration/python`，底层复用 `store_file_io.py`。
