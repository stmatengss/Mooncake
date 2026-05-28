# Mooncake RL 训练/Rollout/Checkpoint 持久化/加载 API 设计与示例

## 1. 推荐 API 签名（Python）

> 推荐与 KV cache、模型权重统一到同一套参数：`save_path/load_path`、`format`、`storage_options`、`tp_rank/tp_size`，统一设计见 `docs/source/design/usage_unified_artifact_io.md`。

```python
class RLStorageManager:
    def __init__(self, store):
        self.store = store

    def put_rollout(self, rollout_key, data, metadata=None) -> int:
        ...
    def get_rollout(self, rollout_key) -> dict:
        ...
    def broadcast_weights(self, model_key, state_dict, use_p2p=True) -> int:
        ...
    def save_checkpoint(self, key, state_dict, path, storage_options=None, format="safetensors") -> int:
        ...
    def load_checkpoint(self, path, target_key=None, storage_options=None, device="cpu") -> dict:
        ...
```

## 2. 典型用法代码片段

### 训练状态保存（S3）
```python
mgr.save_checkpoint(
    key="checkpoint_v100",
    state_dict=model.state_dict(),
    path="s3://my-bucket/rl_checkpoints/exp1/step_100.safetensors",
    storage_options={"key": "...", "secret": "..."}
)
```

### Rollout 数据流动
```python
mgr.put_rollout("rollout_node_5_20240526", rollout_batch)
batch_data = mgr.get_rollout("rollout_node_5_20240526")
```

## 3. 能力集成说明
- **fsspec**：统一 URL 协议，支持多云/本地。
- **P2P Store**：权重广播加速，减轻主节点压力。
- **SafeTensors**：安全高效，支持 mmap/零拷贝。

## 4. 推荐文档目录落点
- 设计文档：`docs/source/design/usage_rl_checkpoint.md`
- 示例代码：`docs/source/getting_started/examples/rl/`
- API 参考：`docs/source/python-api-reference/mooncake-store.md` 增加 RL Extensions 小节。

## 5. 测试与验证建议
- 性能对比 torch.save vs RLStorageManager.save_checkpoint。
- P2P 广播效率测试。
- 并发一致性与 fsspec 兼容性测试。
