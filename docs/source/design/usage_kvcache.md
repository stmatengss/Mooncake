# Mooncake KV Cache 持久化/加载场景 API 设计与示例

## 1. 推荐 API 签名（Python）

> 建议优先收敛到统一接口：`save_artifact/load_artifact`，详见 `docs/source/design/usage_unified_artifact_io.md`。当前 `save_kv_cache/load_kv_cache` 可作为兼容层保留。

```python
from typing import List, Dict, Any, Optional, Union

class MooncakeDistributedStore:
    def save_kv_cache(
        self,
        kv_cache_id: str,
        path: str,
        tensor_keys: Union[List[str], Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        use_tp: bool = True,
        format: str = "safetensors",
    ) -> bool:
        ...

    def load_kv_cache(
        self,
        path: str,
        target_keys_prefix: Optional[str] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        use_tp: bool = True,
        tp_rank: Optional[int] = None,
    ) -> Dict[str, Any]:
        ...
```

## 2. 典型用法代码片段

### 保存到远端文件系统（S3/HDFS）
```python
store.save_kv_cache(
    kv_cache_id="my_alias_A",
    path="s3://inference-cache/prefill-results/",
    tensor_keys=[f"prompt_hash_A_L{i}" for i in range(32)],
    metadata={"prompt": "Once upon a time", "hash": "0xabc123"},
    storage_options={"key": "AWS_KEY", "secret": "AWS_SECRET"},
    use_tp=True
)
```

### 从本地持久化存储加载
```python
result_meta = store.load_kv_cache(
    path="/mnt/local_ssd/offline_cache/prompt_hash_A/",
    target_keys_prefix="reloaded_session_1",
    use_tp=True
)
```

## 3. 能力结合说明
- **fsspec**：支持多种协议路径，`storage_options` 透传认证参数。
- **KV Cache Alias**：`kv_cache_id` 作为别名，简化管理。
- **TP Shard**：自动分片与加载，按 rank 处理。

## 4. 推荐文档目录落点
- 本文档建议路径：`docs/source/design/usage_kvcache.md`
- 内容建议：场景描述、API 概览、存储格式规范、性能调优。

## 5. 测试与验证建议
- 单元测试：保存/加载一致性、TP 多进程模拟、异常注入。
- 基准测试：与 torch.save 对比吞吐、超大 KV cache 加载时延。
- 元数据验证：检查 safetensors header，确保 metadata 注入。
