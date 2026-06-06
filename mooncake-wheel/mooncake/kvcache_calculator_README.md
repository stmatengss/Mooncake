# KV Cache Calculator

A Python module for calculating KV (Key-Value) cache memory requirements for large language model (LLM) inference. This module is part of the Mooncake wheel package and is inspired by the KV cache calculator from kvcache.ai blog.

## Features

- **Multiple Attention Mechanisms**: Supports MHA (Multi-Head Attention), GQA (Grouped Query Attention), and MLA (Multi-Latent Attention)
- **Various Data Types**: FP32, FP16, BF16, FP8, FP4, INT8
- **Predefined Models**: Common models like Llama-7B, Llama-70B, Qwen-7B, DeepSeek-V3
- **Flexible API**: Both Python API and command-line interface
- **Comprehensive Testing**: Full unit test coverage

## Installation

The KV cache calculator is included in the `mooncake-transfer-engine` package:

```bash
pip install mooncake-transfer-engine
```

## Usage

### Command-Line Interface

After installation, you can use the `kvcache_calculator` command:

#### Using Predefined Models

```bash
# Calculate for Llama-7B with 2048 token context
kvcache_calculator --model llama-7b --seq-len 2048

# Calculate for Llama-70B with 4096 context and FP8 precision
kvcache_calculator --model llama-70b --seq-len 4096 --dtype fp8

# Calculate with batch size of 4
kvcache_calculator --model deepseek-v3 --seq-len 8192 --batch-size 4 --dtype fp8
```

Available predefined models:
- `llama-7b`: Llama 7B model
- `llama-13b`: Llama 13B model
- `llama-70b`: Llama 70B model (uses GQA)
- `qwen-7b`: Qwen 7B model
- `deepseek-v3`: DeepSeek V3 model (uses MLA)

#### Using Custom Parameters

```bash
kvcache_calculator \
  --num-layers 40 \
  --num-heads 32 \
  --head-dim 128 \
  --seq-len 2048 \
  --batch-size 1 \
  --dtype fp16
```

Or with hidden size (head_dim will be calculated):

```bash
kvcache_calculator \
  --num-layers 40 \
  --num-heads 32 \
  --hidden-size 4096 \
  --seq-len 2048 \
  --dtype fp16
```

For models with GQA (Grouped Query Attention):

```bash
kvcache_calculator \
  --num-layers 80 \
  --num-heads 64 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --seq-len 4096 \
  --dtype fp16
```

### Python API

```python
from mooncake.kvcache_calculator import KVCacheCalculator, DataType, AttentionType

# Basic calculation
result = KVCacheCalculator.calculate_kv_cache_size(
    num_layers=32,
    num_heads=32,
    head_dim=128,
    seq_len=2048,
    batch_size=1,
    dtype=DataType.FP16
)

print(f"Total KV Cache: {result['mb']:.2f} MB")
print(f"Per Token: {result['kb_per_token']:.2f} KB")

# Using GQA (Grouped Query Attention)
result_gqa = KVCacheCalculator.calculate_kv_cache_size(
    num_layers=80,
    num_heads=64,
    head_dim=128,
    seq_len=4096,
    batch_size=1,
    dtype=DataType.FP16,
    num_kv_heads=8,  # 8 KV heads instead of 64
    attention_type=AttentionType.GQA
)

# From model configuration
from mooncake.kvcache_calculator import get_model_config

model_config = get_model_config("llama-7b")
result = KVCacheCalculator.calculate_from_model_config(
    model_config=model_config,
    seq_len=2048,
    batch_size=1,
    dtype=DataType.FP16
)

# Print formatted summary
KVCacheCalculator.print_summary(
    result,
    model_name="llama-7b",
    seq_len=2048,
    batch_size=1
)
```

### Custom Model Configuration

```python
from mooncake.kvcache_calculator import KVCacheCalculator, DataType

# Define your model configuration
custom_model = {
    "num_layers": 48,
    "num_attention_heads": 40,
    "hidden_size": 5120,
    "num_key_value_heads": 40,  # MHA
}

result = KVCacheCalculator.calculate_from_model_config(
    model_config=custom_model,
    seq_len=4096,
    batch_size=2,
    dtype=DataType.BF16
)

print(f"KV Cache per token: {result['kb_per_token']:.2f} KB")
print(f"Total KV Cache: {result['gb']:.4f} GB")
```

## Output Format

The calculator returns a dictionary with the following keys:

- `bytes`: Total size in bytes
- `kb`: Total size in kilobytes
- `mb`: Total size in megabytes
- `gb`: Total size in gigabytes
- `bytes_per_token`: Size per token in bytes
- `kb_per_token`: Size per token in kilobytes

Example output:

```
============================================================
Model: llama-7b
Sequence Length: 2,048 tokens
Batch Size: 1
------------------------------------------------------------
KV Cache Memory per Token:
  524288.00 bytes (512.00 KB)

Total KV Cache Memory:
  1,073,741,824 bytes
  1,048,576.00 KB
  1,024.00 MB
  1.0000 GB
============================================================
```

## Supported Data Types

| Data Type | Bytes per Element | Description |
|-----------|-------------------|-------------|
| FP32      | 4                 | 32-bit floating point |
| FP16      | 2                 | 16-bit floating point |
| BF16      | 2                 | Brain Float 16 |
| FP8       | 1                 | 8-bit floating point |
| FP4       | 0.5               | 4-bit floating point |
| INT8      | 1                 | 8-bit integer |

## Formula

The KV cache size is calculated using the following formula:

### For MHA (Multi-Head Attention) and GQA (Grouped Query Attention):

```
KV Cache Size = num_layers × batch_size × seq_len × num_kv_heads × head_dim × dtype_bytes × 2
```

Where:
- `num_layers`: Number of transformer layers
- `batch_size`: Number of concurrent sequences
- `seq_len`: Sequence length (context window)
- `num_kv_heads`: Number of KV heads (equals `num_heads` for MHA, smaller for GQA)
- `head_dim`: Dimension per attention head
- `dtype_bytes`: Bytes per element based on data type
- `2`: Factor accounting for both key and value tensors

### Per Token:

```
Per Token = num_layers × num_kv_heads × head_dim × dtype_bytes × 2
```

## Why KV Cache Matters

As context lengths in LLMs grow beyond 200k tokens, KV cache becomes a significant memory bottleneck:

1. **Memory Constraint**: Large KV cache can restrict batch sizes
2. **Performance Impact**: Affects latency and throughput
3. **Cost Optimization**: Understanding KV cache helps optimize deployment costs
4. **Architecture Choices**: GQA and MLA reduce KV cache size while maintaining accuracy

## Examples

### Example 1: Comparing Attention Mechanisms

```python
from mooncake.kvcache_calculator import KVCacheCalculator, DataType, AttentionType

seq_len = 4096

# MHA (Multi-Head Attention)
mha_result = KVCacheCalculator.calculate_kv_cache_size(
    num_layers=32, num_heads=32, head_dim=128,
    seq_len=seq_len, dtype=DataType.FP16,
    attention_type=AttentionType.MHA
)

# GQA (Grouped Query Attention)
gqa_result = KVCacheCalculator.calculate_kv_cache_size(
    num_layers=32, num_heads=32, head_dim=128,
    seq_len=seq_len, dtype=DataType.FP16,
    num_kv_heads=8, attention_type=AttentionType.GQA
)

print(f"MHA: {mha_result['gb']:.4f} GB")
print(f"GQA: {gqa_result['gb']:.4f} GB")
print(f"Memory Saved: {(1 - gqa_result['gb'] / mha_result['gb']) * 100:.1f}%")
```

### Example 2: Comparing Data Types

```python
from mooncake.kvcache_calculator import KVCacheCalculator, DataType

model_config = get_model_config("llama-70b")

for dtype in [DataType.FP16, DataType.FP8]:
    result = KVCacheCalculator.calculate_from_model_config(
        model_config, seq_len=8192, dtype=dtype
    )
    print(f"{dtype.dtype_name}: {result['gb']:.4f} GB")
```

## Testing

Run the test suite:

```bash
python -m unittest tests/test_kvcache_calculator.py -v
```

## References

- [kvcache.ai blog](https://kvcache.ai) - Inspiration for this calculator
- [Mooncake Documentation](https://kvcache-ai.github.io/Mooncake/)
- [GQA Paper](https://arxiv.org/abs/2305.13245) - Grouped Query Attention
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) - Multi-Latent Attention

## License

This module is part of the Mooncake project and is licensed under the Apache License 2.0.
