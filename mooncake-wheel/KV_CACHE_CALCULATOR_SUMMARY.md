# KV Cache Calculator Module - Implementation Summary

## Overview
Successfully added a comprehensive KV cache calculator module to the mooncake-wheel package, inspired by the kvcache.ai blog's KV cache calculator.

## Files Added

### 1. Core Module
- **`mooncake/kvcache_calculator.py`** (378 lines)
  - Main calculator implementation
  - Support for MHA, GQA, and MLA attention mechanisms
  - Support for multiple data types (FP32, FP16, BF16, FP8, FP4, INT8)
  - Predefined configurations for common models (Llama-7B/13B/70B, Qwen-7B, DeepSeek-V3)
  - Both Python API and CLI interface

### 2. Tests
- **`tests/test_kvcache_calculator.py`** (313 lines)
  - 17 comprehensive unit tests
  - Tests for basic calculations, attention mechanisms, data types, batch scaling
  - Tests for predefined models and custom configurations
  - All tests passing ✓

### 3. Documentation
- **`mooncake/kvcache_calculator_README.md`** (327 lines)
  - Complete user guide
  - Usage examples for CLI and Python API
  - Formula explanations
  - Supported data types and models

### 4. Examples
- **`examples/kvcache_calculator_examples.py`** (244 lines)
  - 8 comprehensive examples demonstrating various use cases
  - Comparisons of attention mechanisms, data types, batch sizes, context lengths
  - Custom model configurations

### 5. Configuration
- **`pyproject.toml`** (updated)
  - Added CLI entry point: `kvcache_calculator`

## Key Features

### 1. Flexible Calculation API
```python
result = KVCacheCalculator.calculate_kv_cache_size(
    num_layers=32, num_heads=32, head_dim=128,
    seq_len=2048, batch_size=1, dtype=DataType.FP16
)
```

### 2. Predefined Models
- Llama-7B, Llama-13B, Llama-70B (with GQA)
- Qwen-7B
- DeepSeek-V3 (with MLA)

### 3. CLI Interface
```bash
# Using predefined models
kvcache_calculator --model llama-7b --seq-len 2048

# Custom parameters
kvcache_calculator --num-layers 32 --num-heads 32 --head-dim 128 \
  --seq-len 2048 --dtype fp16
```

### 4. Attention Mechanism Support
- **MHA** (Multi-Head Attention): All heads used for KV cache
- **GQA** (Grouped Query Attention): Reduced KV heads (e.g., 8 instead of 64)
- **MLA** (Multi-Latent Attention): Minimal KV heads (e.g., 1 head)

### 5. Data Type Support
- FP32 (4 bytes), FP16/BF16 (2 bytes), FP8/INT8 (1 byte), FP4 (0.5 bytes)

## Formula

```
KV Cache Size = num_layers × batch_size × seq_len × num_kv_heads × head_dim × dtype_bytes × 2
```

The factor of 2 accounts for both key and value tensors.

## Test Results

All 17 tests pass successfully:
- Data type tests
- Basic calculations
- Batch size scaling
- GQA and MLA attention mechanisms
- Different data types (FP32, FP16, FP8)
- Predefined model configurations
- Custom model configurations
- Unit conversions
- Edge cases

## Example Output

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

## Usage Examples

### Example 1: Compare Attention Mechanisms
```python
# MHA vs GQA - shows 75% memory savings with GQA
mha_result = KVCacheCalculator.calculate_kv_cache_size(...)
gqa_result = KVCacheCalculator.calculate_kv_cache_size(..., num_kv_heads=8, ...)
```

### Example 2: Compare Data Types
```bash
# FP16: 4.0 GB, FP8: 2.0 GB (50% savings)
kvcache_calculator --model llama-7b --seq-len 8192 --dtype fp16
kvcache_calculator --model llama-7b --seq-len 8192 --dtype fp8
```

### Example 3: Batch Size Impact
Shows linear scaling: batch_size=1 → 2GB, batch_size=8 → 16GB

## Integration

The module is fully integrated into the mooncake-wheel package:
- Importable as `from mooncake.kvcache_calculator import KVCacheCalculator`
- Available as CLI command `kvcache_calculator` after installation
- No external dependencies beyond Python standard library

## Benefits

1. **Memory Planning**: Accurately estimate GPU memory needs for LLM inference
2. **Architecture Comparison**: Compare different attention mechanisms and data types
3. **Cost Optimization**: Help optimize deployment costs by understanding memory requirements
4. **Educational**: Learn how KV cache scales with model size, context length, and batch size

## References

- Inspired by kvcache.ai blog KV cache calculator
- Supports modern LLM architectures (Llama, Qwen, DeepSeek)
- Implements industry-standard formulas for KV cache calculation
