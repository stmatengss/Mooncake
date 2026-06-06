#!/usr/bin/env python3
"""
Example usage of the KV Cache Calculator module.

This script demonstrates various ways to use the KV cache calculator
for estimating memory requirements of LLM inference.
"""

from mooncake.kvcache_calculator import (
    KVCacheCalculator,
    DataType,
    AttentionType,
    get_model_config,
    COMMON_MODELS,
)


def example_basic_calculation():
    """Example 1: Basic KV cache calculation."""
    print("\n" + "=" * 70)
    print("Example 1: Basic KV Cache Calculation")
    print("=" * 70)

    result = KVCacheCalculator.calculate_kv_cache_size(
        num_layers=32,
        num_heads=32,
        head_dim=128,
        seq_len=2048,
        batch_size=1,
        dtype=DataType.FP16
    )

    print(f"\nFor a model with:")
    print(f"  - 32 layers")
    print(f"  - 32 attention heads")
    print(f"  - 128 head dimension")
    print(f"  - 2048 token context")
    print(f"  - FP16 precision")
    print(f"\nKV Cache Requirements:")
    print(f"  - Per token: {result['kb_per_token']:.2f} KB")
    print(f"  - Total: {result['mb']:.2f} MB ({result['gb']:.4f} GB)")


def example_predefined_models():
    """Example 2: Using predefined model configurations."""
    print("\n" + "=" * 70)
    print("Example 2: Predefined Model Configurations")
    print("=" * 70)

    seq_len = 4096
    print(f"\nComparing KV cache for {seq_len} token context:\n")

    for model_name in ["llama-7b", "llama-13b", "llama-70b", "deepseek-v3"]:
        model_config = get_model_config(model_name)
        result = KVCacheCalculator.calculate_from_model_config(
            model_config=model_config,
            seq_len=seq_len,
            batch_size=1,
            dtype=DataType.FP16
        )
        print(f"{model_name:15s}: {result['gb']:6.4f} GB  ({result['kb_per_token']:8.2f} KB/token)")


def example_attention_mechanisms():
    """Example 3: Comparing different attention mechanisms."""
    print("\n" + "=" * 70)
    print("Example 3: Attention Mechanisms Comparison")
    print("=" * 70)

    seq_len = 4096

    # MHA (Multi-Head Attention)
    mha_result = KVCacheCalculator.calculate_kv_cache_size(
        num_layers=32,
        num_heads=32,
        head_dim=128,
        seq_len=seq_len,
        batch_size=1,
        dtype=DataType.FP16,
        attention_type=AttentionType.MHA
    )

    # GQA (Grouped Query Attention) - 8 KV heads
    gqa_result = KVCacheCalculator.calculate_kv_cache_size(
        num_layers=32,
        num_heads=32,
        head_dim=128,
        seq_len=seq_len,
        batch_size=1,
        dtype=DataType.FP16,
        num_kv_heads=8,
        attention_type=AttentionType.GQA
    )

    print(f"\nFor a 32-layer model with {seq_len} tokens:")
    print(f"  MHA (32 KV heads): {mha_result['gb']:.4f} GB")
    print(f"  GQA (8 KV heads):  {gqa_result['gb']:.4f} GB")
    print(f"  Memory Saved:      {(1 - gqa_result['gb'] / mha_result['gb']) * 100:.1f}%")


def example_data_types():
    """Example 4: Comparing different data types."""
    print("\n" + "=" * 70)
    print("Example 4: Data Type Comparison")
    print("=" * 70)

    model_config = get_model_config("llama-7b")
    seq_len = 8192

    print(f"\nLlama-7B with {seq_len} token context:\n")

    for dtype in [DataType.FP32, DataType.FP16, DataType.FP8]:
        result = KVCacheCalculator.calculate_from_model_config(
            model_config=model_config,
            seq_len=seq_len,
            batch_size=1,
            dtype=dtype
        )
        print(f"  {dtype.dtype_name:6s}: {result['gb']:6.4f} GB ({result['mb']:8.2f} MB)")


def example_batch_size_impact():
    """Example 5: Impact of batch size."""
    print("\n" + "=" * 70)
    print("Example 5: Batch Size Impact")
    print("=" * 70)

    model_config = get_model_config("llama-7b")
    seq_len = 4096

    print(f"\nLlama-7B with {seq_len} token context (FP16):\n")

    for batch_size in [1, 2, 4, 8, 16]:
        result = KVCacheCalculator.calculate_from_model_config(
            model_config=model_config,
            seq_len=seq_len,
            batch_size=batch_size,
            dtype=DataType.FP16
        )
        print(f"  Batch {batch_size:2d}: {result['gb']:6.4f} GB")


def example_context_length_scaling():
    """Example 6: Context length scaling."""
    print("\n" + "=" * 70)
    print("Example 6: Context Length Scaling")
    print("=" * 70)

    model_config = get_model_config("llama-70b")

    print(f"\nLlama-70B with GQA (FP16):\n")

    for seq_len in [2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        result = KVCacheCalculator.calculate_from_model_config(
            model_config=model_config,
            seq_len=seq_len,
            batch_size=1,
            dtype=DataType.FP16
        )
        print(f"  {seq_len:6d} tokens: {result['gb']:7.4f} GB ({result['kb_per_token']:8.2f} KB/token)")


def example_custom_model():
    """Example 7: Custom model configuration."""
    print("\n" + "=" * 70)
    print("Example 7: Custom Model Configuration")
    print("=" * 70)

    # Define a custom model
    custom_model = {
        "num_layers": 48,
        "num_attention_heads": 40,
        "hidden_size": 5120,
        "num_key_value_heads": 40,
    }

    result = KVCacheCalculator.calculate_from_model_config(
        model_config=custom_model,
        seq_len=4096,
        batch_size=2,
        dtype=DataType.BF16
    )

    print("\nCustom Model (48 layers, 40 heads, 5120 hidden size):")
    print(f"  Context: 4096 tokens, Batch: 2, Dtype: BF16")
    print(f"  Per token: {result['kb_per_token']:.2f} KB")
    print(f"  Total: {result['gb']:.4f} GB")


def example_deepseek_v3_mla():
    """Example 8: DeepSeek-V3 with Multi-Latent Attention."""
    print("\n" + "=" * 70)
    print("Example 8: DeepSeek-V3 with Multi-Latent Attention (MLA)")
    print("=" * 70)

    deepseek_config = get_model_config("deepseek-v3")

    print(f"\nDeepSeek-V3 uses MLA with only 1 KV head!")
    print(f"Model config: {deepseek_config['num_layers']} layers, "
          f"{deepseek_config['num_attention_heads']} heads, "
          f"{deepseek_config['num_key_value_heads']} KV head\n")

    seq_len = 8192
    for dtype in [DataType.FP16, DataType.FP8]:
        result = KVCacheCalculator.calculate_from_model_config(
            model_config=deepseek_config,
            seq_len=seq_len,
            batch_size=1,
            dtype=dtype
        )
        print(f"  {seq_len} tokens ({dtype.dtype_name}): {result['mb']:8.2f} MB "
              f"({result['kb_per_token']:6.2f} KB/token)")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print(" KV Cache Calculator - Usage Examples")
    print("=" * 70)

    example_basic_calculation()
    example_predefined_models()
    example_attention_mechanisms()
    example_data_types()
    example_batch_size_impact()
    example_context_length_scaling()
    example_custom_model()
    example_deepseek_v3_mla()

    print("\n" + "=" * 70)
    print(" All examples completed!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
