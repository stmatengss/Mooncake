#!/usr/bin/env python3
"""
KV Cache Calculator Module

This module provides utilities to calculate KV cache memory requirements
for transformer-based language models. It supports various attention mechanisms
(MHA, GQA, MLA) and data types (FP16, BF16, FP8, FP4).

Inspired by the KV cache calculator from kvcache.ai blog.
"""

from typing import Dict, Optional, Union
from enum import Enum


class AttentionType(Enum):
    """Attention mechanism types."""
    MHA = "mha"  # Multi-Head Attention
    GQA = "gqa"  # Grouped Query Attention
    MLA = "mla"  # Multi-Latent Attention


class DataType(Enum):
    """Data type for KV cache storage."""
    FP32 = ("fp32", 4)
    FP16 = ("fp16", 2)
    BF16 = ("bf16", 2)
    FP8 = ("fp8", 1)
    FP4 = ("fp4", 0.5)
    INT8 = ("int8", 1)

    def __init__(self, name: str, bytes_per_element: float):
        self.dtype_name = name
        self.bytes_per_element = bytes_per_element


class KVCacheCalculator:
    """
    Calculator for KV cache memory requirements.

    The KV cache stores key and value tensors from attention layers during inference.
    Memory usage grows with context length, batch size, and model size.
    """

    def __init__(self):
        """Initialize the KV cache calculator."""
        pass

    @staticmethod
    def calculate_kv_cache_size(
        num_layers: int,
        num_heads: int,
        head_dim: int,
        seq_len: int,
        batch_size: int = 1,
        dtype: DataType = DataType.FP16,
        num_kv_heads: Optional[int] = None,
        attention_type: AttentionType = AttentionType.MHA
    ) -> Dict[str, float]:
        """
        Calculate KV cache memory size.

        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            seq_len: Sequence length (number of tokens to cache)
            batch_size: Number of concurrent sequences (default: 1)
            dtype: Data type for storage (default: FP16)
            num_kv_heads: Number of KV heads for GQA (default: None, uses num_heads)
            attention_type: Type of attention mechanism (default: MHA)

        Returns:
            Dictionary with memory sizes in different units:
                - bytes: Total size in bytes
                - kb: Total size in kilobytes
                - mb: Total size in megabytes
                - gb: Total size in gigabytes
                - bytes_per_token: Size per token in bytes
                - kb_per_token: Size per token in kilobytes

        Formula (for MHA/GQA):
            KV Cache Size = num_layers × batch_size × seq_len × num_kv_heads × head_dim × dtype_bytes × 2

            The factor of 2 accounts for both key and value tensors.
        """
        # Determine effective number of KV heads based on attention type
        if attention_type == AttentionType.GQA:
            effective_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        elif attention_type == AttentionType.MHA:
            effective_kv_heads = num_heads
        else:  # MLA or other
            effective_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        # Calculate total bytes
        # Each layer stores K and V (factor of 2)
        bytes_total = (
            num_layers *
            batch_size *
            seq_len *
            effective_kv_heads *
            head_dim *
            dtype.bytes_per_element *
            2  # for both key and value
        )

        # Calculate per-token size
        bytes_per_token = (
            num_layers *
            effective_kv_heads *
            head_dim *
            dtype.bytes_per_element *
            2
        )

        return {
            "bytes": bytes_total,
            "kb": bytes_total / 1024,
            "mb": bytes_total / (1024 ** 2),
            "gb": bytes_total / (1024 ** 3),
            "bytes_per_token": bytes_per_token,
            "kb_per_token": bytes_per_token / 1024,
        }

    @staticmethod
    def calculate_from_model_config(
        model_config: Dict[str, Union[int, str]],
        seq_len: int,
        batch_size: int = 1,
        dtype: Optional[DataType] = None
    ) -> Dict[str, float]:
        """
        Calculate KV cache size from a model configuration dictionary.

        Args:
            model_config: Dictionary containing model parameters:
                - num_layers (or num_hidden_layers)
                - num_attention_heads (or num_heads)
                - hidden_size (to derive head_dim) or head_dim
                - num_key_value_heads (optional, for GQA)
            seq_len: Sequence length
            batch_size: Batch size (default: 1)
            dtype: Data type (default: FP16)

        Returns:
            Dictionary with memory sizes
        """
        # Extract parameters with various naming conventions
        num_layers = model_config.get("num_layers") or model_config.get("num_hidden_layers")
        num_heads = model_config.get("num_attention_heads") or model_config.get("num_heads")

        # Calculate head_dim
        if "head_dim" in model_config:
            head_dim = model_config["head_dim"]
        elif "hidden_size" in model_config and num_heads:
            head_dim = model_config["hidden_size"] // num_heads
        else:
            raise ValueError("Cannot determine head_dim from model config")

        # Check for GQA configuration
        num_kv_heads = model_config.get("num_key_value_heads")
        attention_type = AttentionType.GQA if num_kv_heads and num_kv_heads != num_heads else AttentionType.MHA

        if dtype is None:
            dtype = DataType.FP16

        return KVCacheCalculator.calculate_kv_cache_size(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            batch_size=batch_size,
            dtype=dtype,
            num_kv_heads=num_kv_heads,
            attention_type=attention_type
        )

    @staticmethod
    def print_summary(
        cache_size: Dict[str, float],
        model_name: Optional[str] = None,
        seq_len: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> None:
        """
        Print a formatted summary of KV cache size.

        Args:
            cache_size: Dictionary returned by calculate_kv_cache_size
            model_name: Optional model name for display
            seq_len: Optional sequence length for display
            batch_size: Optional batch size for display
        """
        print("=" * 60)
        if model_name:
            print(f"Model: {model_name}")
        if seq_len:
            print(f"Sequence Length: {seq_len:,} tokens")
        if batch_size:
            print(f"Batch Size: {batch_size}")
        print("-" * 60)

        print(f"KV Cache Memory per Token:")
        print(f"  {cache_size['bytes_per_token']:.2f} bytes ({cache_size['kb_per_token']:.2f} KB)")

        print(f"\nTotal KV Cache Memory:")
        print(f"  {cache_size['bytes']:,.0f} bytes")
        print(f"  {cache_size['kb']:,.2f} KB")
        print(f"  {cache_size['mb']:,.2f} MB")
        print(f"  {cache_size['gb']:,.4f} GB")
        print("=" * 60)


# Predefined model configurations
COMMON_MODELS = {
    "llama-7b": {
        "num_layers": 32,
        "num_attention_heads": 32,
        "hidden_size": 4096,
        "num_key_value_heads": 32,
    },
    "llama-13b": {
        "num_layers": 40,
        "num_attention_heads": 40,
        "hidden_size": 5120,
        "num_key_value_heads": 40,
    },
    "llama-70b": {
        "num_layers": 80,
        "num_attention_heads": 64,
        "hidden_size": 8192,
        "num_key_value_heads": 8,  # GQA
    },
    "qwen-7b": {
        "num_layers": 32,
        "num_attention_heads": 32,
        "hidden_size": 4096,
        "num_key_value_heads": 32,
    },
    "deepseek-v3": {
        "num_layers": 61,
        "num_attention_heads": 128,
        "hidden_size": 7168,
        "num_key_value_heads": 1,  # MLA
    },
}


def get_model_config(model_name: str) -> Optional[Dict[str, int]]:
    """
    Get predefined configuration for common models.

    Args:
        model_name: Name of the model

    Returns:
        Model configuration dictionary or None if not found
    """
    return COMMON_MODELS.get(model_name.lower())


def main():
    """Command-line interface for KV cache calculator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate KV cache memory requirements for LLM inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Predefined model name (e.g., llama-7b, llama-70b, qwen-7b, deepseek-v3)"
    )
    parser.add_argument("--num-layers", type=int, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, help="Dimension per attention head")
    parser.add_argument("--hidden-size", type=int, help="Hidden size (alternative to head-dim)")
    parser.add_argument("--num-kv-heads", type=int, help="Number of KV heads (for GQA)")
    parser.add_argument("--seq-len", type=int, required=True, help="Sequence length (context window)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16", "fp8", "fp4", "int8"],
        help="Data type (default: fp16)"
    )

    args = parser.parse_args()

    # Get dtype
    dtype_map = {
        "fp32": DataType.FP32,
        "fp16": DataType.FP16,
        "bf16": DataType.BF16,
        "fp8": DataType.FP8,
        "fp4": DataType.FP4,
        "int8": DataType.INT8,
    }
    dtype = dtype_map[args.dtype]

    # Calculate based on model or manual parameters
    if args.model:
        model_config = get_model_config(args.model)
        if model_config is None:
            print(f"Error: Unknown model '{args.model}'")
            print(f"Available models: {', '.join(COMMON_MODELS.keys())}")
            return 1

        cache_size = KVCacheCalculator.calculate_from_model_config(
            model_config=model_config,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            dtype=dtype
        )
        KVCacheCalculator.print_summary(
            cache_size,
            model_name=args.model,
            seq_len=args.seq_len,
            batch_size=args.batch_size
        )
    else:
        # Manual calculation
        if not all([args.num_layers, args.num_heads]):
            print("Error: Must specify either --model or (--num-layers, --num-heads, and --head-dim/--hidden-size)")
            return 1

        # Determine head_dim
        if args.head_dim:
            head_dim = args.head_dim
        elif args.hidden_size:
            head_dim = args.hidden_size // args.num_heads
        else:
            print("Error: Must specify either --head-dim or --hidden-size")
            return 1

        attention_type = AttentionType.GQA if args.num_kv_heads and args.num_kv_heads != args.num_heads else AttentionType.MHA

        cache_size = KVCacheCalculator.calculate_kv_cache_size(
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            head_dim=head_dim,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            dtype=dtype,
            num_kv_heads=args.num_kv_heads,
            attention_type=attention_type
        )
        KVCacheCalculator.print_summary(
            cache_size,
            seq_len=args.seq_len,
            batch_size=args.batch_size
        )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
