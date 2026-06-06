#!/usr/bin/env python3
"""
Unit tests for KV Cache Calculator module.
"""

import unittest
from mooncake.kvcache_calculator import (
    KVCacheCalculator,
    DataType,
    AttentionType,
    get_model_config,
    COMMON_MODELS,
)


class TestDataType(unittest.TestCase):
    """Test DataType enum."""

    def test_dtype_bytes(self):
        """Test that data types have correct byte sizes."""
        self.assertEqual(DataType.FP32.bytes_per_element, 4)
        self.assertEqual(DataType.FP16.bytes_per_element, 2)
        self.assertEqual(DataType.BF16.bytes_per_element, 2)
        self.assertEqual(DataType.FP8.bytes_per_element, 1)
        self.assertEqual(DataType.FP4.bytes_per_element, 0.5)
        self.assertEqual(DataType.INT8.bytes_per_element, 1)


class TestKVCacheCalculator(unittest.TestCase):
    """Test KVCacheCalculator class."""

    def test_basic_calculation(self):
        """Test basic KV cache calculation."""
        result = KVCacheCalculator.calculate_kv_cache_size(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            seq_len=2048,
            batch_size=1,
            dtype=DataType.FP16
        )

        # Expected: 32 layers * 1 batch * 2048 seq * 32 heads * 128 dim * 2 bytes * 2 (K+V)
        expected_bytes = 32 * 1 * 2048 * 32 * 128 * 2 * 2
        self.assertEqual(result["bytes"], expected_bytes)
        self.assertAlmostEqual(result["mb"], expected_bytes / (1024 ** 2), places=2)
        self.assertAlmostEqual(result["gb"], expected_bytes / (1024 ** 3), places=4)

    def test_per_token_calculation(self):
        """Test per-token KV cache calculation."""
        result = KVCacheCalculator.calculate_kv_cache_size(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            seq_len=1,  # Single token
            batch_size=1,
            dtype=DataType.FP16
        )

        # Expected per token: 32 layers * 32 heads * 128 dim * 2 bytes * 2 (K+V)
        expected_per_token = 32 * 32 * 128 * 2 * 2
        self.assertEqual(result["bytes_per_token"], expected_per_token)

    def test_batch_size_scaling(self):
        """Test that batch size scales linearly."""
        result_batch1 = KVCacheCalculator.calculate_kv_cache_size(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            seq_len=2048,
            batch_size=1,
            dtype=DataType.FP16
        )

        result_batch4 = KVCacheCalculator.calculate_kv_cache_size(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            seq_len=2048,
            batch_size=4,
            dtype=DataType.FP16
        )

        self.assertEqual(result_batch4["bytes"], result_batch1["bytes"] * 4)
        # Per-token size should be the same regardless of batch size
        self.assertEqual(result_batch4["bytes_per_token"], result_batch1["bytes_per_token"])

    def test_gqa_attention(self):
        """Test Grouped Query Attention (GQA) calculation."""
        # GQA with 8 KV heads instead of 32 heads
        result_gqa = KVCacheCalculator.calculate_kv_cache_size(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            seq_len=2048,
            batch_size=1,
            dtype=DataType.FP16,
            num_kv_heads=8,
            attention_type=AttentionType.GQA
        )

        # Expected: 32 layers * 1 batch * 2048 seq * 8 kv_heads * 128 dim * 2 bytes * 2 (K+V)
        expected_bytes = 32 * 1 * 2048 * 8 * 128 * 2 * 2
        self.assertEqual(result_gqa["bytes"], expected_bytes)

        # Should be 4x smaller than MHA with 32 heads
        result_mha = KVCacheCalculator.calculate_kv_cache_size(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            seq_len=2048,
            batch_size=1,
            dtype=DataType.FP16,
            num_kv_heads=32,
            attention_type=AttentionType.MHA
        )
        self.assertEqual(result_mha["bytes"], result_gqa["bytes"] * 4)

    def test_different_dtypes(self):
        """Test calculation with different data types."""
        base_result = KVCacheCalculator.calculate_kv_cache_size(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            seq_len=2048,
            batch_size=1,
            dtype=DataType.FP16
        )

        # FP32 should be 2x larger than FP16
        fp32_result = KVCacheCalculator.calculate_kv_cache_size(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            seq_len=2048,
            batch_size=1,
            dtype=DataType.FP32
        )
        self.assertEqual(fp32_result["bytes"], base_result["bytes"] * 2)

        # FP8 should be 2x smaller than FP16
        fp8_result = KVCacheCalculator.calculate_kv_cache_size(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            seq_len=2048,
            batch_size=1,
            dtype=DataType.FP8
        )
        self.assertEqual(fp8_result["bytes"], base_result["bytes"] / 2)

    def test_from_model_config(self):
        """Test calculation from model configuration."""
        model_config = {
            "num_layers": 32,
            "num_attention_heads": 32,
            "hidden_size": 4096,
            "num_key_value_heads": 32,
        }

        result = KVCacheCalculator.calculate_from_model_config(
            model_config=model_config,
            seq_len=2048,
            batch_size=1,
            dtype=DataType.FP16
        )

        # Head dim should be derived: 4096 / 32 = 128
        expected_head_dim = 128
        expected_bytes = 32 * 1 * 2048 * 32 * expected_head_dim * 2 * 2
        self.assertEqual(result["bytes"], expected_bytes)

    def test_from_model_config_gqa(self):
        """Test calculation from model config with GQA."""
        model_config = {
            "num_layers": 80,
            "num_attention_heads": 64,
            "hidden_size": 8192,
            "num_key_value_heads": 8,  # GQA with 8 KV heads
        }

        result = KVCacheCalculator.calculate_from_model_config(
            model_config=model_config,
            seq_len=4096,
            batch_size=1,
            dtype=DataType.FP16
        )

        # Head dim: 8192 / 64 = 128
        expected_bytes = 80 * 1 * 4096 * 8 * 128 * 2 * 2
        self.assertEqual(result["bytes"], expected_bytes)

    def test_from_model_config_with_head_dim(self):
        """Test calculation when head_dim is provided directly."""
        model_config = {
            "num_layers": 32,
            "num_attention_heads": 32,
            "head_dim": 128,
        }

        result = KVCacheCalculator.calculate_from_model_config(
            model_config=model_config,
            seq_len=2048,
            batch_size=1,
            dtype=DataType.FP16
        )

        expected_bytes = 32 * 1 * 2048 * 32 * 128 * 2 * 2
        self.assertEqual(result["bytes"], expected_bytes)

    def test_predefined_models(self):
        """Test that predefined models can be retrieved and calculated."""
        # Test that common models exist
        self.assertIsNotNone(get_model_config("llama-7b"))
        self.assertIsNotNone(get_model_config("llama-70b"))
        self.assertIsNotNone(get_model_config("qwen-7b"))
        self.assertIsNotNone(get_model_config("deepseek-v3"))

        # Test Llama-7B calculation
        llama_config = get_model_config("llama-7b")
        result = KVCacheCalculator.calculate_from_model_config(
            model_config=llama_config,
            seq_len=2048,
            batch_size=1,
            dtype=DataType.FP16
        )
        self.assertGreater(result["bytes"], 0)
        self.assertGreater(result["mb"], 0)

    def test_llama_70b_gqa(self):
        """Test Llama-70B with GQA (8 KV heads)."""
        llama70b_config = get_model_config("llama-70b")
        self.assertEqual(llama70b_config["num_key_value_heads"], 8)

        result = KVCacheCalculator.calculate_from_model_config(
            model_config=llama70b_config,
            seq_len=4096,
            batch_size=1,
            dtype=DataType.FP16
        )

        # Should use 8 KV heads, not 64
        # 80 layers * 4096 seq * 8 kv_heads * 128 head_dim * 2 bytes * 2 (K+V)
        expected_bytes = 80 * 1 * 4096 * 8 * 128 * 2 * 2
        self.assertEqual(result["bytes"], expected_bytes)

    def test_deepseek_v3_mla(self):
        """Test DeepSeek-V3 with MLA (1 KV head)."""
        deepseek_config = get_model_config("deepseek-v3")
        self.assertEqual(deepseek_config["num_key_value_heads"], 1)

        result = KVCacheCalculator.calculate_from_model_config(
            model_config=deepseek_config,
            seq_len=8192,
            batch_size=1,
            dtype=DataType.FP8
        )

        # 61 layers * 8192 seq * 1 kv_head * 56 head_dim (7168/128) * 1 byte * 2 (K+V)
        head_dim = 7168 // 128
        expected_bytes = 61 * 1 * 8192 * 1 * head_dim * 1 * 2
        self.assertEqual(result["bytes"], expected_bytes)

    def test_units_conversion(self):
        """Test that unit conversions are correct."""
        result = KVCacheCalculator.calculate_kv_cache_size(
            num_layers=1,
            num_heads=1,
            head_dim=1,
            seq_len=1,
            batch_size=1,
            dtype=DataType.FP16
        )

        # 1 * 1 * 1 * 1 * 1 * 2 * 2 = 4 bytes
        self.assertEqual(result["bytes"], 4)
        self.assertEqual(result["kb"], 4 / 1024)
        self.assertEqual(result["mb"], 4 / (1024 ** 2))
        self.assertEqual(result["gb"], 4 / (1024 ** 3))

    def test_print_summary(self):
        """Test that print_summary doesn't crash."""
        result = KVCacheCalculator.calculate_kv_cache_size(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            seq_len=2048,
            batch_size=1,
            dtype=DataType.FP16
        )

        # Just verify it doesn't raise an exception
        try:
            KVCacheCalculator.print_summary(
                result,
                model_name="test-model",
                seq_len=2048,
                batch_size=1
            )
        except Exception as e:
            self.fail(f"print_summary raised an exception: {e}")


class TestModelConfigs(unittest.TestCase):
    """Test model configuration utilities."""

    def test_get_model_config_case_insensitive(self):
        """Test that model lookup is case-insensitive."""
        self.assertIsNotNone(get_model_config("llama-7b"))
        self.assertIsNotNone(get_model_config("LLAMA-7B"))
        self.assertIsNotNone(get_model_config("Llama-7B"))

    def test_get_model_config_unknown(self):
        """Test that unknown models return None."""
        self.assertIsNone(get_model_config("unknown-model"))

    def test_all_models_have_required_fields(self):
        """Test that all predefined models have required fields."""
        required_fields = ["num_layers", "num_attention_heads", "hidden_size"]

        for model_name, config in COMMON_MODELS.items():
            for field in required_fields:
                self.assertIn(
                    field, config,
                    f"Model {model_name} missing required field: {field}"
                )


if __name__ == "__main__":
    unittest.main()
