# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""test convert weight"""

import os
import sys
import shutil
from unittest.mock import patch
import torch
import pytest
import safetensors.torch
import convert_weight
from toolkit.weight_convert.deepseekv3.convert_deepseekv3_hf_weight import str2bool, trans_rope_weight, weight_dequant

# Add project root to path to allow imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class MockPool:
    """Mock multiprocessing.Pool to execute tasks synchronously in the main process."""

    def __init__(self, processes=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def imap_unordered(self, func, iterable):
        for item in iterable:
            yield func(item)


class TestConvertWeight:
    """Test convert_weight.py"""
    @classmethod
    def setup_class(cls):
        """ Setup test directory """
        cls.test_dir = os.path.join(PROJECT_ROOT, "tests", "output", "test_convert_weight")
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        os.makedirs(cls.test_dir)

        cls.input_dir = os.path.join(cls.test_dir, "input_hf")
        cls.output_dir = os.path.join(cls.test_dir, "output_ms")
        os.makedirs(cls.input_dir)
        os.makedirs(cls.output_dir)

        cls.script_path = os.path.join(PROJECT_ROOT, "convert_weight.py")

    @classmethod
    def teardown_class(cls):
        """ Cleanup after tests """
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def create_dummy_deepseek_weights(self, path, num_layers=1, with_mtp=False):
        """Create dummy safetensors weights for DeepSeekV3 (Dense layers only for simplicity)"""
        if not safetensors:
            print("Skipping weight creation: safetensors not installed")
            return False

        hidden_size = 64  # Small size for testing
        vocab_size = 100
        moe_ffn_hidden_size = 32
        num_routed_experts = 4

        tensors = {}

        # Embeddings and Head
        tensors["model.embed_tokens.weight"] = torch.randn(vocab_size, hidden_size)
        tensors["model.norm.weight"] = torch.randn(hidden_size)
        tensors["lm_head.weight"] = torch.randn(vocab_size, hidden_size)

        total_layers = num_layers + (1 if with_mtp else 0)

        # Layers
        for i in range(total_layers):
            # Attention (MLA) keys expected by _trans_model_layer_attn_hf_to_ms
            tensors[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(hidden_size, hidden_size)
            tensors[f"model.layers.{i}.self_attn.q_a_layernorm.weight"] = torch.randn(hidden_size)
            tensors[f"model.layers.{i}.self_attn.kv_a_layernorm.weight"] = torch.randn(hidden_size)
            tensors[f"model.layers.{i}.self_attn.q_b_proj.weight"] = torch.randn(hidden_size, hidden_size)  # q_up
            tensors[f"model.layers.{i}.self_attn.kv_b_proj.weight"] = torch.randn(hidden_size, hidden_size)  # kv_up
            tensors[f"model.layers.{i}.self_attn.q_a_proj.weight"] = torch.randn(hidden_size, hidden_size)  # q_down
            tensors[f"model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight"] = torch.randn(hidden_size,
                                                                                           hidden_size)  # kv_down

            # Layer Norms
            tensors[f"model.layers.{i}.input_layernorm.weight"] = torch.randn(hidden_size)
            tensors[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.randn(hidden_size)

            # MTP Specific weights for the last layer if MTP enabled
            if with_mtp and i == total_layers - 1:
                tensors[f"model.layers.{i}.enorm.weight"] = torch.randn(hidden_size)
                tensors[f"model.layers.{i}.hnorm.weight"] = torch.randn(hidden_size)
                tensors[f"model.layers.{i}.eh_proj.weight"] = torch.randn(hidden_size, hidden_size)
                tensors[f"model.layers.{i}.embed_tokens.weight"] = torch.randn(vocab_size, hidden_size)
                tensors[f"model.layers.{i}.shared_head.norm.weight"] = torch.randn(hidden_size)

            # MLP Logic (Mixed Dense and MoE)
            if i == 0:
                # Dense Layer
                tensors[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(hidden_size, hidden_size)
                tensors[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(hidden_size, hidden_size)
                tensors[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(hidden_size, hidden_size)
            else:
                # MoE weights
                tensors[f"model.layers.{i}.mlp.gate.weight"] = torch.randn(num_routed_experts, hidden_size)
                tensors[f"model.layers.{i}.mlp.gate.e_score_correction_bias"] = torch.randn(num_routed_experts)
                tensors[f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"] = torch.randn(moe_ffn_hidden_size,
                                                                                               hidden_size)
                tensors[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"] = torch.randn(moe_ffn_hidden_size,
                                                                                             hidden_size)
                tensors[f"model.layers.{i}.mlp.shared_experts.down_proj.weight"] = torch.randn(hidden_size,
                                                                                               moe_ffn_hidden_size)

                for exp_id in range(num_routed_experts):
                    tensors[f"model.layers.{i}.mlp.experts.{exp_id}.gate_proj.weight"] = torch.randn(
                        moe_ffn_hidden_size, hidden_size)
                    tensors[f"model.layers.{i}.mlp.experts.{exp_id}.up_proj.weight"] = torch.randn(moe_ffn_hidden_size,
                                                                                                   hidden_size)
                    tensors[f"model.layers.{i}.mlp.experts.{exp_id}.down_proj.weight"] = \
                        torch.randn(hidden_size, moe_ffn_hidden_size)

            # Dummy weight for dequant test (FP8 scale_inv)
            if i == 0:
                # Fake FP8 weight
                tensors[f"model.layers.{i}.self_attn.q_b_proj.weight"] = torch.randint(0, 127,
                                                                                       (hidden_size, hidden_size),
                                                                                       dtype=torch.int8)
                tensors[f"model.layers.{i}.self_attn.q_b_proj.weight_scale_inv"] = torch.randn(1, 1)

        save_path = os.path.join(path, "model.safetensors")
        safetensors.torch.save_file(tensors, save_path)
        return True

    def create_dummy_ms_deepseek_weights(self, path, num_layers=1, with_mtp=False):
        """Create dummy MS safetensors weights for DeepSeekV3 Reverse (MS -> HF)"""
        if not safetensors:
            return False

        hidden_size = 64
        vocab_size = 100
        ffn_hidden_size = 64
        moe_ffn_hidden_size = 32
        num_routed_experts = 4

        tensors = {}
        # MS format keys
        tensors["embedding.word_embeddings.weight"] = torch.randn(vocab_size, hidden_size)
        tensors["decoder.final_layernorm.weight"] = torch.randn(hidden_size)
        tensors["output_layer.weight"] = torch.randn(vocab_size, hidden_size)

        total_layers = num_layers + (1 if with_mtp else 0)

        for i in range(total_layers):
            # Determine if this is an MTP layer or regular layer
            is_mtp = False
            layer_idx = i
            if i >= num_layers:
                is_mtp = True
                layer_idx = i - num_layers
                layer_prefix = f"mtp.layers.{layer_idx}.transformer_layer"

                # MTP specific weights
                tensors[f"mtp.layers.{layer_idx}.enorm.weight"] = torch.randn(hidden_size)
                tensors[f"mtp.layers.{layer_idx}.hnorm.weight"] = torch.randn(hidden_size)
                tensors[f"mtp.layers.{layer_idx}.eh_proj.weight"] = torch.randn(hidden_size, hidden_size)
                tensors[f"mtp.layers.{layer_idx}.final_layernorm.weight"] = torch.randn(hidden_size)
            else:
                layer_prefix = f"decoder.layers.{layer_idx}"

            # MLA
            tensors[f"{layer_prefix}.input_layernorm.weight"] = torch.randn(hidden_size)
            tensors[f"{layer_prefix}.self_attention.linear_q_down_proj.weight"] = torch.randn(hidden_size, hidden_size)
            tensors[f"{layer_prefix}.self_attention.linear_kv_down_proj.weight"] = torch.randn(hidden_size, hidden_size)
            tensors[f"{layer_prefix}.self_attention.q_layernorm.weight"] = torch.randn(hidden_size)
            tensors[f"{layer_prefix}.self_attention.kv_layernorm.weight"] = torch.randn(hidden_size)
            tensors[f"{layer_prefix}.self_attention.linear_q_up_proj.weight"] = torch.randn(hidden_size, hidden_size)
            tensors[f"{layer_prefix}.self_attention.linear_kv_up_proj.weight"] = torch.randn(hidden_size, hidden_size)
            tensors[f"{layer_prefix}.self_attention.linear_proj.weight"] = torch.randn(hidden_size, hidden_size)

            # Norm before MLP
            tensors[f"{layer_prefix}.pre_mlp_layernorm.weight"] = torch.randn(hidden_size)

            # MLP: Dense or MoE
            # We simulate layer 0 as Dense, others (including MTP) as MoE
            # hf_origin_layer_id logic in script: (num_layers + layer_idx) if mtp else layer_idx
            hf_layer_id = num_layers + layer_idx if is_mtp else layer_idx

            if hf_layer_id == 0:
                # Dense
                tensors[f"{layer_prefix}.mlp.linear_fc1.weight"] = torch.randn(ffn_hidden_size * 2, hidden_size)
                tensors[f"{layer_prefix}.mlp.linear_fc2.weight"] = torch.randn(hidden_size, ffn_hidden_size)
            else:
                # MoE
                tensors[f"{layer_prefix}.mlp.router.weight"] = torch.randn(num_routed_experts, hidden_size)
                tensors[f"{layer_prefix}.mlp.router.expert_bias"] = torch.randn(num_routed_experts)

                # Shared experts (Dense-like but small)
                tensors[f"{layer_prefix}.mlp.shared_experts.linear_fc1.weight"] = \
                    torch.randn(moe_ffn_hidden_size * 2, hidden_size)
                tensors[f"{layer_prefix}.mlp.shared_experts.linear_fc2.weight"] = \
                    torch.randn(hidden_size, moe_ffn_hidden_size)

                # Routed Experts
                # weight1: (num * hidden, 2 * moe)
                tensors[f"{layer_prefix}.mlp.experts.weight1"] = \
                    torch.randn(num_routed_experts * hidden_size, moe_ffn_hidden_size * 2)
                # weight2: (num * moe, hidden)
                tensors[f"{layer_prefix}.mlp.experts.weight2"] = \
                    torch.randn(num_routed_experts * moe_ffn_hidden_size, hidden_size)

        save_path = os.path.join(path, "model.safetensors")
        safetensors.torch.save_file(tensors, save_path)
        return True

    def create_dummy_ms_qwen3_weights(self, path, num_layers=1):
        """Create dummy MS safetensors weights for Qwen3 Reverse"""
        if not safetensors: return False

        hidden_size = 64
        ffn_hidden_size = 32
        vocab_size = 100

        # QKV calculation for split_qkv_weight
        # qkv_weight shape: [((num_attention_heads // n_kv_heads) + 2) * head_dim * n_kv_heads, hidden_size]

        tensors = {}
        tensors["embedding.word_embeddings.weight"] = torch.randn(vocab_size, hidden_size)
        tensors["decoder.final_layernorm.weight"] = torch.randn(hidden_size)
        tensors["output_layer.weight"] = torch.randn(vocab_size, hidden_size)

        for i in range(num_layers):
            layer_prefix = f"decoder.layers.{i}"
            tensors[f"{layer_prefix}.input_layernorm.weight"] = torch.randn(hidden_size)
            # Merged QKV
            tensors[f"{layer_prefix}.self_attention.linear_qkv.weight"] = torch.randn(32, hidden_size)
            tensors[f"{layer_prefix}.self_attention.q_layernorm.weight"] = torch.randn(hidden_size)
            tensors[f"{layer_prefix}.self_attention.k_layernorm.weight"] = torch.randn(hidden_size)
            tensors[f"{layer_prefix}.self_attention.linear_proj.weight"] = torch.randn(hidden_size, hidden_size)

            tensors[f"{layer_prefix}.pre_mlp_layernorm.weight"] = torch.randn(hidden_size)
            # Merged Gate+Up
            tensors[f"{layer_prefix}.mlp.linear_fc1.weight"] = torch.randn(ffn_hidden_size * 2, hidden_size)
            tensors[f"{layer_prefix}.mlp.linear_fc2.weight"] = torch.randn(hidden_size, ffn_hidden_size)

        save_path = os.path.join(path, "model.safetensors")
        safetensors.torch.save_file(tensors, save_path)
        return True

    def create_dummy_ms_qwen3_moe_weights(self, path, num_layers=1):
        """Create dummy MS safetensors weights for Qwen3-MoE Reverse"""
        if not safetensors: return False

        hidden_size = 64
        moe_ffn_hidden_size = 32
        num_routed_experts = 4
        vocab_size = 100

        tensors = {}
        tensors["embedding.word_embeddings.weight"] = torch.randn(vocab_size, hidden_size)
        tensors["decoder.final_layernorm.weight"] = torch.randn(hidden_size)
        tensors["output_layer.weight"] = torch.randn(vocab_size, hidden_size)

        for i in range(num_layers):
            layer_prefix = f"decoder.layers.{i}"
            # Norms
            tensors[f"{layer_prefix}.pre_mlp_layernorm.weight"] = torch.randn(hidden_size)
            tensors[f"{layer_prefix}.input_layernorm.weight"] = torch.randn(hidden_size)

            # MLA (same as Qwen3 basically, but simplified here for MoE focus if needed,
            # but script calls _mla_ms_to_pt so we need it)
            tensors[f"{layer_prefix}.self_attention.linear_qkv.weight"] = torch.randn(32,
                                                                                      hidden_size)  # reused dims from Qwen3
            tensors[f"{layer_prefix}.self_attention.q_layernorm.weight"] = torch.randn(hidden_size)
            tensors[f"{layer_prefix}.self_attention.k_layernorm.weight"] = torch.randn(hidden_size)
            tensors[f"{layer_prefix}.self_attention.linear_proj.weight"] = torch.randn(hidden_size, hidden_size)

            # MoE
            tensors[f"{layer_prefix}.mlp.router.weight"] = torch.randn(num_routed_experts, hidden_size)
            # Experts
            # weight1: [num_routed_experts * hidden_size, 2 * moe_ffn_hidden_size] - Wait, script reshapes
            # Script: experts_weight1.reshape(num_routed_experts, hidden_size, moe_ffn_hidden_size * 2)
            # So total elements matches.
            tensors[f"{layer_prefix}.mlp.experts.weight1"] = \
                torch.randn(num_routed_experts * hidden_size, 2 * moe_ffn_hidden_size)
            # weight2
            tensors[f"{layer_prefix}.mlp.experts.weight2"] = \
                torch.randn(num_routed_experts * hidden_size, moe_ffn_hidden_size)

        save_path = os.path.join(path, "model.safetensors")
        safetensors.torch.save_file(tensors, save_path)
        return True

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_arg_parsing_error(self):
        """Test that script fails gracefully without required args"""
        # Calling main without args or with empty args should raise error or exit
        # argparse usually exits with code 2
        with pytest.raises(SystemExit) as cm:
            convert_weight.main([])
        assert cm.value.code != 0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_deepseekv3_conversion(self):
        """Test DeepSeekV3 Forward Conversion (HF -> MS) with MTP and MoE coverage"""
        if not safetensors:
            pytest.skip("safetensors library not found")

        # 1. Create dummy weights (Num layers=2: 1 Dense, 1 MoE + 1 MTP if configured)
        success = self.create_dummy_deepseek_weights(self.input_dir, num_layers=2, with_mtp=True)
        assert success, "Failed to create dummy weights"

        # 2. Run conversion script
        args = [
            "--model", "deepseekv3",
            "--input_path", self.input_dir,
            "--output_path", self.output_dir,
            "--num_layers", "2",
            "--hidden_size", "64",
            "--ffn_hidden_size", "64",
            "--moe_ffn_hidden_size", "32",
            "--num_routed_experts", "4",
            "--num_nextn_predict_layers", "1",
            "--first_k_dense_replace", "1",  # Layer 0 Dense, Layer 1 MoE
            "--dtype", "bf16",
            "--qkv_concat", "True"
        ]

        print(f"Running main with args: {args}")
        try:
            convert_weight.main(args)
        except SystemExit as e:
            pytest.fail(f"convert_weight.main raised SystemExit: {e}")
        except Exception as e:
            pytest.fail(f"convert_weight.main failed with error: {e}")

        # 3. Verify output
        assert os.path.exists(os.path.join(self.output_dir, "ms-model-00001-of-00003.safetensors"))
        assert os.path.exists(os.path.join(self.output_dir, "ms-model-00002-of-00003.safetensors"))
        assert os.path.exists(os.path.join(self.output_dir, "ms-model-00003-of-00003.safetensors"))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_deepseek_utils_coverage(self):
        """Directly test helper functions in deepseek script if they are not reachable via main flow"""
        # Test str2bool
        assert str2bool("true")
        assert str2bool("True")
        assert not str2bool("false")
        with pytest.raises(Exception):
            str2bool("invalid")

        # Test trans_rope_weight
        dim = 4
        weight = torch.randn(1, 8, dim)
        res = trans_rope_weight(weight.clone(), 4)
        assert res.shape == weight.shape

        # Test weight_dequant error cases
        w = torch.randn(128, 128)
        s = torch.randn(1, 1)
        res = weight_dequant(w, s, block_size=128)
        assert res.shape == w.shape

        with pytest.raises(ValueError):
            weight_dequant(w, torch.randn(2, 1), block_size=128)

        with pytest.raises(ValueError):
            weight_dequant(torch.randn(128), s)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_dispatch_mechanism(self):
        """Test that convert_weight.py correctly resolves different models"""
        args = [
            "--model", "invalid_model_name",
            "--input_path", "dummy",
            "--output_path", "dummy"
        ]

        with pytest.raises(ValueError) as cm:
            convert_weight.main(args)
        assert "is not supported" in str(cm.value)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_deepseekv3_reverse_conversion(self):
        """Test DeepSeekV3 Reverse Conversion (MS -> HF) with MTP and MoE"""
        if not safetensors:
            pytest.skip("safetensors library not found")

        ms_dir = os.path.join(self.test_dir, "deepseek_ms_input")
        hf_dir = os.path.join(self.test_dir, "deepseek_hf_output")
        os.makedirs(ms_dir, exist_ok=True)
        os.makedirs(hf_dir, exist_ok=True)

        # Create weights for 2 layers (0:Dense, 1:MoE) + 1 MTP layer
        self.create_dummy_ms_deepseek_weights(ms_dir, num_layers=2, with_mtp=True)

        args = [
            "--model", "deepseekv3",
            "--reversed",
            "--input_path", ms_dir,
            "--output_path", hf_dir,
            "--num_layers", "2",
            "--hidden_size", "64",
            "--ffn_hidden_size", "64",
            "--moe_ffn_hidden_size", "32",
            "--num_routed_experts", "4",
            "--num_nextn_predict_layers", "1",
            "--first_k_dense_replace", "1"  # Layer 0 Dense, Layer 1 MoE
        ]

        try:
            convert_weight.main(args)
        except Exception as e:
            pytest.fail(f"convert_weight.main (reverse) failed with error: {e}")

        # Check output files
        # Total layers = 2 (base) + 1 (MTP) = 3
        assert os.path.exists(os.path.join(hf_dir, "model-00001-of-00003.safetensors"))
        assert os.path.exists(os.path.join(hf_dir, "model-00002-of-00003.safetensors"))
        assert os.path.exists(os.path.join(hf_dir, "model-00003-of-00003.safetensors"))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_qwen3_reverse_conversion(self):
        """Test Qwen3 Reverse Conversion (MS -> HF)"""
        if not safetensors:
            pytest.skip("safetensors library not found")

        ms_dir = os.path.join(self.test_dir, "qwen3_ms_input")
        hf_dir = os.path.join(self.test_dir, "qwen3_hf_output")
        os.makedirs(ms_dir, exist_ok=True)
        os.makedirs(hf_dir, exist_ok=True)

        self.create_dummy_ms_qwen3_weights(ms_dir, num_layers=1)

        args = [
            "--model", "qwen3",
            "--reversed",
            "--input_path", ms_dir,
            "--output_path", hf_dir,
            "--num_layers", "1",
            "--hidden_size", "64",
            "--ffn_hidden_size", "32",
            "--num_attention_heads", "4",
            "--num_query_groups", "2",
            "--kv_channels", "4",
            "--max_worker", "1"
        ]

        # Patch multiprocessing.Pool to avoid subprocess coverage issues
        with patch('multiprocessing.Pool', new=MockPool):
            try:
                convert_weight.main(args)
            except Exception as e:
                pytest.fail(f"convert_weight.main (qwen3 reverse) failed with error: {e}")

        assert os.path.exists(os.path.join(hf_dir, "model-00001-of-00001.safetensors"))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_qwen3_moe_reverse_conversion(self):
        """Test Qwen3-MoE Reverse Conversion (MS -> HF)"""
        if not safetensors:
            pytest.skip("safetensors library not found")

        ms_dir = os.path.join(self.test_dir, "qwen3_moe_ms_input")
        hf_dir = os.path.join(self.test_dir, "qwen3_moe_hf_output")
        os.makedirs(ms_dir, exist_ok=True)
        os.makedirs(hf_dir, exist_ok=True)

        self.create_dummy_ms_qwen3_moe_weights(ms_dir, num_layers=1)

        args = [
            "--model", "qwen3-moe",
            "--reversed",
            "--input_path", ms_dir,
            "--output_path", hf_dir,
            "--num_layers", "1",
            "--hidden_size", "64",
            "--moe_ffn_hidden_size", "32",
            "--num_routed_experts", "4",
            "--num_attention_heads", "4",
            "--num_query_groups", "2",
            "--kv_channels", "4",
            "--max_worker", "1"
        ]

        # Patch multiprocessing.Pool to avoid subprocess coverage issues
        with patch('multiprocessing.Pool', new=MockPool):
            try:
                convert_weight.main(args)
            except Exception as e:
                pytest.fail(f"convert_weight.main (qwen3-moe reverse) failed with error: {e}")

        assert os.path.exists(os.path.join(hf_dir, "model-00001-of-00001.safetensors"))
