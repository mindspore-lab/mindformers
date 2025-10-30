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
"""GPTModel weight consistency test."""
# pylint: disable=redefined-outer-name
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import pytest

from mindformers import build_context
from mindformers.parallel_core.training_graph.base_models.gpt.gpt_model import GPTModel as GPTModelTrain
from mindformers.parallel_core.training_graph.base_models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from mindformers.parallel_core.training_graph.base_models.gpt.gpt_layer_specs import \
    get_gpt_decoder_block_spec as get_gpt_decoder_block_spec_train
from mindformers.parallel_core.inference.base_models.gpt.gpt_layer_specs import \
    get_gpt_decoder_block_spec as get_gpt_decoder_block_spec_infer
from mindformers.parallel_core.inference.base_models.gpt.gpt_model import GPTModel as GPTModelInfer
from mindformers.parallel_core.inference.quantization.golden_stick.a8w8 import A8W8LinearMethod
from mindformers.parallel_core.inference.quantization.golden_stick.a8dynw4 import A8W4DynamicLinearMethod
from mindformers.parallel_core.inference.quantization.golden_stick.a8dynw8 import A8W8DynamicLinearMethod
from mindformers.parallel_core.inference.quantization.golden_stick.config import GoldenStickConfig
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups


class DummyGoldenStickConfig(GoldenStickConfig):
    def get_quant_method(self, layer, prefix):
        if "experts" in prefix:
            return A8W4DynamicLinearMethod(self)
        if "self_attn" in prefix:
            return A8W8LinearMethod(self)
        return A8W8DynamicLinearMethod(self)


class GPTModelWeightConsistencyTest:
    """Test GPTModel weight consistency between versions."""

    def __init__(self, test_dir: Path = None):
        self.test_dir = test_dir or Path(__file__).parent
        self.golden_file = self.test_dir / "weight_struct.json"

        # Test configuration
        self.base_config = {
            "hidden_size": 128,
            "ffn_hidden_size": 256,
            "num_attention_heads": 4,
            "num_layers": 2,
            "seq_length": 32,
            "vocab_size": 1000,
            "max_position_embeddings": 128,
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "data_parallel_size": 1,
            "normalization": "RMSNorm",
            "hidden_act": "silu",
            "position_embedding_type": "learned_absolute",
            "add_bias_linear": False,
            "gated_linear_unit": True,
            "mla_qkv_concat": False,
            "use_fused_mla": True,
            "use_flash_attention": True,
            "use_legacy": False,
            "shared_expert_num": 2,
            "moe_grouped_gemm": True,
            "moe_router_score_function": "sigmoid",
            "moe_router_enable_expert_bias": True,
            "use_shared_expert_gating": False,
            "mtp_num_layers": 1,
            "first_k_dense_replace": 1,
        }
        build_context(self.base_config)
        self.base_config.pop("use_legacy")
        self.base_config.pop("local_rank")
        self.base_config.pop("device_num")

        self.train_configurations = [
            {
                "multi_latent_attention": False,
                "mla_qkv_concat": False,
                "num_moe_experts": None
            },
            {
                "multi_latent_attention": True,
                "mla_qkv_concat": False,
                "num_moe_experts": 2
            },
            {
                "multi_latent_attention": True,
                "mla_qkv_concat": True,
                "num_moe_experts": None
            },
        ]

        self.infer_configurations = [
            {
                "multi_latent_attention": False,
                "num_moe_experts": None
            },
            {
                "multi_latent_attention": True,
                "num_moe_experts": 2
            },
        ]

    def _build_model(self,
                     model_type: str,
                     multi_latent_attention: bool = False,
                     mla_qkv_concat: bool = False,
                     num_moe_experts: int = None) -> object:
        """Build real GPTModel."""
        # pylint: disable=unexpected-keyword-arg
        config = MLATransformerConfig(**self.base_config)
        config.multi_latent_attention = multi_latent_attention
        config.mla_qkv_concat = mla_qkv_concat
        config.num_moe_experts = num_moe_experts

        if model_type == "train":
            transformer_layer_spec = get_gpt_decoder_block_spec_train(config)
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec)
            model = GPTModelTrain(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=config.vocab_size,
                max_sequence_length=config.max_position_embeddings,
                position_embedding_type=config.position_embedding_type,
                mtp_block_spec=mtp_block_spec
            )
        elif model_type == "infer":
            # Build inference version model
            config.position_embedding_type = "rope"
            transformer_layer_spec = get_gpt_decoder_block_spec_infer(config, config.normalization)
            model_comm_pgs = ModelCommProcessGroups.get_default_model_comm_pgs()
            quant_config = DummyGoldenStickConfig.from_config({})
            quant_config.fa3_quant = True
            quant_config.fa3_quant_layer = {0}
            quant_config.full_config = {"group_size": 128}
            model = GPTModelInfer(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=config.vocab_size,
                max_sequence_length=config.max_position_embeddings,
                position_embedding_type=config.position_embedding_type,
                model_comm_pgs=model_comm_pgs,
                quant_config=quant_config
            )
        else:
            raise ValueError("model_type only support `train` or `infer`")
        return model

    def _extract_weight_structure(self, model: object) -> Dict[str, List[int]]:
        """Extract weight structure from model."""
        weights = {}
        if hasattr(model, 'parameters_and_names'):
            for name, param in model.parameters_and_names():
                if param.data is not None:
                    weights[name] = list(param.data.shape)
        return weights

    def _generate_golden_standard(self, model_type: str, golden_data, configurations):
        for i, cf in enumerate(configurations):
            model = self._build_model(model_type=model_type, **cf)
            weights = self._extract_weight_structure(model)
            golden_data[f"{model_type}_version"][f"configuration_{i}"] = {}
            golden_data[f"{model_type}_version"][f"configuration_{i}"]["weights"] = weights
            golden_data[f"{model_type}_version"][f"configuration_{i}"]["weight_count"] = len(weights)

    def generate_golden_standard(self) -> Dict:
        """Generate golden weight standard."""
        print("Generating golden weight standard...")

        golden_data = {
            "train_version": {},
            "infer_version": {},
            "metadata": {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        }

        # Training version
        self._generate_golden_standard("train", golden_data, self.train_configurations)

        # Inference version
        self._generate_golden_standard("infer", golden_data, self.infer_configurations)

        # Save to file
        with open(self.golden_file, 'w', encoding='utf-8') as f:
            json.dump(golden_data, f, indent=2)

        return golden_data

    def _test_weight_consistency(self, model_type: str, all_differences, golden_data, configurations):
        for i, cf in enumerate(configurations):
            model = self._build_model(model_type=model_type, **cf)
            current_weights = self._extract_weight_structure(model)
            golden_weights = golden_data[f"{model_type}_version"][f"configuration_{i}"]["weights"]
            differences = self._compare_weights(current_weights, golden_weights, model_type)
            all_differences.extend(differences)

    def test_weight_consistency(self) -> Tuple[bool, List[str]]:
        """Test weight consistency against golden standard."""
        if not self.golden_file.exists():
            print("Golden standard not found, generating...")
            self.generate_golden_standard()

        print("Testing weight consistency...")

        # Load golden standard
        with open(self.golden_file, 'r', encoding='utf-8') as f:
            golden_data = json.load(f)

        all_differences = []

        # Test training version
        self._test_weight_consistency("train", all_differences, golden_data, self.train_configurations)

        # Test inference version
        self._test_weight_consistency("infer", all_differences, golden_data, self.infer_configurations)

        return all_differences

    def _compare_weights(self, current: Dict[str, List[int]], golden: Dict[str, List[int]],
                         version: str) -> List[str]:
        """Compare weight structures."""
        differences = []

        # Check for missing weights
        missing = set(golden.keys()) - set(current.keys())
        if missing:
            differences.append(f"[{version}] Missing weights: {sorted(missing)}")

        # Check for extra weights
        extra = set(current.keys()) - set(golden.keys())
        if extra:
            differences.append(f"[{version}] Extra weights: {sorted(extra)}")

        # Check for shape mismatches
        for name in set(current.keys()) & set(golden.keys()):
            if current[name] != golden[name]:
                differences.append(
                    f"[{version}] Shape mismatch for {name}: "
                    f"current {current[name]} vs golden {golden[name]}"
                )

        return differences


# Pytest test functions
@pytest.fixture
def weight_tester():
    """Create test instance for pytest."""
    return GPTModelWeightConsistencyTest()


@pytest.fixture
def golden_data(weight_tester):
    """Generate or load golden data."""
    if not weight_tester.golden_file.exists():
        weight_tester.generate_golden_standard()

    with open(weight_tester.golden_file, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_golden_standard(weight_tester):
    """
    Feature: Test golden standard
    Description: Verify that the golden standard file exists and contains complete and valid structure
                 including train/infer versions and metadata information
    Expectation: success
    """
    assert weight_tester.golden_file.exists(), "Golden standard file should exist"
    with open(weight_tester.golden_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert "train_version" in data, "Golden standard should contain train version"
    assert "infer_version" in data, "Golden standard should contain infer version"
    assert "metadata" in data, "Golden standard should contain metadata"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_weight_consistency_check(weight_tester):
    """
    Feature: Test the actual weight consistency check
    Description: Verify that the model weights remain consistent across different training and inference
                 runs by comparing current weights with the golden standard reference
    Expectation: success
    """
    differences = weight_tester.test_weight_consistency()
    assert not differences, f"Weight consistency check failed, the differences are: {differences}"
