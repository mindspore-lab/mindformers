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
"""
Test model_config_utils for
`ignore_and_delete_parameter` and `register_mf_model_parameter` example.
"""

import pytest

from mindformers.parallel_core.mf_model_config import MFModelConfig
from mindformers.models.model_config_utils import (
    NotSupportedInfo,
    ignore_and_delete_parameter,
    register_mf_model_parameter
)
from mindformers.models.configuration_utils import PretrainedConfig


class HuggingFaceModelConfigExample1(PretrainedConfig):
    """HuggingFaceModelConfigExample1"""

    @ignore_and_delete_parameter(extra_ignore_param=[
        ('n_shared_experts', NotSupportedInfo.useless),
        ('num_attention_heads', NotSupportedInfo.useless),
        ('moe_intermediate_size', "This parameter will be converted to `moe_ffn_hidden_size` in TransformerConfig.")
    ])
    def __init__(
            self,
            vocab_size=129280,
            hidden_size=7168,
            intermediate_size=18432,
            moe_intermediate_size=2048,
            num_hidden_layers=61,
            num_nextn_predict_layers=1,
            num_attention_heads=128,
            **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        super().__init__(**kwargs)


class HuggingFaceModelConfigExample2(PretrainedConfig):
    """HuggingFaceModelConfigExample2"""

    @register_mf_model_parameter(mf_model_kwargs=MFModelConfig(
        compute_dtype='fp32',
        layernorm_compute_dtype="float32"
    ))
    def __init__(
            self,
            vocab_size=129280,
            hidden_size=7168,
            intermediate_size=18432,
            moe_intermediate_size=2048,
            num_hidden_layers=61,
            num_nextn_predict_layers=1,
            num_attention_heads=128,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads

        super().__init__(**kwargs)


class HuggingFaceModelConfigExample3(PretrainedConfig):
    """HuggingFaceModelConfigExample3"""

    @ignore_and_delete_parameter(
        extra_ignore_param=[
            ("n_shared_experts", NotSupportedInfo.useless),
            ("num_attention_heads", NotSupportedInfo.useless),
            ("moe_intermediate_size", "this parameter is xxx"),
        ]
    )
    @register_mf_model_parameter(
        mf_model_kwargs=MFModelConfig(
            compute_dtype="bfloat16",
            attention_pre_tokens=2025
        )
    )
    def __init__(
            self,
            vocab_size=129280,
            hidden_size=7168,
            intermediate_size=18432,
            moe_intermediate_size=2048,
            num_hidden_layers=61,
            num_nextn_predict_layers=1,
            num_attention_heads=128,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads

        super().__init__(**kwargs)


class TestModelConfigUtils:
    """A test class for HF ModelConfig utils."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_ignore_parameter_case(self):
        """
        Feature: ignore_and_delete_parameter
        Description: A test function for testing ignore_and_delete_parameter
        Exception: AttributeError
        """
        config = HuggingFaceModelConfigExample1(n_shared_experts=2)
        try:
            ignore_n_shared_experts = config.n_shared_experts
        except AttributeError:
            ignore_n_shared_experts = "useless"

        assert ignore_n_shared_experts == "useless"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_register_parameter_case(self):
        """
        Feature: Register model parameter to MindSpore Transformers
        Description: A test function for testing register_mf_model_parameter
        Exception: ValueError
        """
        config = HuggingFaceModelConfigExample1(
            compute_dtype="float32",
            layernorm_compute_dtype="bf16"
        )
        assert config.compute_dtype == "float32"
        assert config.layernorm_compute_dtype == "bf16"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_ignore_and_register_parameter_case(self):
        """
        Feature: Both of ignore and register parameters
        Description: A test function for testing
            both of ignore_and_delete_parameter and register_mf_model_parameter
        Exception: ValueError, AttributeError
        """
        config = HuggingFaceModelConfigExample3(
            compute_dtype="float32",
            layernorm_compute_dtype="bf16",
            n_shared_experts=2
        )
        try:
            ignore_n_shared_experts = config.n_shared_experts
        except AttributeError:
            ignore_n_shared_experts = "useless"

        assert ignore_n_shared_experts == "useless"
        assert config.compute_dtype == "float32"
        assert config.layernorm_compute_dtype == "bf16"
        assert config.attention_pre_tokens == 2025
