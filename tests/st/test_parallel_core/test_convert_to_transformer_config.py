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
"""Test Convert To TransformerConfig Function"""

import pytest

from mindformers.parallel_core.transformer_config_utils import convert_to_transformer_config
from mindformers.parallel_core.transformer_config import TransformerConfig, MLATransformerConfig

class DummyConfig(dict):
    """A dummy config that behaves like a dict and supports attribute access."""

    def __getattr__(self, item):
        return self[item]

    def __contains__(self, item):
        return dict.__contains__(self, item)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_normal_execution_case():
    """
    Feature: Test case for verifying normal execution.
    Description: Input the parameters that do not need to be converted normally.
    Expectation: TransformerConfig can be instantiated correctly.
    """
    config = DummyConfig({
        'num_layers': 2,
        'hidden_size': 128,
        'num_heads': 8,
        'parallel_config': {
            'pipeline_stage': 1,
            'model_parallel': 2
        },
        'context_parallel': 1,
    })
    result = convert_to_transformer_config(config)
    assert isinstance(result, TransformerConfig)
    assert result.num_layers == 2
    assert result.hidden_size == 128
    assert result.num_attention_heads == 8
    assert result.tensor_model_parallel_size == 2
    assert result.pipeline_model_parallel_size == 1
    assert result.context_parallel_size == 1


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_passed_in_additional_map_case():
    """
    Feature: Test additional map case.
    Description: Input the additional_map parameter for coverage verification.
    Expectation: Convert map will be updated, and the TransformerConfig can be instantiated correctly.
    """
    config = DummyConfig({
        'num_layers': 2,
        'hidden_size': 128,
        'num_heads': 8,
        'pipeline_stage': 1,
        'context_parallel': 1,
        'foo': 16
    })
    additional_map = {'foo': 'data_parallel_size', 'num_heads': 'num_attention_heads'}
    result = convert_to_transformer_config(config, additional_map=additional_map)
    assert result.data_parallel_size == 16
    assert result.num_attention_heads == 8


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_mla_model_case():
    """
    Feature: Test MLA Model case
    Description: Input the is_mla_model parameter to instance a MLATransformerConfig
    Expectation: MLATransformerConfig can be instantiated correctly.
    """
    config = DummyConfig({
        'num_layers': 2,
        'hidden_size': 128,
        'num_heads': 8,
        'pipeline_stage': 1,
        'context_parallel': 1,
        'multi_latent_attention': True
    })
    result = convert_to_transformer_config(config, is_mla_model=True)
    assert isinstance(result, MLATransformerConfig)
    assert result.multi_latent_attention is True


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_passed_in_none_model_config_case():
    """
    Feature: Test none model config case.
    Description: Input None config to raise ValueError.
    Expectation: Capture the raised ValueError.
    """
    with pytest.raises(ValueError):
        convert_to_transformer_config(None)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_trans_func_case():
    """
    Feature: Test the function `trans_func` can run normally.
    Description: Input config contains the key that will trigger `trans_func`,
        such as 'residual_dtype', 'softmax_compute_dtype', 'first_k_dense_replace', 'use_gating_sigmoid'.
    Expectation: `trans_func` can convert the mapping of special keys, and instantiate TransformerConfig successfully.
    """
    config = DummyConfig({
        'residual_dtype': 'fp32',
        'softmax_compute_dtype': 'float16',
        'moe_config': {
            'first_k_dense_replace': 2,
            'use_gating_sigmoid': True
        },
        'num_layers': 2,
        'hidden_size': 128,
        'num_heads': 8,
        'pipeline_stage': 1,
        'context_parallel': 1,
    })
    result = convert_to_transformer_config(config)
    used_parameter = True
    unused_parameter = False
    assert result.num_layers == 2
    assert result.hidden_size == 128
    assert result.num_attention_heads == 8
    assert result.pipeline_model_parallel_size == 1
    assert result.context_parallel_size == 1
    assert result.fp32_residual_connection == used_parameter
    assert result.attention_softmax_in_fp32 == unused_parameter
    assert result.moe_layer_freq == 2
    assert result.moe_router_score_function == "sigmoid"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_not_exist_key_in_mapping_case():
    """
    Feature: Test not exist keys in mapping case.
    Description: Input config has key that do not need to be converted.
    Expectation: Capture the raised ValueError.
    """
    config = DummyConfig({
        'num_layers': 2,
        'hidden_size': 128,
        'num_heads': 8,
        'model_parallel': 2,
        'pipeline_stage': 1,
        'context_parallel': 1,
        'not_exist_key': 999
    })
    with pytest.raises(ValueError):
        convert_to_transformer_config(config)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_invalid_first_k_dense_replace_case():
    """
    Feature: Test invalid type of first_k_dense_replace case.
    Description: Input first_k_dense_replace is a string, not int.
    Expectation: Capture the raised ValueError.
    """
    config = DummyConfig({
        'moe_config': {
            'first_k_dense_replace': 'not_int',
        },
        'num_layers': 2,
        'hidden_size': 128,
        'num_heads': 8,
        'pipeline_stage': 1,
        'context_parallel': 1,
    })
    with pytest.raises(ValueError):
        convert_to_transformer_config(config)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_empty_str_to_convert_str_to_mstype_case():
    """
    Feature: Test function convert_str_to_mstype.
    Description: Input str is empty.
    Expectation: Capture the raised ValueError.
    """
    from mindformers.parallel_core.model_parallel_config import convert_str_to_mstype
    with pytest.raises(ValueError):
        convert_str_to_mstype('')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_passed_in_dtype_case():
    """
    Feature: Test dtype passed in convert_str_to_mstype.
    Description: Input a mindspore dtype, and a numpy dtype.
    Expectation: No interception for mindspore dtype, but for numpy dtype.
    """
    from mindformers.parallel_core.model_parallel_config import convert_str_to_mstype
    from mindspore import dtype as mstype
    result = convert_str_to_mstype(mstype.float32)
    assert result == mstype.float32

    import numpy as np
    with pytest.raises(TypeError):
        convert_str_to_mstype(np.float32)
