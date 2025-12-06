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
"""Test get op groups info for GPT model."""

from unittest.mock import patch

import mindspore as ms
import pytest

from mindformers import build_context
from mindformers.checkpoint.sharded_tensor import build_sharded_tensor
from mindformers.parallel_core.training_graph.base_models.gpt import gpt_model
from mindformers.parallel_core.training_graph.base_models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec, \
    get_gpt_mtp_block_spec
from mindformers.parallel_core.training_graph.base_models.gpt.gpt_model import GPTModel, \
    compute_repeat_num_and_model_parallel_size, get_op_group_name
from mindformers.parallel_core.transformer_config import TransformerConfig


def build_transformer_config() -> TransformerConfig:
    """Create a minimal transformer config for tensor-parallel unit tests."""
    return TransformerConfig(
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=1,
        # model architecture
        vocab_size=1024,
        position_embedding_type="rope",
        num_attention_heads=2,
        num_layers=2,
        hidden_size=128,
        ffn_hidden_size=512,
        # moe architecture
        num_moe_experts=4,
        first_k_dense_replace=1,
        mtp_num_layers=1,
        add_bias_linear=False,
        moe_grouped_gemm=True
    )


def build_gpt_model():
    """Construct a GPTModel instance with the default test configuration."""
    config = build_transformer_config()
    transformer_layer_spec = get_gpt_decoder_block_spec(config)
    mtp_block_spec = None
    if config.mtp_num_layers is not None:
        mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec)
    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=config.vocab_size,
        max_sequence_length=config.max_position_embeddings,
        position_embedding_type=config.position_embedding_type,
        rotary_percent=1.0,
        rotary_base=config.rotary_base,
        rope_scaling=False,
        mtp_block_spec=mtp_block_spec
    )
    return model


def build_sharded_info(local_shape, axis_fragmentations):
    """Helper to create a simple ShardedTensor descriptor."""
    return build_sharded_tensor(
        param_name="test",
        param_dtype=ms.float32,
        local_shape=local_shape,
        global_shape=local_shape,
        axis_fragmentations=axis_fragmentations,
        global_offset=(0,) * len(local_shape),
    )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gpt_model_sharded_state_dict():
    """
    Feature: GPTModel
    Description: Test the sharded state dict of GPT model.
    Expectation: The sharded state dict has all the trainable parameters and the shape is correct.
    """
    build_context({"use_legacy": False})
    model = build_gpt_model()
    sharded_state_dict = model.sharded_state_dict()

    params = model.trainable_params()
    for param in params:
        assert param.name in sharded_state_dict
        assert param.shape == sharded_state_dict[param.name].global_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    "axis_fragmentations, world_size, pipeline_parallel, opt_group_size, local_shape, expected",
    [
        # case 0: real_op_size == opt_group_size
        ((1, 1), 12, 2, 4, (12, 4), (4, 1)),
        # case 1: real_op_size < opt_group_size
        ((2, 1), 16, 2, 8, (12, 4), (4, 2)),
        # case 2: real_op_size = 1 due to local shape not divisible by real_op_size
        ((4, 1), 32, 2, 4, (10, 4), (1, 4)),
    ],
)
def test_compute_repeat_num_and_model_parallel_size(axis_fragmentations, world_size, pipeline_parallel,
                                                    opt_group_size, local_shape, expected):
    """
    Feature: compute_repeat_num_and_model_parallel_size()
    Description: Test the compute repeat num and model parallel size.
    Expectation: The compute repeat num and model parallel size should be correct.
    """
    sharded_info = build_sharded_info(local_shape, axis_fragmentations)
    assert compute_repeat_num_and_model_parallel_size(
        sharded_info,
        world_size=world_size,
        pp=pipeline_parallel,
        op=opt_group_size,
    ) == expected


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_compute_repeat_num_and_model_parallel_size_multiple_axis_error():
    """
    Feature: compute_repeat_num_and_model_parallel_size()
    Description: Test the error of compute repeat num and model parallel size.
    Expectation: The ValueError should be raised.
    """
    sharded_info = build_sharded_info((8, 8), (2, 2))
    with pytest.raises(ValueError):
        compute_repeat_num_and_model_parallel_size(sharded_info, world_size=16, pp=1, op=2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("mindformers.parallel_core.training_graph.base_models.gpt.gpt_model.create_communication_group")
def test_get_op_group_name_with_mock(mock_create_group):
    """
    Feature: get_op_group_name()
    Description: Test the get op group name with mock.
    Expectation: The get op group name with mock should be correct.
    """
    mock_create_group.return_value = "mock_group"
    gpt_model.OP_GROUP_NAME.clear()

    # case 0: model_parallel_size > 1
    result = get_op_group_name(rank_id=3, real_op_size=2, model_parallel_size=2)
    assert result == ("mock_group", [1, 3])
    mock_create_group.assert_called_once_with([1, 3])

    second_result = get_op_group_name(rank_id=3, real_op_size=2, model_parallel_size=2)
    assert second_result == result
    mock_create_group.assert_called_once()

    # case 1: model_parallel_size = 1
    result = get_op_group_name(rank_id=3, real_op_size=2, model_parallel_size=1)
    assert result == ("mock_group", [2, 3])

    # case 2: model_parallel_size = 4
    result = get_op_group_name(rank_id=3, real_op_size=2, model_parallel_size=4)
    assert result == ("mock_group", [3, 7])
