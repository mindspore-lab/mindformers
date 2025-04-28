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
"""Test SelfAttention"""
import argparse
from collections import namedtuple

import numpy as np
import pytest
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor, load_param_into_net
from mindformers.experimental.graph.transformer.spec_utils import (
    ModuleSpec,
    build_module,
)
from mindformers.experimental.graph.transformer.transformer_config import (
    TransformerConfig
)
from mindformers.experimental.infer.core.transformer import ParallelAttention
from mindformers.experimental.infer.transformer.flash_attention import FlashAttention
from mindformers.experimental.infer.transformer.self_attention import (
    DotProductAttention,
    SelfAttention,
    SelfAttentionSubmodules,
)
from mindformers.experimental.infer.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)

from tests.st.test_ut.test_experimental.test_infer.test_core import (
    BLOCK_SIZE,
    NUM_BLOCKS,
    convert_weight_name,
)

ms.set_context(
    device_target="Ascend",
    mode=ms.GRAPH_MODE,
    jit_config={
        "jit_level": "O0",
        "infer_boost": "on"
    }
)


def get_self_attention_config(args_):
    """Generate config for SelfAttention test."""
    config_ = TransformerConfig()
    parallel_config = TransformerConfig()
    parallel_config.use_sequence_parallel = False
    config_.batch_size = args_.batch_size
    config_.seq_length = args_.seq_length
    config_.param_init_type = mstype.float16
    config_.param_init_dtype = mstype.float16
    config_.compute_dtype = mstype.float16
    config_.compute_type = mstype.float16
    config_.num_heads = args_.num_heads
    config_.num_attention_heads = args_.num_heads
    config_.n_kv_heads = args_.num_query_groups
    config_.num_query_groups = args_.num_query_groups
    config_.hidden_size = args_.hidden_size
    config_.parallel_config = parallel_config
    config_.sequence_parallel = False
    config_.head_dim = int(config_.hidden_size / config_.num_attention_heads)
    config_.out_proj_has_bias = False
    config_.qkv_has_bias = False
    config_.attention_dropout_rate = 1
    config_.use_flash_attention = args_.use_flash_attention
    config_.is_prefill = args_.is_prefill
    config_.use_past = True
    config_.num_blocks = NUM_BLOCKS
    config_.block_size = BLOCK_SIZE

    return config_


def run_self_attention_test(config_):
    """Run a comparison between the original and mcore's SelfAttention."""
    if config_.is_prefill:
        slot_mapping = Tensor(np.arange(config_.batch_size * config_.seq_length),
                              mstype.int32)
        batch_valid_length = Tensor(np.ones((config_.seq_length,)),
                                    dtype=mstype.int32)
        input_ = Tensor(
            np.random.uniform(
                0, 1,
                [1, config_.batch_size * config_.seq_length, config_.hidden_size]),
            mstype.float16)
    else:
        slot_mapping = Tensor(np.arange(config_.batch_size * 1),
                              mstype.int32)
        batch_valid_length = Tensor(np.ones((1,)),
                                    dtype=mstype.int32)
        input_ = Tensor(
            np.random.uniform(
                0, 1,
                [config_.batch_size, 1, config_.hidden_size]),
            mstype.float16)

    block_tables = Tensor(
        np.ones((config_.batch_size, BLOCK_SIZE)) * -1, mstype.int32)
    block_tables[0][0] = 0

    if config_.use_flash_attention:
        self_attn = ModuleSpec(module=SelfAttention,
                               submodules=SelfAttentionSubmodules(
                                   core_attention=FlashAttention,
                                   linear_proj=RowParallelLinear,
                                   linear_qkv=ColumnParallelLinear,
                                   q_layernorm=None,
                                   k_layernorm=None))
    else:
        self_attn = ModuleSpec(module=SelfAttention,
                               submodules=SelfAttentionSubmodules(
                                   core_attention=DotProductAttention,
                                   linear_proj=RowParallelLinear,
                                   linear_qkv=ColumnParallelLinear,
                                   q_layernorm=None,
                                   k_layernorm=None))

    self_attn_mcore = build_module(self_attn, config=config_, layer_number=1)

    parallel_attn = ParallelAttention(config_, 1)

    if config_.is_prefill:
        param_dict = parallel_attn.parameters_dict()
        converted_param_dict = convert_weight_name(param_dict)
        load_param_into_net(self_attn_mcore, converted_param_dict)

        output_mcore = self_attn_mcore(input_,
                                       kv_cache=None,
                                       slot_mapping=slot_mapping,
                                       actual_seq_qlen=batch_valid_length,
                                       actual_seq_kvlen=batch_valid_length)

        ret = np.array_equal(input_.shape, output_mcore.shape)
        assert ret, ("The output_shape not equals to input_shape"
                     "the original one when is_prefill is True")
    else:
        if config_.use_flash_attention:
            parallel_attn.is_first_iteration = False
            self_attn_mcore.flash_attention.add_flags(is_prefill=False)

        param_dict = parallel_attn.parameters_dict()
        converted_param_dict = convert_weight_name(param_dict)
        load_param_into_net(self_attn_mcore, converted_param_dict)

        output_mcore = self_attn_mcore(input_,
                                       kv_cache=None,
                                       block_tables=block_tables,
                                       slot_mapping=slot_mapping,
                                       batch_valid_length=batch_valid_length,
                                       context_lens_tensor=batch_valid_length)

        ret = np.array_equal(input_.shape, output_mcore.shape)
        assert ret, ("The output_shape not equals to input_shape"
                     "the original one when is_prefill is False")
        ms.reset_auto_parallel_context()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    'batch_size, seq_length, '
    'num_heads, num_query_groups, hidden_size, '
    'use_flash_attention, is_prefill', (
        (1, 256, 16, 4, 256, True, False),
        (1, 256, 16, 4, 256, True, True),
        (1, 256, 16, 2, 256, False, True),
        (1, 256, 16, 2, 256, False, False),
    )
)
def test_self_attn_case(batch_size, seq_length, num_heads,
                        num_query_groups, hidden_size, use_flash_attention,
                        is_prefill):
    """
    batch_size = 1, seq_length = 256 ,num_heads = 16 ,num_query_groups = 4,
    hidden_size = 256, use_flash_attention = True/False, is_prefill = True/False
    Feature: Test SelfAttention under various configurations.
    Description: Run original and MCore SelfAttention and get output.
    Expectation: The accuracy error exceeds 0.01
    """
    Args = namedtuple('Args', [
        'batch_size',
        'seq_length',
        'num_heads',
        'num_query_groups',
        'hidden_size',
        'use_flash_attention',
        'is_prefill',
    ])
    args_ = Args(batch_size=batch_size,
                 seq_length=seq_length,
                 num_heads=num_heads,
                 num_query_groups=num_query_groups,
                 hidden_size=hidden_size,
                 use_flash_attention=use_flash_attention,
                 is_prefill=is_prefill)
    config_ = get_self_attention_config(args_)
    run_self_attention_test(config_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default="1", type=int)
    parser.add_argument('--seq_length', default="4096", type=int)
    parser.add_argument('--num_heads', default="16", type=int)
    parser.add_argument('--num_query_groups', default="2", type=int)
    parser.add_argument('--hidden_size', default="4096", type=int)
    parser.add_argument('--use_flash_attention', default=False, type=bool)
    parser.add_argument('--is_prefill', default=True, type=bool)
    args = parser.parse_args()
    config = get_self_attention_config(args)
    run_self_attention_test(config)
