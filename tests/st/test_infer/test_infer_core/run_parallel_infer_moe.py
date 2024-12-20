# Copyright 2024 Huawei Technologies Co., Ltd
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
"""test parallel MoE."""

import argparse
import numpy as np

from mindspore.common import dtype as mstype
import mindspore.ops.operations as P
from mindspore import nn, Tensor, set_context, set_seed
from mindspore.communication import init

from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from mindformers import LlamaConfig
from mindformers.modules.transformer.moe import MoEConfig
from mindformers.experimental.infer.models.llama.utils import convert_model_config
from mindformers.experimental.infer.core.moe import ParallelMoE
from mindformers.experimental.infer.core import get_act_func
from mindformers.experimental.infer.core.layers import ColumnParallelLinear, RowParallelLinear


class MoEParallelMLP(nn.Cell):
    r"""
    FeedForward for MoE Infer implemented with grouped matmul.

    .. math::
            (xW_1 * xW_3)W_2

        Inputs:
            - **x** (Tensor) - should be `[batch, seq_length, hidden_size] or [batch * seq_length, hidden_size]`.
              Float tensor.

        Outputs:
            Tensor, the output of this layer after mapping. The shape is `[batch, seq_length, hidden_size] or
            [batch * seq_length, hidden_size]`.

        Raises:
            ValueError: `hidden_dim` is not a multiple of the model parallel way.
            ValueError: `dim` is not a multiple of the model parallel way.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.ffn_hidden_size = self.config.ffn_hidden_size
        self.mul = P.Mul()
        self.cast = P.Cast()
        self.act_type = self.config.hidden_act
        self.act_func = get_act_func(self.act_type)

        self.w1 = ColumnParallelLinear(
            self.hidden_size,
            self.ffn_hidden_size,
            self.config.parallel_config,
            is_expert=True,
            param_init_type=self.config.param_init_dtype,
            expert_num=self.config.moe_config.expert_num,
        )

        self.w2 = RowParallelLinear(
            self.ffn_hidden_size,
            self.hidden_size,
            input_is_parallel=True,
            config=self.config.parallel_config,
            is_expert=True,
            bias=False,
            param_init_type=self.config.param_init_dtype,
            expert_num=self.config.moe_config.expert_num,
        )
        self.w3 = ColumnParallelLinear(
            self.hidden_size,
            self.ffn_hidden_size,
            self.config.parallel_config,
            is_expert=True,
            param_init_type=self.config.param_init_dtype,
            expert_num=self.config.moe_config.expert_num,
        )

    def construct(self, x, group_list=None):
        """Forward process of the FeedForward"""
        x = self.cast(x, self.config.compute_dtype)
        # [bs, seq, hidden_dim] or [bs * seq, hidden_dim]
        gate = self.w1(x, group_list=group_list)  # dp,1 -> dp, mp
        gate = self.act_func(gate)
        hidden = self.w3(x, group_list=group_list)  # dp,1 -> dp, mp
        hidden = self.mul(hidden, gate)  # dp,mp -> dp, mp
        output = self.w2(hidden, group_list=group_list)  # dp,mp -> dp, 1
        return output


class MoENet(nn.Cell):
    """testcase of MoE"""

    def __init__(self, ffn, hidden_size, moe_config, use_fused_op):
        super().__init__()
        self.model = ParallelMoE(ffn, hidden_size, moe_config, use_fused_op)

    def construct(self, input_tensor):
        output = self.model(input_tensor)
        return output


def get_config():
    """get config of testcase"""
    base_config = LlamaConfig(
        param_init_dtype=mstype.float16,
        compute_dtype=mstype.float16,
        use_past=True,
        qkv_concat=True,
        num_heads=16,
        hidden_size=256,
        use_flash_attention=True,
        qkv_has_bias=False,
        rotary_dtype=mstype.float16,
        num_blocks=16,
        block_size=256,
        out_proj_has_bias=False,
        vocab_size=1000,
        num_layers=2,
        seq_length=128,
        mlp_has_gate=True,
        ffn_concat=True,
        intermediate_size=4096,
        batch_size=2,
    )
    base_config = convert_model_config(base_config)
    base_config.moe_config = MoEConfig(
        expert_num=4,
        num_experts_chosen=2,
        norm_topk_prob=True,
        router_dense_type="float32"
    )

    return base_config


def _test_parallel_moe():
    """
    Test case for the ParallelMoE.

    This function initializes various input tensors based on the configuration,
    passes them through the `net` (ParallelMoE), and asserts that
    the output tensor has the expected shape [batch_size, seq_length, hidden_size].
    """
    base_config = get_config()
    ffn = MoEParallelMLP(base_config)
    hidden_size = base_config.hidden_size
    moe_config = base_config.moe_config
    batch_size = base_config.batch_size
    seq_length = base_config.seq_length
    use_fused_op = True
    # test module
    net = MoENet(ffn, hidden_size, moe_config, use_fused_op)

    tokens = Tensor(np.ones((batch_size, seq_length, hidden_size)), dtype=mstype.float16)
    output = net(tokens)

    assert output.shape == (batch_size, seq_length, hidden_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=0, help='test mode of parallel transformer')

    args = parser.parse_args()

    jit_level = "O0"
    infer_boost = "on"
    set_context(mode=args.mode, jit_config={"jit_level": jit_level, "infer_boost": infer_boost})
    set_seed(1234)
    # init communication
    init()
    initialize_model_parallel(tensor_model_parallel_size=2)

    _test_parallel_moe()
