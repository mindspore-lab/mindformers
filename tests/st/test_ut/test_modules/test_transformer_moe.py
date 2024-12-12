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
"""The unit test of mindformers.modules.transformer.moe.py."""
from unittest import mock

import pytest
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.ops import operations as P

from mindformers.modules.transformer import TransformerOpParallelConfig
from mindformers.modules.transformer.moe import (
    MoE,
    MoEConfig,
    MoEInfer,
    MoEV2,
    Router,
    TopkRouter,
    TopkRouterV2,
    default_moe_config,
)
from mindformers.modules.transformer.op_parallel_config import (
    default_moeparallel_config,
)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    "hidden_size, ffn_hidden_size, dropout_rate, hidden_act, moe_config, parallel_config",
    [(128, 256, 0.1, 'gelu', default_moe_config, default_moeparallel_config),
     (
         128, 256, 0.1, 'gelu',
         MoEConfig(
             enable_cold_hot_expert=True, expert_num=2, hot_expert_num=1
         ), default_moeparallel_config
     )]
)
def test_moe_init(
        hidden_size, ffn_hidden_size, dropout_rate, hidden_act, moe_config,
        parallel_config
):
    """
    Feature: MoE
    Description: Test init function of Mot.
    Expectation: No Exception
    """
    moe = MoE(
        hidden_size,
        ffn_hidden_size,
        dropout_rate,
        hidden_act,
        moe_config=moe_config,
        parallel_config=parallel_config
    )
    assert moe.hidden_size == hidden_size
    assert moe.router.d_model == hidden_size
    assert moe.router.moe_config == moe_config
    assert moe.router.routing_policy == "TopkRouterV1"


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@mock.patch('mindformers.modules.transformer.moe._get_parallel_mode')
@mock.patch('mindformers.modules.transformer.moe._is_sharding_propagation')
def test_moe_init_parallel(mock_sharding_propagation, mock_parallel_mode):
    """
    Feature: MoE
    Description: Test init function of Mot.
    Expectation: No Exception
    """
    mock_sharding_propagation.return_value = True
    mock_parallel_mode.return_value = ParallelMode.AUTO_PARALLEL
    moe = MoE(
        128,
        256,
        0.1,
        'gelu',
        moe_config=default_moe_config,
        parallel_config=default_moeparallel_config
    )
    assert moe.hidden_size == 128
    mock_parallel_mode.assert_called()
    mock_sharding_propagation.assert_called()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_moev2_init():
    """
    Feature: MoEV2
    Description: Test init function of MoEV2.
    Expectation: No Exception
    """
    moe_v2 = MoEV2(
        ffn=None,
        dim=128,
        moe_config=MoEConfig(routing_policy='TopkRouterV2'),
        parallel_config=default_moeparallel_config,
        return_extra_loss=True
    )

    assert moe_v2.hidden_size == 128
    assert moe_v2.expert_dim is not None
    assert moe_v2.return_extra_loss
    assert moe_v2.capacity_factor is not None
    assert moe_v2.num_experts_chosen is not None
    assert moe_v2.dp_group is not None
    assert moe_v2.dp is not None
    assert moe_v2.ep is not None
    assert moe_v2.mp is not None
    assert moe_v2.group_wise_a2a is not None
    assert isinstance(moe_v2.add_loss, P.Add)
    assert moe_v2.dp_moe is not None
    assert isinstance(moe_v2.dp_range, Tensor)
    assert moe_v2.ffn is None
    assert isinstance(moe_v2.reshape, P.Reshape)
    assert isinstance(moe_v2.shape, P.Shape)
    assert isinstance(moe_v2.cast, P.Cast)
    assert isinstance(moe_v2.transpose_4dim_dp1, P.Transpose)
    assert isinstance(moe_v2.transpose_4dim_dp0, P.Transpose)
    assert isinstance(moe_v2.transpose_5dim_ep2, P.Transpose)
    assert isinstance(moe_v2.concat_dp, P.Concat)
    assert isinstance(moe_v2.stride_slice, P.StridedSlice)
    assert isinstance(moe_v2.stride_slice_dp, P.StridedSlice)
    assert isinstance(moe_v2.stride_slice_ep, P.StridedSlice)
    assert isinstance(moe_v2.stride_slice_dp_mp, P.StridedSlice)
    assert isinstance(moe_v2.stride_slice_ep_mp, P.StridedSlice)
    assert isinstance(moe_v2.stride_slice_outer_dp_mp, P.StridedSlice)
    assert isinstance(moe_v2.stride_slice_outer_ep_mp, P.StridedSlice)
    assert isinstance(moe_v2.stride_slice_outer_ep, P.StridedSlice)
    assert isinstance(moe_v2.stride_slice_outer_dp, P.StridedSlice)
    assert isinstance(moe_v2.transpose_5dim_ep1, P.Transpose)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    "routing_policy", ["TopkRouterV1", "TopkRouterV2", None]
)
def test_router_init(routing_policy):
    """
    Feature: Router
    Description: Test init function of Router.
    Expectation: No Exception
    """
    # Create a Router instance with the specified routing policy
    router = Router(
        d_model=128,
        moe_config=default_moe_config,
        routing_policy=routing_policy,
        training=True,
        parallel_config=TransformerOpParallelConfig()
    )

    # Assert that the attributes are set correctly
    assert router.d_model == 128
    assert router.moe_config == default_moe_config
    assert router.training is True

    # Check the type of the router based on the routing policy
    if routing_policy == "TopkRouterV1":
        assert isinstance(router.router, TopkRouter)
    elif routing_policy == "TopkRouterV2":
        assert isinstance(router.router, TopkRouterV2)
    elif routing_policy is None:
        assert router.router is None


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@mock.patch('mindformers.modules.transformer.moe._get_parallel_mode')
@mock.patch('mindformers.modules.transformer.moe._is_sharding_propagation')
def test_topk_router_init(mock_sharding_propagation, mock_parallel_mode):
    """
    Feature: TopkRouter
    Description: Test init function of TopkRouter.
    Expectation: No Exception
    """
    # Set the necessary parameters
    mock_sharding_propagation.return_value = True
    mock_parallel_mode.return_value = ParallelMode.AUTO_PARALLEL
    d_model = 128
    training = True
    moe_config = default_moe_config

    # Create the TopkRouter instance
    router = TopkRouter(
        d_model, moe_config, training, TransformerOpParallelConfig()
    )

    # Assert the properties and operations
    assert router.d_model == d_model
    assert router.expert_dim == moe_config.expert_num
    assert router.capacity_factor == moe_config.capacity_factor
    assert router.training == training


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_moe_infer_init():
    """
    Feature: MoEInfer
    Description: Test init function of MoEInfer.
    Expectation: No Exception
    """
    moe_infer = MoEInfer(
        None, 128, default_moe_config, TransformerOpParallelConfig()
    )

    # Check if the attributes are set correctly
    assert moe_infer.hidden_size == 128
    assert moe_infer.expert_dim == default_moe_config.expert_num
    assert moe_infer.topk_norm_prob == default_moe_config.norm_topk_prob
    assert moe_infer.num_experts_chosen == default_moe_config.num_experts_chosen
