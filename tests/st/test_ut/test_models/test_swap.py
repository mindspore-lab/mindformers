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
"""test swap"""

import pytest
import mindspore as ms

from mindformers.parallel_core.training_graph.transformer.transformer_layer import TransformerLayer, \
    TransformerLayerSubmodules
from mindformers.parallel_core.training_graph.transformer.utils import LayerSetting
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.core.context.build_context import build_context
from mindformers.tools.logger import get_logger

logger = get_logger()
logger.propagate = True

build_context({"use_legacy": False})
ms.set_context(device_target='CPU', mode=ms.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_full_layer_swap_logs(caplog):
    """
    Feature: Layer swap logging when recompute=True
    Description: Enable full layer swap with cpu_offloading and recompute=True, apply to layer 0
    Expectation: INFO log 'Set layer swap at layer 0' is recorded
    """
    caplog.set_level('INFO')

    parallel_config = TransformerConfig(
        num_attention_heads=2,
        num_layers=2,
        cpu_offloading=True,
        cpu_offloading_num_layers=[{"layers": True, "backward_prefetch": "Auto"}],
        recompute=True,
        select_recompute={}
    )
    layer_setting = LayerSetting(num_layers=4, offset=0, parallel_config=parallel_config)

    submodules_spec = TransformerLayerSubmodules()

    layer_0 = TransformerLayer(
        config=parallel_config,
        submodules=submodules_spec,
        layer_number=0
    )

    layer_setting(layer_0, 0)

    target = "Set layer swap at layer 0"
    assert any(target in record.message for record in caplog.records), \
        f"Expected log containing '{target}', but got: {[r.message for r in caplog.records]}"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_partial_layer_swap_logs(caplog):
    """
    Feature: Layer swap logging for specific layers with list-based recompute
    Description: Enable layer swap only for layer 1, with recompute=[0, 1]
    Expectation: INFO log 'Set layer swap at layer 1' is recorded
    """
    caplog.set_level('INFO')

    parallel_config = TransformerConfig(
        num_attention_heads=2,
        num_layers=2,
        cpu_offloading=True,
        cpu_offloading_num_layers=[{"layers": [1], "backward_prefetch": 2}],
        recompute=[0, 1],
        select_recompute={}
    )
    layer_setting = LayerSetting(num_layers=4, offset=0, parallel_config=parallel_config)

    submodules_spec = TransformerLayerSubmodules()

    layer_1 = TransformerLayer(
        config=parallel_config,
        submodules=submodules_spec,
        layer_number=1
    )

    layer_setting(layer_1, 1)

    target = "Set layer swap at layer 1"
    assert any(target in record.message for record in caplog.records), \
        f"Expected log containing '{target}', but got: {[r.message for r in caplog.records]}"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_op_level_swap_logs(caplog):
    """
    Feature: Operator-level swap logging via op_swap configuration
    Description: Configure op_swap for 'mlp' on layer 0 with select_recompute enabled
    Expectation: INFO log 'Set op_swap at layer 0: .mlp, value=Auto' is recorded
    """
    caplog.set_level('INFO')

    parallel_config = TransformerConfig(
        num_attention_heads=2,
        num_layers=2,
        cpu_offloading=True,
        cpu_offloading_num_layers=[],
        op_swap=[
            {
                'op_name': r'mlp',
                'layers': [0],
                'backward_prefetch': 'Auto'
            }
        ],
        recompute=False,
        select_recompute={
            "mlp": True,
            "self_attention": [0]
        }
    )
    layer_setting = LayerSetting(num_layers=4, offset=0, parallel_config=parallel_config)

    submodules_spec = TransformerLayerSubmodules()

    layer_0 = TransformerLayer(
        config=parallel_config,
        submodules=submodules_spec,
        layer_number=0
    )

    layer_setting(layer_0, 0)

    target = "Set op_swap at layer 0: .mlp, value=Auto"
    assert any(target in record.message for record in caplog.records), \
        f"Expected log containing '{target}', but got: {[r.message for r in caplog.records]}"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_recompute_op_swap_conflict_logs(caplog):
    """
    Feature: Conflict warning between layer recompute and op-level swap
    Description: Both layer recompute and op_swap are enabled on layer 0
    Expectation: WARNING log containing 'mlp.dense operator in layer 0 that' is recorded
    """
    caplog.set_level('WARNING')

    parallel_config = TransformerConfig(
        num_attention_heads=2,
        num_layers=2,
        cpu_offloading=True,
        cpu_offloading_num_layers=[{"layers": [0], "backward_prefetch": "Auto"}],
        op_swap=[
            {
                'op_name': 'mlp.dense',
                'layers': [0],
                'backward_prefetch': 'Auto'
            }
        ],
        recompute=[0],
        select_recompute={}
    )

    _ = LayerSetting(num_layers=2, offset=0, parallel_config=parallel_config)

    target = "mlp.dense operator in layer 0 that"
    assert any(target in record.message for record in caplog.records), \
        f"Expected warning containing '{target}', but got: {[r.message for r in caplog.records]}"
