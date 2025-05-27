"""
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

pytest tests/st/test_ut/test_models/test_recompute.py
"""
import os
import sys

import mindspore
import pytest

from mindformers.models.llama.llama_transformer import LLamaDecodeLayer
from mindformers.models.utils import LayerSetting
from mindformers.modules.transformer.transformer import TransformerOpParallelConfig

mindspore.set_context(device_target='CPU', mode=0)


for path in sys.path:
    if path.endswith('/testcases'):
        new_path = os.path.join(path, 'research')
        if new_path not in sys.path:
            sys.path.append(new_path)


class TestLayerSetting:
    """ test LayerSetting """
    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_recompute(self):
        """ test recompute """
        parallel_config = {
            'pipeline_stage': 2,
            'recompute': {
                'recompute': False,
                'select_recompute': {
                    r'feed_forward\.w1': True,
                    r'feed_forward\.mul': True,
                    r'feed_forward\.w2\.reshape': True
                },
                'parallel_optimizer_comm_recompute': True,
                'mp_comm_recompute': True,
                'recompute_slice_activation': True,
                'select_recompute_exclude': {
                    r'feed_forward\.w1\.activation\.silu': [1]
                },
                'select_comm_recompute_exclude': {
                    r'feed_forward\.mul': [1]
                }
            }
        }

        parallel_config = TransformerOpParallelConfig(**parallel_config)

        layer_setting = LayerSetting(4, 0, parallel_config)
        layer_0 = LLamaDecodeLayer(512, 0)
        layer_3 = LLamaDecodeLayer(512, 3)
        layer_setting(layer_0, 0)
        layer_setting(layer_3, 3)

        # pylint: disable=W0212
        assert layer_0.feed_forward.w1._scope == 'recompute_'                    # test select recompute
        assert layer_0.feed_forward.w1.activation.silu.recompute is False        # test select recompute exclude
        assert layer_3.feed_forward.w1.activation.silu.recompute is not False    # test select recompute exclude
