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
"""Test module for testing FlashAttention in SlidingWindowAttention used for mindformers."""
import pytest
from mindspore import nn, ParameterTuple
from tests.st.test_ut.test_parallel_core.test_training_graph.test_transformer.test_attention.test_sliding_window_attention.data_gen_utils import get_init_params, get_init_tnd_params, get_gpu_datas, get_golden
from tests.utils.double_benchmark import DoubleBenchmarkComparator
from mindformers.parallel_core.training_graph.transformer.flash_attention import FlashAttention
from mindformers.parallel_core.training_graph.device_matrix import layout
from mindformers.parallel_core.transformer_config import MLATransformerConfig


def compare_value(compare_type=None, npu_output=None):
    """Check the accuracy results"""
    gpu_output = GPU_DATA[compare_type]
    golden_output = GOLDEN_DATA[compare_type]
    assert DoubleBenchmarkComparator.check_pass_or_not(npu_output, gpu_output, golden_output), (
        f"FlashAttention compare_type={compare_type} test failed.\n"
        f"NPU output:\n{npu_output}\n\n"
        f"GPU output:\n{gpu_output}\n\n"
        f"Golden output:\n{golden_output}"
    )


class TestFlashAttention:
    """A test class for testing FlashAttention in SlidingWindowAttention"""

    def run_test(self, attention_dropout=0.0, soft_max_scale=None, accuracy=True, compare_type=None):
        """Helper function to run test and check results"""
        self.flash_attention = FlashAttention(config=self.config, layer_number=0, attention_dropout=attention_dropout,
                                              softmax_scale=soft_max_scale)
        if compare_type == "bnsd":
            weights = ParameterTuple(self.flash_attention.trainable_params())
            train_network = nn.ForwardValueAndGrad(self.flash_attention, weights=weights, get_all=True,
                                                   get_by_list=True)
            output, grads = train_network(self.inputs["query"], self.inputs["key"], self.inputs["value"],
                                          self.inputs["attention_mask"])
            npu_output = output.asnumpy()
            query_grad = grads[0][0].asnumpy()
            key_grad = grads[0][1].asnumpy()
            value_grad = grads[0][2].asnumpy()
            if accuracy:
                compare_value(compare_type, npu_output)
                compare_value("query", query_grad)
                compare_value("key", key_grad)
                compare_value("value", value_grad)
        if compare_type == "tnd":
            output = self.flash_attention(**self.inputs)
            npu_output = output.asnumpy()
            if accuracy:
                compare_value(compare_type, npu_output)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_bnsd_case(self):
        """
        Feature: FlashAttention
        Description: Test Case: input_layout=bnsd
        """
        self.config = MLATransformerConfig(multi_latent_attention=False,
                                           hidden_size=4,
                                           num_attention_heads=2,
                                           num_layers=1,
                                           window_size=(10,0),
                                           model_architecture="yoco"
                                           )

        layout.init_layout(self.config)
        self.inputs = get_init_params(self.config)
        self.run_test(compare_type="bnsd")

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_tnd_case(self):
        """
        Feature: FlashAttention
        Description: Test Case: input_layout=tnd
        """
        self.config = MLATransformerConfig(multi_latent_attention=False,
                                           hidden_size=4,
                                           num_attention_heads=2,
                                           num_layers=1,
                                           window_size=(10,0),
                                           use_eod_attn_mask_compression=True,
                                           model_architecture="yoco"
                                           )

        layout.init_layout(self.config)
        self.inputs = get_init_tnd_params(self.config)
        self.run_test(compare_type="tnd")

GOLDEN_DATA = get_golden()
GPU_DATA = get_gpu_datas()
