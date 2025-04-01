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
"""
Test telechat2 predict.
How to run this:
    pytest tests/st/test_model/test_telechat2_model/test_predict.py
"""
import pytest
import numpy as np

import mindspore as ms

from .base_model import get_config, get_model

ms.set_context(mode=0, jit_config={"jit_level": "O0", "infer_boost": "on"})


class TestTelechat2Predict:
    """A test class for testing model prediction."""

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_base_model(self):
        """
        Feature: SLora model predict
        Description: Test llama slora model prediction.
        Expectation: AssertionError
        """
        model_config = get_config()
        model_config.use_past = True
        model_config.run_mode = 'predict'
        model_config.num_layers = 1
        model_config.num_heads = 4
        model_config.n_kv_heads = 2
        model_config.hidden_size = 512
        model_config.intermediate_size = 256
        model_config.block_size = 128
        model_config.num_blocks = 256
        model_config.is_dynamic = True
        model_config.out_proj_has_bias = True
        model_config.batch_size = 2  # set batch size for prediction
        model_config.vocab_size = 32000  # default to use llama2 tokenizer

        model = get_model(model_config)

        input_ids = np.random.randint(0, 128, size=(2, 256), dtype=np.int32)
        _ = model.generate(input_ids=input_ids, max_new_tokens=16)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_moe_model(self):
        """
        Feature: Moe model predict
        Description: Test llama moe model prediction.
        Expectation: AssertionError
        """
        model_config = get_config(is_moe=True)
        model_config.use_past = True
        model_config.run_mode = 'predict'
        model_config.num_layers = 1
        model_config.num_heads = 4
        model_config.n_kv_heads = 2
        model_config.hidden_size = 512
        model_config.intermediate_size = 256
        model_config.block_size = 128
        model_config.num_blocks = 256
        model_config.is_dynamic = True
        model_config.out_proj_has_bias = True
        model_config.batch_size = 2  # set batch size for prediction
        model_config.vocab_size = 32000  # default to use llama2 tokenizer

        model = get_model(model_config, is_moe=True)

        input_ids = np.random.randint(0, 128, size=(2, 256), dtype=np.int32)
        _ = model.generate(input_ids=input_ids, max_new_tokens=16)
