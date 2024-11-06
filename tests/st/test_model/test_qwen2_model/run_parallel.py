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
Test module for testing the paralleled qwen2 interface used for mindformers.
How to run this:
    pytest tests/st/test_model/test_qwen2_model/test_parallel.py
"""
import argparse
import mindspore as ms

from mindformers import build_context

from tests.st.test_model.test_llama2_model.base_model import get_config, get_model
from tests.utils.model_tester import ModelTester


def parallel_predict_mp2():
    """test qwen2 predict in model_parallel=2."""
    # dp=1, mp=4, pp=1
    parallel_config = {
        'use_parallel': True,
        'model_parallel': 2,
        'use_seq_parallel': False
    }
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    runner = ModelTester(run_mode='predict', batch_size=8, experiment_mode=False, **parallel_config)
    build_context(runner.args)

    model_config = get_config()
    model_config.parallel_config = runner.args.get_parallel_config()
    model_config.batch_size = runner.batch_size  # set batch size for prediction
    model_config.use_past = True
    model_config.is_dynamic = True
    model_config.use_flash_attention = True
    model_config.qkv_concat = True
    model_config.qkv_has_bias = True
    model_config.num_heads = 40
    model_config.hidden_size = 5120
    model_config.intermediate_size = 13696

    model = get_model(model_config)

    outputs = r"hello world.azione Berliner BerlinerfeedJsJsJsJsJsJsnotinnotinnotinnotin McK McK"
    runner.set_predict(model=model, expect_outputs=outputs)


TEST_MAP = {
    'parallel_predict_mp2': parallel_predict_mp2,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='test mode of qwen2 model.')

    args = parser.parse_args()
    TEST_MAP[args.mode]()
