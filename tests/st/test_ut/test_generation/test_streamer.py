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
"""Test stremer inference"""
import os
import pytest

from transformers import AutoTokenizer

from mindspore.nn.utils import no_init_parameters

from mindformers import AutoModel, build_context, MindFormerConfig
from mindformers import pipeline, TextStreamer, TextIteratorStreamer


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_streamer():
    """
    Feature: Streamer inference.
    Description: Test streamer inference.
    Expectation: Success.
    """
    config_path = os.path.join(os.path.dirname(__file__), "qwen3_0_6b_infer.yaml")
    config = MindFormerConfig(config_path)
    config.use_parallel = False
    config.parallel_config.model_parallel = 1
    build_context(config)

    inputs = ["I love Beijing, because", "请介绍北京", "生成以换行符结尾的句子"]

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_dir, trust_remote_code=True)

    with no_init_parameters():
        network = AutoModel.from_config(config)
        network.load_weights(config.pretrained_model_dir)

    streamer = TextStreamer(tokenizer)
    text_generation_pipeline = pipeline(task="text_generation", model=network, tokenizer=tokenizer, streamer=streamer)
    _ = text_generation_pipeline(inputs, max_length=64, do_sample=False, top_k=3, top_p=1)

    streamer = TextIteratorStreamer(tokenizer)
    text_generation_pipeline = pipeline(task="text_generation", model=network, tokenizer=tokenizer, streamer=streamer)
    _ = text_generation_pipeline(inputs, max_length=64, do_sample=False, top_k=3, top_p=1)
