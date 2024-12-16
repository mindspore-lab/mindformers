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
Test module for testing pipeline function.
How to run this:
pytest tests/st/test_pipeline/test_pipeline.py
"""
import pytest

import mindspore as ms

from mindformers import AutoTokenizer, LlamaForCausalLM
from mindformers.pipeline import TextGenerationPipeline
from mindformers import LlamaConfig

ms.set_context(mode=0)

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_text_generation_pipeline():
    """
    Feature: text_generation_pipeline interface.
    Description: Test basic function of text_generation_pipeline api.
    Expectation: success
    """
    input_data = ["My name is Wolfgang and I live in llama2_7b - Where do I live?"]
    tokenizer = AutoTokenizer.from_pretrained('llama2_7b')
    llama2_config = LlamaConfig(seq_length=128, num_layers=2)
    model = LlamaForCausalLM(llama2_config)
    qa_pipeline = TextGenerationPipeline(task='question_answering',
                                         model=model,
                                         tokenizer=tokenizer)
    output = qa_pipeline(input_data)
    assert "llama2_7b" in output[0]["text_generation_text"][0]
