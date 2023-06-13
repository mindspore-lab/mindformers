# Copyright 2022 Huawei Technologies Co., Ltd
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
Test Module for classification function of
ZeroShotImageClassificationPipeline

How to run this:
windows:
pytest .\\tests\\st\\test_pipeline\\test_translation_pipeline.py
linux:
pytest ./tests/st/test_pipeline/test_translation_pipeline.py

Note:
    pipeline also supports a dataset input
"""
import os
import shutil

# import pytest
from mindformers.pipeline import TranslationPipeline

from mindformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, T5Processor
from mindspore.dataset import GeneratorDataset


def modify_batch_size(net, batch_size):
    """Change the batch size of the net"""
    if hasattr(net, 'batch_size'):
        net.batch_size = batch_size
    for cell in net.cells():
        modify_batch_size(cell, batch_size)


# @pytest.mark.level0
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.env_onecard
def test_translation_pipeline():
    """
    Feature: Test translation pipeline class
    Description: Test the pipeline functions
    Expectation: No errors
    """
    output_path = 'test_translation_pipeline_outer_path'
    os.makedirs(output_path, exist_ok=True)
    tokenizer = T5Tokenizer.from_pretrained(os.path.join(os.path.dirname(__file__), '../test_model/test_t5_model'))
    tokenizer.save_pretrained(output_path)

    model = T5ForConditionalGeneration(T5Config(num_layers=1, hidden_dropout_rate=0.0,
                                                attention_dropout_rate=0.0,
                                                hidden_size=512,
                                                num_heads=8,
                                                vocab_size=100,
                                                batch_size=1, seq_length=32,
                                                max_decode_length=8))
    model.save_pretrained(output_path)
    model = T5ForConditionalGeneration.from_pretrained(output_path)

    processor = T5Processor(tokenizer=tokenizer)
    processor.save_pretrained(output_path)

    translator = TranslationPipeline(model=output_path)
    output = translator("abc")
    assert len(output) == 1

    output = translator(["abc", "bc"])

    assert len(output) == 2

    dataset_input = GeneratorDataset(["abc" for i in range(3)], column_names=["text"])
    output = translator(dataset_input)
    assert len(output) == 3

    # test case with batch size.
    dataset_input = GeneratorDataset(["abc" for i in range(3)], column_names=["text"])
    modify_batch_size(translator.model, batch_size=3)
    translator(dataset_input, batch_size=3)
    shutil.rmtree(output_path, ignore_errors=True)
