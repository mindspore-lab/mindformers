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
TestFillMaskBertPipeline

How to run this:
windows:
pytest .\\tests\\st\\test_model\\test_bert_model
\\test_fill_mask_bert_pipeline.py
linux:
pytest ./tests/st/test_model/test_bert_model
/test_fill_mask_bert_pipeline.py

Note:
    pipeline also supports a dataset input
"""
import os

# import pytest
from mindformers.pipeline import FillMaskPipeline, pipeline

from mindformers import BertTokenizer, BertProcessor
from mindformers.models import BertForPreTraining, BertConfig


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
    output_path = 'bert_out_path'
    os.makedirs(output_path, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained("bert_base_uncased")
    tokenizer.save_pretrained(output_path)
    bert = BertForPreTraining(BertConfig(num_hidden_layers=1, hidden_dropout_prob=0.0,
                                         attention_probs_dropout_prob=0.0,
                                         batch_size=1, seq_length=16, is_training=False))
    bert.save_pretrained(output_path)
    processor = BertProcessor(tokenizer=tokenizer, max_length=16, padding="max_length")
    processor.save_pretrained(output_path)
    output = processor("Paris is a city.")
    assert output['text'].shape == (1, 16)
    fillmask = FillMaskPipeline(model=output_path, max_length=16, padding="max_length")
    output = fillmask("Hello I'm a [MASK] model.")
    assert len(output) == 1
    task = pipeline(task='fill_mask',
                    model=bert,
                    tokenizer=tokenizer,
                    max_length=16,
                    padding='max_length')
    output = task("Hello I'm a [MASK] model. That is [MASK] .")
    assert len(output) == 1 and len(output[0]) == 3
    output = task(["Hello I'm a [MASK] model.", "Paris is the [MASK] of France"])
    assert len(output) == 2
