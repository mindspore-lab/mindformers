# Copyright 2023 Huawei Technologies Co., Ltd
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
TextClassificationPipeline

How to run this:
windows:
pytest .\\tests\\st\\test_pipeline\\test_text_classification_pipeline.py
linux:
pytest ./tests/st/test_pipeline/test_text_classification_pipeline.py
"""
# import pytest

from mindformers.pipeline import TextClassificationPipeline
from mindformers import AutoTokenizer, BertForMultipleChoice, AutoConfig


# @pytest.mark.level0
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.env_onecard
def test_text_classification_pipeline():
    """
    Feature: TextClassificationPipeline class
    Description: Test the pipeline functions
    Expectation: NotImplementedError, ValueError
    """

    input_data = ["The new rights are nice enough-Everyone really likes the newest benefits ",
                  "i don't know um do you do a lot of camping-I know exactly."]

    tokenizer = AutoTokenizer.from_pretrained('txtcls_bert_base_uncased_mnli')
    txtcls_mnli_config = AutoConfig.from_pretrained('txtcls_bert_base_uncased_mnli')

    # Because batch_size parameter is required when bert model is created, and pipeline
    # function deals with samples one by one, the batch_size parameter is seted one.
    txtcls_mnli_config.batch_size = 1

    model = BertForMultipleChoice(txtcls_mnli_config)
    txtcls_pipeline = TextClassificationPipeline(task='text_classification',
                                                 model=model,
                                                 tokenizer=tokenizer,
                                                 max_length=model.config.seq_length,
                                                 padding="max_length")

    results = txtcls_pipeline(input_data, top_k=1)
    assert len(results) == 2
