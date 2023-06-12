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
Test Module for QuestionAnsweringPipeline

How to run this:
windows:
pytest .\\tests\\st\\test_pipeline\\test_question_answering_pipeline.py
linux:
pytest ./tests/st/test_pipeline/test_question_answering_pipeline.py
"""
# import pytest

from mindformers.pipeline import QuestionAnsweringPipeline
from mindformers import AutoTokenizer, BertForQuestionAnswering, AutoConfig


# @pytest.mark.level0
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.env_onecard
def test_question_answering_pipeline():
    """
    Feature: QuestionAnsweringPipeline class
    Description: Test the pipeline functions
    Expectation: NotImplementedError, ValueError
    """

    input_data = ["My name is Wolfgang and I live in Berlin - Where do I live?"]


    tokenizer = AutoTokenizer.from_pretrained('qa_bert_base_uncased_squad')
    qa_squad_config = AutoConfig.from_pretrained('qa_bert_base_uncased_squad')

    # This is a known issue, you need to specify batch size equal to 1 when creating bert model.
    qa_squad_config.batch_size = 1

    model = BertForQuestionAnswering(qa_squad_config)
    qa_pipeline = QuestionAnsweringPipeline(task='question_answering',
                                            model=model,
                                            tokenizer=tokenizer)

    results = qa_pipeline(input_data)
    assert isinstance(results, list)
    assert isinstance(results[0], dict)
