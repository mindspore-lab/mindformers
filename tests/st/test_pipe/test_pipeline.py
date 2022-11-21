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
Test module for testing the pipeline
How to run this:
1.add proper eval data path
2.pytest tests/test_pipeline.py
"""
import pytest
import mindspore
from mindspore import context
from mindtransformer import pipeline

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pipeline_qa_task():
    """
    Feature: The pipeline test for question_answering
    Description: Using pipeline interface to test question_answering task
    Expectation: The returned ret is not 0.
    """
    context.set_context(mode=mindspore.GRAPH_MODE, device_target="GPU")
    pipe = pipeline("question_answering")
    vocab_file = "/home/workspace/mindtransformer/pipe/vocab.txt"
    eval_data = "/home/workspace/mindtransformer/pipe/eval.json"
    print(pipe(vocab_file_path=vocab_file, eval_data_path=eval_data))

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pipeline_text_classification_task():
    """
    Feature: The pipeline test for text_classification
    Description: Using pipeline interface to test text_classification task
    Expectation: The returned ret is not 0.
    """
    context.set_context(mode=mindspore.GRAPH_MODE, device_target="GPU")
    pipe = pipeline("text_classification")
    print(pipe(eval_data_path="/home/workspace/mindtransformer/pipe/eval.tf_record"))

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pipeline_language_modeling_task():
    """
    Feature: The pipeline test for language_modeling
    Description: Using pipeline interface to test language_modeling task
    Expectation: The returned ret is not 0.
    """
    context.set_context(mode=mindspore.GRAPH_MODE, device_target="GPU")
    pipe = pipeline("language_modeling")
    print(pipe(eval_data_path="/home/workspace/mindtransformer/pipe/test-mindrecord"))
