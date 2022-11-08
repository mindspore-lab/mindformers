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


def test_pipeline_qa_task():
    """
    Feature: The pipeline test for question_answering
    Description: Using pipeline interface to test question_answering task
    Expectation: The returned ret is not 0.
    """
    from mindtransformer import pipeline
    pipe = pipeline("question_answering")
    print(pipe(vocab_file_path="./vocab.txt", eval_data_path="./eval.json"))


def test_pipeline_text_classification_task():
    """
    Feature: The pipeline test for text_classification
    Description: Using pipeline interface to test text_classification task
    Expectation: The returned ret is not 0.
    """
    from mindtransformer import pipeline
    pipe = pipeline("text_classification")
    print(pipe(eval_data_path="./eval.tf_record"))


def test_pipeline_language_modeling_task():
    """
    Feature: The pipeline test for language_modeling
    Description: Using pipeline interface to test language_modeling task
    Expectation: The returned ret is not 0.
    """
    from mindtransformer import pipeline
    pipe = pipeline("language_modeling")
    print(pipe(eval_data_path="./wikitext-2/test/test-mindrecord"))
