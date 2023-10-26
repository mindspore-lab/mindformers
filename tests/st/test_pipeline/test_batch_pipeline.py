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
Test module for testing the batch infer for pipeline.
How to run this:
pytest tests/st/test_pipeline/test_batch_pipeline.py
"""
import mindspore as ms

from mindformers import TextGenerationPipeline, pipeline

ms.set_context(mode=0)


class TestBatchPipeline:
    """A test class for testing pipeline features."""
    def setup_method(self):
        """setup method."""
        self.task_name = "text_generation"
        self.model_name = "glm2_6b"
        self.batch_size = 2
        self.use_past = True
        self.seq_len = 128

    def test_text_generation(self):
        """
        Feature: text_generation pipeline.
        Description: Test batch generate for text_generation pipeline.
        Expectation: TypeError, ValueError, RuntimeError
        """
        question_list = ["你好", "请介绍下华为"]
        task_pipeline = TextGenerationPipeline(self.model_name,
                                               batch_size=self.batch_size,
                                               use_past=self.use_past,
                                               seq_length=self.seq_len)
        output = task_pipeline(question_list,
                               max_new_tokens=32,
                               do_sample=True,
                               top_k=3)
        print(output)

    def test_pipeline(self):
        """
        Feature: pipeline interface.
        Description: Test batch generate for pipeline interface.
        Expectation: TypeError, ValueError, RuntimeError
        """
        question_list = ["你好", "请介绍下华为"]
        task_pipeline = pipeline(self.task_name,
                                 self.model_name,
                                 batch_size=self.batch_size,
                                 use_past=self.use_past,
                                 seq_length=self.seq_len)
        output = task_pipeline(question_list,
                               max_new_tokens=32,
                               do_sample=True,
                               top_k=3)
        print(output)
