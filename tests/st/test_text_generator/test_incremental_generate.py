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
Test module for testing the incremental generate for text generator.
How to run this:
pytest tests/st/test_text_generator/test_incremental_generate.py
"""
import mindspore as ms

from mindformers import AutoConfig, AutoModel, AutoTokenizer

ms.set_context(mode=0)


class TestIncrementalGenerate:
    """A test class for testing text generate features."""
    def setup_method(self):
        """setup method."""
        self.test_model_list = ['glm_6b', 'llama_7b', 'bloom_560m']

    def test_incremental_generate(self):
        """
        Feature: incremental generate.
        Description: Test incremental generate by input model type.
        Expectation: TypeError, ValueError, RuntimeError
        """
        for model_type in self.test_model_list:
            config = AutoConfig.from_pretrained(model_type)
            # set incremental infer config
            config.batch_size = 1
            config.use_past = True
            model = AutoModel.from_config(config)
            tokenizer = AutoTokenizer.from_pretrained(model_type)
            input_ids = tokenizer("hello")["input_ids"]
            output = model.generate(input_ids, max_length=20)
            print(tokenizer.decode(output))
            del tokenizer
            del model
