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
Test module for testing the batch infer for text generator.
How to run this:
pytest tests/st/test_text_generator/test_batch_generate.py
"""
import mindspore as ms

from mindformers import AutoConfig, AutoModel, AutoTokenizer

ms.set_context(mode=0)


class TestBatchGenerate:
    """A test class for testing text generate features."""
    def setup_method(self):
        """setup method."""
        self.test_model_list = ['glm_6b', 'bloom_560m']
        # self.test_model_list = ['glm_6b', 'llama_7b', 'bloom_560m']

    def test_batch_generate(self):
        """
        Feature: batch generate.
        Description: Test batch generate for language models.
        Expectation: TypeError, ValueError, RuntimeError
        """
        question_list = [
            "Hello",
            "Introduce yourself",
            "I love Beijing, because",
            "what color is the sky?"
        ]
        batch_size = len(question_list)
        seq_len = 256
        for model_name in self.test_model_list:
            # set model config
            config = AutoConfig.from_pretrained(model_name)
            config.batch_size = batch_size
            config.seq_length = seq_len
            config.max_decode_length = seq_len
            config.use_past = True
            print(f"config is: {config}")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_config(config)

            inputs = tokenizer(question_list, max_length=seq_len, padding="max_length")["input_ids"]
            print(f"inputs is: {inputs}")
            outputs = model.generate(inputs)
            for output in outputs:
                print(tokenizer.decode(output))
            del tokenizer
            del model
