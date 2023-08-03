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
pytest tests/st/test_batch_generate.py
"""
import mindspore as ms

from mindformers import AutoTokenizer
from mindformers.models import GLMChatModel, GLMConfig

ms.set_context(mode=0)


class TestBatchGenerate:
    """A test class for testing batch generate."""
    def test_glm_batch_generate(self):
        """
        Feature: batch generate.
        Description: Test batch generate using glm model.
        Expectation: TypeError, ValueError, RuntimeError
        """
        question_list = [
            "你好",
            "Hello",
            "请介绍一下你自己",
            "请问为什么说地球是独一无二的？",
        ]
        batch_size = len(question_list)
        ckpt_file = "glm_6b_chat"
        seq_len = 256
        config = GLMConfig(
            batch_size=batch_size,
            seq_length=seq_len,
            max_decode_length=seq_len,
            use_past=True,
            checkpoint_name_or_path=ckpt_file,
        )
        tokenizer = AutoTokenizer.from_pretrained("glm_6b")
        model = GLMChatModel(config)

        inputs = tokenizer(question_list, max_length=seq_len, padding="max_length")["input_ids"]
        outputs = model.generate(inputs)
        for output in outputs:
            print(tokenizer.decode(output))
