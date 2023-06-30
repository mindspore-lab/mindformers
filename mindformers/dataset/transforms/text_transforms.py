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
"""Text Transforms."""
import numpy as np
from ...tools.register import MindFormerRegister, MindFormerModuleType


__all__ = [
    'RandomChoiceTokenizerForward', 'TokenizerForward', 'TokenizeWithLabel', 'LabelPadding'
]


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class RandomChoiceTokenizerForward:
    """Random Choice Tokenizer Forward"""
    def __init__(self, tokenizer, max_length=77, padding="max_length", random_seed=2022):
        self.max_length = max_length
        self.padding = padding
        self.tokenizer = tokenizer
        self.random_seed = random_seed

    def __call__(self, text):
        text_list = text.tolist()
        np.random.seed(self.random_seed)
        index = np.random.choice(len(text_list))

        token_id = self.tokenizer(
            text_list[index].decode("utf-8", "ignore") if \
                isinstance(text_list[index], bytes) else text_list[index],
            max_length=self.max_length,
            padding=self.padding
        )["input_ids"]
        return token_id

@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class TokenizerForward:
    """Tokenizer Forward"""
    def __init__(self, tokenizer, max_length=77, padding="max_length"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding

    def __call__(self, text):
        """call method"""
        text = text.tolist() if isinstance(text, np.ndarray) else text
        for i in range(len(text)):
            if isinstance(text[i], bytes):
                text[i] = text[i].decode("utf-8", "ignore")
        token_id = self.tokenizer(
            text, max_length=self.max_length,
            padding=self.padding)["input_ids"]
        return token_id

@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class TokenizeWithLabel:
    """Tokenizer With Label"""
    def __init__(self, tokenizer, max_length=128, padding="max_length"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding

    def __call__(self, *inputs):
        """call method"""
        text = inputs[0]
        label_id = inputs[1]
        text = text.tolist() if isinstance(text, np.ndarray) else text
        if isinstance(text, bytes):
            text = text.decode("utf-8", "ignore")
        output = self.tokenizer(text,
                                max_length=self.max_length, padding=self.padding)

        input_ids = np.array(output["input_ids"], dtype=np.int32)
        token_type_ids = np.array(output["token_type_ids"], dtype=np.int32)
        attention_mask = np.array(output["attention_mask"], dtype=np.int32)

        return input_ids, token_type_ids, attention_mask, label_id

@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class LabelPadding:
    """Label Padding"""
    def __init__(self, max_length, padding_value=0):
        self.max_length = max_length
        self.padding_value = padding_value

    def __call__(self, label_id):
        """call method"""
        if len(label_id) > self.max_length - 1:
            label_id = label_id[:(self.max_length - 1)]

        pad_label_id = []
        # For CLS token
        pad_label_id.append(self.padding_value)
        for i in range(len(label_id)):
            pad_label_id.append(label_id[i])

        while len(pad_label_id) < self.max_length:
            pad_label_id.append(self.padding_value)

        pad_label_id = np.array(pad_label_id, dtype=np.int32)

        return pad_label_id
