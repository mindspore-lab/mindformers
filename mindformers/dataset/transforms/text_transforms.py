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
    'RandomChoiceTokenizerForward'
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
        np.random.seed(self.random_seed)
        index = np.random.choice(len(text.tolist()))
        token_id = self.tokenizer(
            text.tolist()[index],
            max_length=self.max_length,
            padding=self.padding
        )["input_ids"]
        return token_id
