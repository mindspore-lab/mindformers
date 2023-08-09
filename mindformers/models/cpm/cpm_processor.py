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

"""CPMProcessor
"""

from mindformers.mindformer_book import MindFormerBook
from mindformers.models.base_tokenizer import BaseTokenizer
from mindformers.models.base_processor import BaseProcessor
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ['CPMProcessor']


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class CPMProcessor(BaseProcessor):
    """
    GLM processor,
    consists of a tokenizer (BaseTokenizer) for text input.
    """
    _support_list = MindFormerBook.get_processor_support_list()['cpm']

    def __init__(self, tokenizer=None,
                 max_length=2048, padding='max_length', return_tensors='ms'):
        super(CPMProcessor, self).__init__(
            tokenizer=tokenizer,
            max_length=max_length,
            padding=padding,
            return_tensors=return_tensors
        )

    def __call__(self, text_input=None, image_input=None):
        """call function"""
        output = {}
        if text_input is not None and self.tokenizer:
            if not isinstance(self.tokenizer, BaseTokenizer):
                raise TypeError(f"tokenizer should inherited from the BaseTokenizer,"
                                f" but got {type(self.tokenizer)}.")
            # Format the input into a batch
            if isinstance(text_input, str):
                text_input = [text_input]
            text_output = self.tokenizer(text_input, return_tensors=self.return_tensors,
                                         max_length=self.max_length,
                                         padding=self.padding)["input_ids"]
            output['text'] = text_output
        return output
