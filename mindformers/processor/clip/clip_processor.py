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

'''
ClipProcessor
'''
from mindformers.mindformer_book import MindFormerBook
from ..base_processor import BaseProcessor
from ...tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class ClipProcessor(BaseProcessor):
    '''
    Clip processor,
    consists of a feature extractor for image input,
    and a tokenizer for text input.
    '''
    _support_list = MindFormerBook.get_model_support_list()['clip']

    def __init__(self, feature_extractor=None, tokenizer=None):
        super(ClipProcessor, self).__init__(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer
        )
