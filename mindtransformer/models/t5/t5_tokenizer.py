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
"""T5 Tokenzier"""

import sentencepiece as spm


class T5Tokenzier:
    """
        The tokenizer for T5 model
    """
    def __init__(self, sp_model):
        """
        Initialize the sentence piece model according to the model path
        Args:
             sp_model(str): the sentence piece model path.
        """
        self.s = spm.SentencePieceProcessor(model_file=sp_model)


    def tokenize(self, txt):
        """
        Tokenize the text.

        Args:
            txt(str): The text string.

        Return:
            List of tokens.
        """
        token_list = self.s.encode(txt, out_type=str)
        return token_list

    def convert_str_to_ids(self, txt):
        """
        Given the text and convert it to a list of ids

        Args:
            txt(str): The text string.

        Return
            a list. Where each element is an id number in the vocab file.
        """
        return self.s.encode(txt)

    def convert_ids_to_str(self, id_list):
        """
        Given the id list and convert it to the string

        Args:
            id_list(list): A list of the

        Return
            a list. Where each element is an id number in the vocab file.
        """
        return self.s.decode(id_list)
