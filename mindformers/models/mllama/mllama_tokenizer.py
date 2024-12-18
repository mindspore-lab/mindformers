# Copyright 2024 Huawei Technologies Co., Ltd
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
"""llama3 tokenizer APIs."""

import base64
from typing import Dict
import json

from mindformers.mindformer_book import MindFormerBook
from mindformers.models.mllama.llama3_2_tokenizer import Llama3Tokenizer
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.utils import check_file

try:
    import tiktoken
except ImportError as e:
    raise ImportError("Package 'tiktoken' required to run Llama3. please install it with pip.") from e

__all__ = ['MllamaTokenizer']

PAT_STR = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| " \
          r"?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
    with open(tiktoken_bpe_file, "rb") as f:
        contents = f.read()
    return {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }


def _load_tokenizer_json(json_file):
    with open(json_file, "rb") as f:
        contents = json.loads(f.read())
    return {
        bytes(token, encoding='utf8'): int(rank)
        for token, rank in contents['model']['vocab'].items()
    }


# pylint: disable=W0223
@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class MllamaTokenizer(Llama3Tokenizer):
    """Mllama Tokenizer"""
    VOCAB_FILES = {'vocab_file': 'tokenizer.json'}
    FILE_LIST = []
    _support_list = MindFormerBook.get_tokenizer_support_list()['mllama']
    special_tokens: Dict[str, int]

    def __init__(self,
                 vocab_file,
                 bos_token="<|begin_of_text|>",
                 eos_token="<|end_of_text|>",
                 pad_token="<|finetune_right_pad_id|>",
                 add_bos_token=False,
                 add_eos_token=False,
                 errors="replace",
                 num_reserved_special_tokens=256,
                 **kwargs):
        super().__init__(vocab_file,
                         bos_token,
                         eos_token,
                         pad_token,
                         add_bos_token,
                         add_eos_token,
                         errors,
                         num_reserved_special_tokens,
                         **kwargs)
        self.errors = errors
        self.vocab_file = vocab_file
        check_file(vocab_file, "tokenizer")
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        if vocab_file.split('.')[-1] == 'json':
            self.mergeable_ranks = _load_tokenizer_json(vocab_file)
        else:
            self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)  # type: dict[bytes, int]
        num_base_tokens = len(self.mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|finetune_right_pad_id|>",
            "<|step_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>",
            "<|eot_id|>",
            "<|python_tag|>",
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(2, num_reserved_special_tokens - 9)
        ] + ["<|image|>"]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }

        self.tokenizer = tiktoken.Encoding(
            "Llama3",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.decoder = {
            v: k for k, v in self.mergeable_ranks.items()
        }  # type: dict[int, bytes|str]
        self.decoder.update({v: k for k, v in self.special_tokens.items()})
