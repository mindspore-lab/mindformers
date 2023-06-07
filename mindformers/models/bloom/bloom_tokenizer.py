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
""" Bloom Tokenzier """
import json
from functools import lru_cache
import regex as re
from mindformers import GPT2Tokenizer
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ['BloomTokenizer']

@lru_cache()
def bytes_to_unicode():
    """
    bytes to unicode
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(i) for i in cs]
    return dict(zip(bs, cs))


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class BloomTokenizer(GPT2Tokenizer):
    r"""
    Tokenize the input string and convert them into the ids.

    Args:
        vocab_file(str): The vocabulary file path.
        merge_file(str): The merge file path.
        add_prefix_space(bool): whether to add a whitespace in the front of text. Default "False"

    Outputs:
        A dict contains the processed ids
    """
    def __init__(
            self,
            tokenizer_json_file,
            merge_file,
            add_prefix_space=False):
        super(BloomTokenizer, self).__init__(tokenizer_json_file, merge_file)
        with open(tokenizer_json_file, 'r', encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)["model"]["vocab"]
        with open(tokenizer_json_file, 'r', encoding="utf-8") as vocab_handle:
            bpe_merges = json.load(vocab_handle)["model"]["merges"]

        self.decoder = {v: k for k, v in self.encoder.items()}
        self.vocab_size = len(self.decoder)

        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]

        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.add_prefix_space = add_prefix_space
        self.cache = {}

        self.unk_token = "<|unk|>"
        self.unk_token_id = 0
        self.bos_token = "<|s|>"
        self.bos_token_id = 1
        self.eos_token = "<|/s|>"
        self.eos_token_id = 2
        self.pad_token = "<|pad|>"
        self.pad_token_id = 3
