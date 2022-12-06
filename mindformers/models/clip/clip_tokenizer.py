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

"""
This is a temporary version of clip tokenizer
"""
import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re
import mindspore as ms

from ...mindformer_book import MindFormerBook
from ...tools.register import MindFormerRegister, MindFormerModuleType
from ...tools.download_tools import downlond_with_progress_bar
from ..base_tokenizer import PretrainedTokenizer

@lru_cache()
def default_bpe():
    """bpe path"""
    path = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                        'clip', "bpe_simple_vocab_16e6.txt.gz")
    if not os.path.exists(path):
        url = "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/" \
              "XFormer_for_mindspore/clip/bpe_simple_vocab_16e6.txt.gz"
        downlond_with_progress_bar(url, path)
    return path

def get_pairs(input_wd):
    """get_pairs"""
    output = set()
    prev_char = input_wd[0]
    for char in input_wd[1:]:
        output.add((prev_char, char))
        prev_char = char
    return output

@lru_cache()
def bytes_to_unicode():
    """bytes_to_unicode"""
    input_bt = list(range(ord("!"), ord("~")+1))\
         +list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    output_cd = input_bt[:]
    num = 0
    for item in range(2**8):
        if item not in input_bt:
            input_bt.append(item)
            output_cd.append(2**8+num)
            num += 1
    output_cd = [chr(item) for item in output_cd]
    return dict(zip(input_bt, output_cd))

def whitespace_clean(input_text):
    """whitespace clean"""
    input_text = re.sub(r'\s+', ' ', input_text)
    input_text = input_text.strip()
    return input_text

def basic_clean(input_text):
    """basic_clean"""
    input_text = ftfy.fix_text(input_text)
    input_text = html.unescape(html.unescape(input_text))
    return input_text.strip()

class TempTokenizer:
    """Simple Tokenizer"""
    def __init__(self, text_path):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + '</w>' for v in vocab]

        merges = gzip.open(text_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]

        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])

        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|
        've|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
        self.flag_dict = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}

    def tokenize_alg(self, input_tk):
        """bpe"""
        if input_tk in self.flag_dict:
            return self.flag_dict[input_tk]
        word = tuple(input_tk[:-1]) + (input_tk[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return input_tk+'</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        word = ' '.join(word)
        self.flag_dict[input_tk] = word
        return word

    def decode(self, input_ids):
        """decode"""
        output_text = ''.join([self.decoder[input_id] for input_id in input_ids])
        output_text = bytearray([self.byte_decoder[c] for
                                 c in output_text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return output_text

    def encode(self, content):
        """encode"""
        output_ids = []
        content = whitespace_clean(basic_clean(content)).lower()
        for token in re.findall(self.pat, content):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            output_ids.extend(self.encoder[bpe_token] for bpe_token in self.tokenize_alg(token).split(' '))
        return output_ids

@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class ClipTokenizer(PretrainedTokenizer):
    '''clip tokenizer'''
    def __init__(self):
        super(ClipTokenizer, self).__init__()
        path = default_bpe()
        self.tool = TempTokenizer(path)

    def __call__(self, input_texts, token_num=77, drop_tail=False):
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        end_flag = self.tool.encoder["<|endoftext|>"]
        start_flag = self.tool.encoder["<|startoftext|>"]

        all_flag = [[start_flag] + self.tool.encode(text) + [end_flag] for text in input_texts]
        output = ms.ops.Zeros()((len(all_flag), token_num), ms.int64)

        for i, tokens in enumerate(all_flag):
            if len(tokens) > token_num:
                if drop_tail:
                    tokens = tokens[:token_num]
                    tokens[-1] = end_flag
                else:
                    raise RuntimeError(f"Input {input_texts[i]} is too"
                                       f" long for context length {token_num}")
            output[i, :len(tokens)] = ms.Tensor(tokens)

        return ms.Tensor(output)

    def _tokenize(self, text, **kwargs):
        if isinstance(text, str):
            text = [text]
        end_flag = self.tool.encoder["<|endoftext|>"]
        start_flag = self.tool.encoder["<|startoftext|>"]
        all_flag = [[start_flag] + self.tool.encode(text) + [end_flag] for text in text]
        return all_flag

    def save_vocabulary(self, save_directory, filename_prefix):
        output_file_path = os.path.join(save_directory, filename_prefix)
        with open(output_file_path, 'w') as fp:
            for k in self.tool.encoder.keys():
                fp.write(k + '\n')
        return output_file_path

    def tokenize(self, text):
        """Tokenizer the input_text"""
        if not isinstance(text, str):
            raise ValueError("Text should be type str, but found type", type(text))
        return self._tokenize(text)
