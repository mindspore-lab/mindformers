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
# This file was refer to project:
# https://github.com/zzwj66/models/blob/master/research/nlp/albert/src/tokenization.py
# ============================================================================
"""The bert tokenizer"""
import collections
import json
import os
import unicodedata

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.base_tokenizer import Tokenizer
from ...mindformer_book import MindFormerBook

__all__ = ['BertTokenizer', 'BasicTokenizer']

def convert_to_unicode(text):
    """
    Convert text into unicode type.
    Args:
        text: input str.

    Returns:
        input str in unicode.
    """
    ret = text
    if isinstance(text, str):
        ret = text
    elif isinstance(text, bytes):
        ret = text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))
    return ret

def vocab_to_dict_key_token(vocab_file):
    """Loads a vocab file into a dict, key is token."""
    if vocab_file.endswith(".json"):
        return json.load(open(vocab_file, 'r'))
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r") as fp:
        for line in fp:
            if not line:
                break
            line = line.strip()
            token = convert_to_unicode(line)
            vocab[token] = index
            index += 1
    return vocab

def vocab_to_dict_key_id(vocab_file):
    """Loads a vocab file into a dict, key is id."""
    vocab_key_id = vocab_to_dict_key_token(vocab_file)
    vocab = {v: k for k, v in vocab_key_id.items()}
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    return text.split() if text else []

def convert_tokens_to_ids(vocab_file, tokens):
    """
    Convert tokens to ids.
    Args:
        vocab_file: path to vocab.txt.
        tokens: list of tokens.

    Returns:
        list of ids.
    """
    vocab_dict = vocab_to_dict_key_token(vocab_file)
    output = []
    for token in tokens:
        output.append(vocab_dict[token])
    return output


def convert_ids_to_tokens(vocab_file, ids):
    """
    Convert ids to tokens.
    Args:
        vocab_file: path to vocab.txt.
        ids: list of ids.

    Returns:
        list of tokens.
    """
    vocab_dict = vocab_to_dict_key_id(vocab_file)
    output = []
    for item in ids:
        output.append(vocab_dict[item])
    return output


def convert_tokens_to_string(tokens):
    """
    For OPT model, the vocab contains the Ġ for each sub word, so we need to remove them.
    """
    string = " ".join(tokens)
    string = string.replace(' Ġ', ' ')
    return string


class FullTokenizer:
    """
    Full tokenizer
    """
    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab_dict = vocab_to_dict_key_token(vocab_file)
        self.do_lower_case = do_lower_case
        self.basic_tokenize = BasicTokenizer(do_lower_case)
        self.wordpiece_tokenize = WordpieceTokenizer(self.vocab_dict)

    def tokenize(self, text):
        """
        Do full tokenization.
        Args:
            text: str of text.

        Returns:
            list of tokens.
        """
        tokens_ret = []
        text = convert_to_unicode(text)
        for tokens in self.basic_tokenize.tokenize(text):
            wordpiece_tokens = self.wordpiece_tokenize.tokenize(tokens)
            tokens_ret.extend(wordpiece_tokens)
        return tokens_ret


class BasicTokenizer:
    """
    Basic tokenizer
    """
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    _CHINESE_SPACE = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF),
                      (0x2A700, 0x2B73F), (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF), (0xF900, 0xFAFF),
                      (0x2F800, 0x2FA1F))
    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case

    def _clean_and_tokenizer(self, text):
        """Clean and tokenizer the text"""
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        return orig_tokens

    def tokenize(self, text):
        """
        Do basic tokenization.
        Args:
            text: text in unicode.

        Returns:
            a list of tokens split from text
        """
        orig_tokens = self._clean_and_tokenizer(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = [char for char in text if unicodedata.category(char) != 'Mn']
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        start_new_word = True
        output = []
        for char in text:
            is_punctuated = _is_punctuation(char)
            if not is_punctuated:
                if start_new_word:
                    output.append([])
                    start_new_word = False
                output[-1].append(char)
            else:
                output.append([char])
                start_new_word = True
        return ["".join(x) for x in output]

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        filter_special = filter(lambda x: ord(x) not in (0, 0xfffd, _is_control(x)), text)
        output = map(lambda char: " " if _is_whitespace(char) else char, filter_special)
        return "".join(list(output))

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            is_chinese = self._is_chinese_char(cp)
            if is_chinese:
                output.append(" ")
            output.append(char)
            if is_chinese:
                output.append(" ")
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        res = map(lambda item: item[0] <= cp <= item[1], self._CHINESE_SPACE)
        return any(res)


class WordpieceTokenizer:
    """
    Wordpiece tokenizer
    """
    def __init__(self, vocab):
        self.vocab_dict = vocab

    def tokenize(self, tokens):
        """
        Do word-piece tokenization
        Args:
            tokens: a word.

        Returns:
            a list of tokens that can be found in vocab dict.
        """
        output_tokens = []
        tokens = convert_to_unicode(tokens)
        for token in whitespace_tokenize(tokens):
            chars = list(token)
            len_chars = len(chars)
            start = 0
            end = len_chars
            while start < len_chars:
                while start < end:
                    substr = "".join(token[start:end])
                    if start != 0:
                        substr = "##" + substr
                    if substr in self.vocab_dict:
                        output_tokens.append(substr)
                        start = end
                        end = len_chars
                    else:
                        end = end - 1
                if start == end and start != len_chars:
                    output_tokens.append("[UNK]")
                    break
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char in (" ", "\t", "\n", "\r") or unicodedata.category(char) == 'Zs':
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    control_char = ["\t", "\n", "\r"]
    if char in control_char:
        return False
    if unicodedata.category(char) in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    distance = ((33, 47), (58, 64), (91, 96), (123, 126))
    for start, end in distance:
        if start <= cp <= end:
            return True
    if unicodedata.category(char).startswith("P"):
        return True
    return False

@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class BertTokenizer(Tokenizer):
    """
        Bert Tokenizer.
    """
    VOCAB_FILES = {'vocab_file': 'vocab.txt'}
    FILE_LIST = ['tokenizer_config.json', 'special_tokens_map.json']
    _support_list = MindFormerBook.get_tokenizer_support_list()['bert']
    _support_list.extend(MindFormerBook.get_config_support_list()['tokcls']['bert'])
    _support_list.extend(MindFormerBook.get_config_support_list()['txtcls']['bert'])
    _support_list.extend(MindFormerBook.get_config_support_list()['qa']['bert'])

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 do_basic_tokenize=True,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 is_tokenize_char=False,
                 **kwargs):
        super(BertTokenizer, self).__init__(do_lower_case=do_lower_case,
                                            do_basic_tokenize=do_basic_tokenize,
                                            unk_token=unk_token,
                                            sep_token=sep_token,
                                            pad_token=pad_token,
                                            cls_token=cls_token,
                                            mask_token=mask_token,
                                            **kwargs)
        self.do_lower_case = do_lower_case
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

        self.vocab_dict = vocab_to_dict_key_token(vocab_file)
        self.vocab_id2token = {v: k for k, v in self.vocab_dict.items()}
        self.word_piece_tokenizer = WordpieceTokenizer(vocab=self.vocab_dict)
        self.mask_index = []
        self.is_tokenize_char = is_tokenize_char

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1

        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

    def tokenize(self, text):
        text = convert_to_unicode(text)
        if not isinstance(text, str):
            raise ValueError("Text should be type str, but found type", type(text))
        return self._tokenize(text)

    def _process_mask_tokens(self, text):
        """process mask tokens in text"""
        text_tokenize = []
        if self._mask_token in text:
            while self._mask_token in text:
                ind = text.index(self._mask_token)
                text_tokenize.extend(self.basic_tokenizer.tokenize(text[:ind]))
                text_tokenize.append(self._mask_token)
                text = text[ind + len(self._mask_token):]
            text_tokenize.extend(self.basic_tokenizer.tokenize(text))
            self.mask_index = [ind for ind, x in enumerate(text_tokenize) if x == self._mask_token]
        else:
            text_tokenize = self.basic_tokenizer.tokenize(text)
        return text_tokenize

    def _tokenize(self, text, **kwargs):
        tokens_ret = []
        if self.is_tokenize_char:
            for character in text:
                if self.do_lower_case:
                    character = character.lower()
                if character in self.vocab_dict:
                    tokens_ret.append(character)
                else:
                    tokens_ret.append(self.unk_token)
        else:
            if self.do_basic_tokenize:
                for tokens in self._process_mask_tokens(text):
                    wordpiece_tokens = self.word_piece_tokenizer.tokenize(tokens)
                    tokens_ret.extend(wordpiece_tokens)
            else:
                tokens_ret = self.word_piece_tokenizer.tokenize(text)
        return tokens_ret

    def _convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab_dict[tokens]
        output = []
        for token in tokens:
            output.append(self.vocab_dict[token])
        return output

    def _convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.vocab_id2token[ids]

        if isinstance(ids, list):
            output = []
            for item in ids:
                output.append(self.vocab_id2token[item])
            return output
        raise TypeError(f"The type of ids should be int or list, but found {type(ids)}.")

    def save_vocabulary(self, save_directory, filename_prefix):
        """write the word to the files"""
        output_file_path = os.path.join(save_directory, filename_prefix)
        with open(output_file_path, 'w') as fp:
            for k in self.vocab_dict.keys():
                fp.write(k + '\n')
        return output_file_path

    @property
    def vocab_size(self):
        """Return the vocab size"""
        return len(self.vocab_dict)
