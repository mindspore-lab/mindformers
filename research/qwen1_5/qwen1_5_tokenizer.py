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
"""Tokenization classes for Qwen2."""

import json
import os
import unicodedata
from functools import lru_cache
from typing import Optional, Tuple, Union, List, Dict
from packaging import version
import regex as re
from tokenizers import AddedToken

from mindspore import log as logger
from mindformers.models.base_tokenizer import Tokenizer, TensorType
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"qwen/qwen-tokenizer": "vocab.json"},
    "merges_file": {"qwen/qwen-tokenizer": "merges.txt"},
}

MAX_MODEL_INPUT_SIZES = {"qwen/qwen-tokenizer": 32768}

PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

IMSTART = "<|im_start|>"  # used in Qwen2-72B-chat
IMEND = "<|im_end|>"  # used in Qwen2-72B-chat
IMSTARTID = 151644
IMENDID = 151645


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class Qwen2Tokenizer(Tokenizer):
    """Qwen2 Tokenizer"""

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = MAX_MODEL_INPUT_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
            self,
            vocab_file,
            merges_file,
            errors="replace",
            unk_token="<|endoftext|>",
            bos_token=None,
            eos_token="<|endoftext|>",
            pad_token="<|endoftext|>",
            clean_up_tokenization_spaces=False,
            split_special_tokens=False,
            **kwargs,
    ):
        # Qwen vocab does not contain control tokens; added tokens need to be special
        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(unk_token, str)
            else unk_token
        )
        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(pad_token, str)
            else pad_token
        )
        im_start_token = AddedToken(IMSTART, lstrip=False, rstrip=False, special=True, normalized=False)
        im_end_token = AddedToken(IMEND, lstrip=False, rstrip=False, special=True, normalized=False)
        self.special_tokens = {
            IMSTART: IMSTARTID,
            IMEND: IMENDID,
        }
        self.im_start_id = self.special_tokens[IMSTART]
        self.im_end_id = self.special_tokens[IMEND]
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        bpe_merges = []
        with open(merges_file, encoding="utf-8") as merges_handle:
            for line in merges_handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                bpe_merges.append(tuple(line.split()))
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        self.cache = {}

        self.pat = re.compile(PRETOKENIZE_REGEX)

        if kwargs.get("add_prefix_space", False):
            logger.warning(
                "Qwen2Tokenizer does not support `add_prefix_space`, setting it to True has no effect."
            )

        self.chat_template = kwargs.get("chat_template", None)

        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            split_special_tokens=split_special_tokens,
            **kwargs,
        )
        self.add_tokens(im_start_token, special_tokens=True)
        self.add_tokens(im_end_token, special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        """byte pair encoding"""
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text, **kwargs):
        """Tokenize a string."""
        logger.info(kwargs)
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def decode(
            self,
            token_ids,
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: Optional[bool] = False,
            spaces_between_special_tokens: bool = False,
            **kwargs,
    ) -> str:
        """decode"""
        return super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, str]:
        """save vocabulary"""
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return None
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    def prepare_for_tokenization(self, text, **kwargs):
        text = unicodedata.normalize("NFC", text)
        return text, kwargs

    def apply_chat_template(
            self,
            conversation: Union[List[Dict[str, str]], "Conversation"],
            chat_template: Optional[str] = None,
            add_generation_prompt: bool = False,
            tokenize: bool = True,
            padding: bool = False,
            truncation: bool = False,
            max_length: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            **tokenizer_kwargs,
    ) -> Union[str, List[int]]:
        """
                Converts a Conversation object or a list of dictionaries with `"role"` and `"content"` keys to a list
                of token ids. This method is intended for use with chat models, and will read the tokenizer's
                chat_template attribute to determine the format and control tokens to use when converting.
                When chat_template is None, it will fall back to the default_chat_template specified at the class level.

                    Args:
                    conversation (Union[List[Dict[str, str]], "Conversation"]): A Conversation object or list of dicts
                        with "role" and "content" keys, representing the chat history so far.
                    chat_template (str, *optional*): A Jinja template to use for this conversion. If
                        this is not passed, the model's default chat template will be used instead.
                    add_generation_prompt (bool, *optional*): Whether to end the prompt with the token(s) that indicate
                        the start of an assistant message. This is useful when you want to generate a response from
                        the model. Note that this argument will be passed to the chat template, and so it must be
                        supported in the template for this argument to have any effect.
                    tokenize (`bool`, defaults to `True`):
                        Whether to tokenize the output. If `False`, the output will be a string.
                    padding (`bool`, defaults to `False`):
                        Whether to pad sequences to the maximum length. Has no effect if tokenize is `False`.
                    truncation (`bool`, defaults to `False`):
                        Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.
                    max_length (`int`, *optional*):
                        Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize
                        is `False`. If not specified, the tokenizer's `max_length` attribute will be used as a default.
                    return_tensors (`str` or [`~utils.TensorType`], *optional*):
                        If set, will return tensors of a particular framework. Has no effect if tokenize is `False`.
                        Acceptable values are:
                            - `'tf'`: Return TensorFlow `tf.Tensor` objects.
                            - `'pt'`: Return PyTorch `torch.Tensor` objects.
                            - `'np'`: Return NumPy `np.ndarray` objects.
                            - `'jax'`: Return JAX `jnp.ndarray` objects.
                    **tokenizer_kwargs: Additional kwargs to pass to the tokenizer.

                Returns:
                    `List[int]`: A list of token ids representing the tokenized chat so far, including control tokens.
                    This output is ready to pass to the model, either directly or via methods like `generate()`.
                """

        if hasattr(conversation, "messages"):
            # Indicates it's a Conversation object
            conversation = conversation.messages

        # priority: `chat_template` argument > `tokenizer.chat_template` > `tokenizer.default_chat_template`
        if chat_template is None:
            if self.chat_template is not None:
                chat_template = self.chat_template
            else:
                chat_template = self.default_chat_template

        # Compilation function uses a cache to avoid recompiling the same template
        compiled_template = self._compile_jinja_template(chat_template)

        rendered = compiled_template.render(
            messages=conversation, add_generation_prompt=add_generation_prompt, **self.special_tokens_map
        )

        if padding is True:
            padding = "max_length"  # There's only one sequence here, so "longest" makes no sense
        if tokenize:
            return self.encode(
                rendered,
                add_special_tokens=False,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                **tokenizer_kwargs,
            )
        return rendered

    @lru_cache(128)
    def _compile_jinja_template(self, chat_template):
        """_compile_jinja_template"""
        try:
            import jinja2
            from jinja2.exceptions import TemplateError
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError:
            raise ImportError("apply_chat_template requires jinja2 to be installed.")

        if version.parse(jinja2.__version__) <= version.parse("3.0.0"):
            raise ImportError(
                "apply_chat_template requires jinja2>=3.0.0 to be installed. Your version is " f"{jinja2.__version__}."
            )

        def raise_exception(message):
            raise TemplateError(message)

        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = raise_exception
        return jinja_env.from_string(chat_template)

    @property
    def default_chat_template(self):
        """
        This template formats inputs in the standard ChatML format.
        """
        logger.warning(
            "\nNo chat template is defined for this tokenizer - using a default chat template "
            "that implements the ChatML format (without BOS/EOS tokens!). If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
        )
        return (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
