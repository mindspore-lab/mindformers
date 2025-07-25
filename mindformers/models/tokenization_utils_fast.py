# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
 Tokenization classes for fast tokenizers (provided by HuggingFace's tokenizers library). For slow (python) tokenizers
 see tokenization_utils.py
"""
import copy
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import tokenizers.pre_tokenizers as pre_tokenizers_fast
from tokenizers import Encoding as EncodingFast
from tokenizers import Tokenizer as TokenizerFast
from tokenizers.decoders import Decoder as DecoderFast
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer
from mindformers.tools.utils import FILE_PERMISSION
from .convert_slow_tokenizer import convert_slow_tokenizer
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import (
    AddedToken,
    BatchEncoding,
    PreTokenizedInput,
    PreTokenizedInputPair,
    PreTrainedTokenizerBase,
    SpecialTokensMixin,
    TextInput,
    TextInputPair,
    TruncationStrategy,
    PaddingStrategy
)


# Fast tokenizers (provided by HuggingFace tokenizer's library) can be saved in a single file
TOKENIZER_FILE = "tokenizer.json"
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

# Slow tokenizers have an additional added tokens files
ADDED_TOKENS_FILE = "added_tokens.json"

MODEL_TO_TRAINER_MAPPING = {
    "BPE": BpeTrainer,
    "Unigram": UnigramTrainer,
    "WordLevel": WordLevelTrainer,
    "WordPiece": WordPieceTrainer,
}

VOCAB_FILES_NAMES = {"tokenizer_file": TOKENIZER_FILE}


# pylint: disable=W0223
class PreTrainedTokenizerFast(PreTrainedTokenizerBase):
    r"""
    Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

    Handles all the shared methods for tokenization and special tokens, as well as methods for
    downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.

    This class also contains the added tokens in a unified way on top of all tokenizers so we don't have to handle the
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).

    Args:
        **kwargs (Any): Keyword arguments.

            - model_max_length (int, optional):
              The maximum length (in number of tokens) for the inputs to the transformer model.
              Set when the tokenizer is loaded with ``from_pretrained()`` based on the model's
              ``max_model_input_sizes`` attribute.  Default: ``1e-30``.
            - padding_side (str, optional):
              Specifies the side on which the model should have padding applied. Options are
              ['right', 'left']. The default value is picked from the class attribute of the
              same name.
            - truncation_side (str, optional):
              Specifies the side on which the model should have truncation applied.
              Options are ['right', 'left']. The default value is picked from the class
              attribute of the same name.
            - chat_template (str, optional):
              A Jinja template string used to format lists of chat messages.
              Default: ``"{% for message in messages %}{{'<|im_start|>' + message['role'] +
              '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{
              '<|im_start|>assistant\n' }}{% endif %}"``.
            - model_input_names (List[str], optional):
              Lists the names of inputs accepted by the forward pass of the model,
              such as "token_type_ids" or "attention_mask". Defaults to values picked
              from the class attribute of the same name. Default: ``None``.
            - bos_token (Union[str, tokenizers.AddedToken], optional):
              Represents the beginning of a sentence and is associated with
              ``self.bos_token`` and ``self.bos_token_id``. Default: ``None``.
            - eos_token (Union[str, tokenizers.AddedToken], optional):
              Represents the end of a sentence and is associated with ``self.eos_token``
              and ``self.eos_token_id``. Default: ``None``.
            - unk_token (Union[str, tokenizers.AddedToken], optional):
              Represents an out-of-vocabulary token and is associated with
              ``self.unk_token`` and ``self.unk_token_id``. Default: ``None``.
            - sep_token (Union[str, tokenizers.AddedToken], optional):
              A special token separating two different sentences in the same input
              (used by BERT, for example) and is associated with ``self.sep_token``
              and ``self.sep_token_id``. Default: ``None``.
            - pad_token (Union[str, tokenizers.AddedToken], optional):
              Used to make arrays of tokens the same size for batching purposes and
              will be ignored by attention mechanisms or loss computation. It is
              associated with ``self.pad_token`` and ``self.pad_token_id``. Default: ``None``.
            - cls_token (Union[str, tokenizers.AddedToken], optional):
              Represents the class of the input (used by BERT, for example) and is
              associated with ``self.cls_token`` and ``self.cls_token_id``. Default: ``None``.
            - mask_token (Union[str, tokenizers.AddedToken], optional):
              Represents a masked token (used by masked-language modeling pretraining
              objectives like BERT) and is associated with ``self.mask_token`` and
              ``self.mask_token_id``. Default: ``None``.
            - additional_special_tokens (Union[tuple, list, tokenizers.AddedToken], optional):
              Lists additional special tokens that are ensured to be skipped when
              decoding with ``skip_special_tokens`` set to True. They will be added
              at the end of the vocabulary if not already part of it. Default: ``None``.
            - clean_up_tokenization_spaces (bool, optional):
              Determines whether to clean-up spaces that were added when splitting the
              input text during the tokenization process. Default: ``True``.
            - split_special_tokens (bool, optional):
              Specifies whether special tokens should be split during the tokenization
              process. This affects the internal state of the tokenizer. By default, special
              tokens are not split. For example, if ``<s>`` is the ``bos_token``, then
              ``tokenizer.tokenize("<s>") = ['<s>']``. If ``split_special_tokens = True``,
              then ``tokenizer.tokenize("<s>")`` would result in ``['<','s', '>']``. Default: ``False``.
            - tokenizer_object (tokenizers.Tokenizer):
              A ``tokenizers.Tokenizer`` object from tokenizers to instantiate from.
            - tokenizer_file (str):
              A path to a local JSON file representing a previously serialized ``tokenizers.Tokenizer`` object
              from tokenizers.

    Returns:
        PreTrainedTokenizerFast instance.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class: PreTrainedTokenizer = None

    def __init__(self, *args, **kwargs):
        fast_tokenizer, slow_tokenizer, added_tokens_decoder, kwargs = self.init_atom_1(*args, **kwargs)

        self._tokenizer = fast_tokenizer

        if slow_tokenizer is not None:
            kwargs.update(slow_tokenizer.init_kwargs)

        self._decode_use_source_tokenizer = False

        kwargs = self.init_atom_2(**kwargs)

        # We call this after having initialized the backend tokenizer because we update it.
        super().__init__(**kwargs)

        # The following logic will be replaced with a single add_tokens once a fix is pushed to tokenizers
        # allows converting a slow -> fast, non-legacy: if the `tokenizer.json` does not have all the added tokens
        # uses the information stored in `added_tokens_decoder`.
        # this is costly for fast tokenizers as we re-compute the regex again. But not all tokens are added tokens
        tokens_to_add = [
            token
            for index, token in sorted(added_tokens_decoder.items(), key=lambda x: x[0])
            if token not in self.added_tokens_decoder
        ]
        encoder = list(self.added_tokens_encoder.keys()) + [str(token) for token in tokens_to_add]
        # if some of the special tokens are strings, we check if we don't already have a token
        tokens_to_add += [
            token for token in self.all_special_tokens_extended if token not in encoder and token not in tokens_to_add
        ]
        if tokens_to_add:
            # super hack: if a token.special is set, tokenizer ignores it for now so FIXME @ArthurZ
            # Accumulate added tokens into batches of special/non-special tokens, because calling add_tokens() for
            # individual tokens would repeatedly rebuild a trie, which can be slow.
            is_last_special = None
            tokens = []
            special_tokens = self.all_special_tokens
            for token in tokens_to_add:
                is_special = (
                    (token.special or str(token) in special_tokens)
                    if isinstance(token, AddedToken)
                    else str(token) in special_tokens
                )
                if is_last_special is None or is_last_special == is_special:
                    tokens.append(token)
                else:
                    self._add_tokens(tokens, special_tokens=is_last_special)
                    tokens = [token]
                is_last_special = is_special
            if tokens:
                self._add_tokens(tokens, special_tokens=is_last_special)

    def init_atom_1(self, *args, **kwargs):
        """init atom 1"""
        tokenizer_object = kwargs.pop("tokenizer_object", None)
        slow_tokenizer = kwargs.pop("__slow_tokenizer", None)
        fast_tokenizer_file = kwargs.pop("tokenizer_file", None)
        from_slow = kwargs.pop("from_slow", False)
        added_tokens_decoder = kwargs.pop("added_tokens_decoder", {})

        if from_slow and slow_tokenizer is None and self.slow_tokenizer_class is None:
            raise ValueError(
                "Cannot instantiate this tokenizer from a slow version. If it's based on sentencepiece, make sure you "
                "have sentencepiece installed."
            )

        if tokenizer_object is not None:
            fast_tokenizer = copy.deepcopy(tokenizer_object)
        elif fast_tokenizer_file is not None and not from_slow:
            # We have a serialization from tokenizers which let us directly build the backend
            fast_tokenizer_file = os.path.realpath(fast_tokenizer_file)
            if os.path.isfile(fast_tokenizer_file) and os.access(fast_tokenizer_file, os.R_OK):
                fast_tokenizer = TokenizerFast.from_file(fast_tokenizer_file)
            else:
                raise ValueError(f"The path of fast_tokenizer_file is {fast_tokenizer_file}, "
                                 "but it is not a valid file.")
        elif slow_tokenizer is not None:
            # We need to convert a slow tokenizer to build the backend
            fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)
        elif self.slow_tokenizer_class is not None:
            # We need to create and convert a slow tokenizer to build the backend
            # pylint: disable=E1102
            slow_tokenizer = self.slow_tokenizer_class(*args, **kwargs)
            fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)
        else:
            raise ValueError(
                "Couldn't instantiate the backend tokenizer from one of: \n"
                "(1) a `tokenizers` library serialization file, \n"
                "(2) a slow tokenizer instance to convert or \n"
                "(3) an equivalent slow tokenizer class to instantiate and convert. \n"
                "You need to have sentencepiece installed to convert a slow tokenizer to a fast one."
            )
        return fast_tokenizer, slow_tokenizer, added_tokens_decoder, kwargs

    def init_atom_2(self, **kwargs):
        """init atom 2"""
        truncation = self._tokenizer.truncation

        if truncation is not None:
            self._tokenizer.enable_truncation(**truncation)
            kwargs.setdefault("max_length", truncation["max_length"])
            kwargs.setdefault("truncation_side", truncation["direction"])
            kwargs.setdefault("stride", truncation["stride"])
            kwargs.setdefault("truncation_strategy", truncation["strategy"])
        else:
            self._tokenizer.no_truncation()

        padding = self._tokenizer.padding
        if padding is not None:
            self._tokenizer.enable_padding(**padding)
            kwargs.setdefault("pad_token", padding["pad_token"])
            kwargs.setdefault("pad_token_type_id", padding["pad_type_id"])
            kwargs.setdefault("padding_side", padding["direction"])
            kwargs.setdefault("max_length", padding["length"])
            kwargs.setdefault("pad_to_multiple_of", padding["pad_to_multiple_of"])
        return kwargs

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def can_save_slow_tokenizer(self) -> bool:
        """
        `bool`, Whether or not the slow tokenizer can be saved. Usually for sentencepiece based slow tokenizer, this
        can only be `True` if the original `"sentencepiece.model"` was not deleted.
        """
        return True

    @property
    def vocab_size(self) -> int:
        """
        `int`, Size of the base vocabulary (without the added tokens).
        """
        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab(with_added_tokens=True)

    @property
    def vocab(self) -> Dict[str, int]:
        return self.get_vocab()

    @property
    def added_tokens_encoder(self) -> Dict[str, int]:
        """
        Returns the sorted mapping from string to index. The added tokens encoder is cached for performance
        optimisation in `self._added_tokens_encoder` for the slow tokenizers.

        Returns:
            A dict, the added tokens.
        """
        return {k.content: v for v, k in sorted(self.added_tokens_decoder.items(), key=lambda item: item[0])}

    @property
    def added_tokens_decoder(self) -> Dict[int, AddedToken]:
        """
        Returns the added tokens in the vocabulary as a dictionary of index to AddedToken.

        Returns:
            A dict, the added tokens.
        """
        return self._tokenizer.get_added_tokens_decoder()

    def get_added_vocab(self) -> Dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index.

        Returns:
            A dict, the added tokens.
        """
        return {k.content: v for v, k in sorted(self.added_tokens_decoder.items(), key=lambda item: item[0])}

    def __len__(self) -> int:
        """
        Size of the full vocabulary with the added tokens.
        """
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def backend_tokenizer(self) -> TokenizerFast:
        """
        `tokenizers.implementations.PreTrainedTokenizerBase`, The Rust tokenizer used as a backend.
        """
        return self._tokenizer

    @property
    def decoder(self) -> DecoderFast:
        """
        `tokenizers.decoders.Decoder`, The Rust decoder for this tokenizer.
        """
        return self._tokenizer.decoder

    def _convert_encoding(
            self,
            encoding: EncodingFast,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
    ) -> Tuple[Dict[str, Any], List[EncodingFast]]:
        """
        Convert the encoding representation (from low-level HuggingFace tokenizer output) to a python Dict and a list
        of encodings, take care of building a batch from overflowing tokens.

        Overflowing tokens are converted to additional examples (like batches) so the output values of the dict are
        lists (overflows) of lists (tokens).

        Output shape: (overflows, sequence length)
        """
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        if return_overflowing_tokens and encoding.overflowing is not None:
            encodings = [encoding] + encoding.overflowing
        else:
            encodings = [encoding]

        encoding_dict = defaultdict(list)
        for e in encodings:
            encoding_dict["input_ids"].append(e.ids)

            if return_token_type_ids:
                encoding_dict["token_type_ids"].append(e.type_ids)
            if return_attention_mask:
                encoding_dict["attention_mask"].append(e.attention_mask)
            if return_special_tokens_mask:
                encoding_dict["special_tokens_mask"].append(e.special_tokens_mask)
            if return_offsets_mapping:
                encoding_dict["offset_mapping"].append(e.offsets)
            if return_length:
                encoding_dict["length"].append(len(e.ids))

        return encoding_dict, encodings

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (Union[str, List[str]]): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`, The token id or list of token ids.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        return [self._convert_token_to_id_with_added_voc(token) for token in tokens]

    def _convert_token_to_id_with_added_voc(self, token: str) -> int:
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index

    def _convert_id_to_token(self, index: int) -> Optional[str]:
        return self._tokenizer.id_to_token(int(index))

    def _add_tokens(self, new_tokens: List[Union[str, AddedToken]], special_tokens=False) -> int:
        if special_tokens:
            self.reset_special_tokens_cache()
            return self._tokenizer.add_special_tokens(new_tokens)

        return self._tokenizer.add_tokens(new_tokens)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        Note:
            This encodes a dummy input and checks the number of added tokens, and is therefore not efficient.
            Do not put this inside your training loop.

        Args:
            pair (bool, optional): Whether the number of added tokens should be computed in the case of
                a sequence pair or a single sequence. Default: ``False``.

        Returns:
            `int`, Number of special tokens added to sequences.
        """
        return self._tokenizer.num_special_tokens_to_add(pair)

    def convert_ids_to_tokens(
            self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens,
        using the vocabulary and added tokens.

        Args:
            ids (Union[int, List[int]]):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (bool, optional):
                Whether to remove special tokens in the decoding.  Default: ``False``.

        Returns:
            `str` or `List[str]`, The decoded token(s).
        """
        if isinstance(ids, int):
            return self._tokenizer.id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self._tokenizer.id_to_token(index))
        return tokens

    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        return self.encode_plus(text=text, text_pair=pair, add_special_tokens=add_special_tokens, **kwargs).tokens()

    def set_truncation_and_padding(
            self,
            padding_strategy: PaddingStrategy,
            truncation_strategy: TruncationStrategy,
            max_length: int,
            stride: int,
            pad_to_multiple_of: Optional[int],
    ):
        """
        Define the truncation and the padding strategies for fast tokenizers (provided by HuggingFace tokenizers
        library) and restore the tokenizer settings after.

        The provided tokenizer has no padding / truncation strategy before the managed section. If your tokenizer set a
        padding / truncation strategy before, then it will be reset to no padding / truncation when exiting the managed
        section.

        Args:
            padding_strategy (PaddingStrategy):
                The kind of padding that will be applied to the input
            truncation_strategy (TruncationStrategy):
                The kind of truncation that will be applied to the input
            max_length (int):
                The maximum size of a sequence.
            stride (int):
                The stride to use when handling overflow.
            pad_to_multiple_of (int, optional):
                If set will pad the sequence to a multiple of the provided value. This is especially useful
                to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
                Default: ``None``.
        """
        truncation = self._tokenizer.truncation
        padding = self._tokenizer.padding
        # Set truncation and padding on the backend tokenizer
        if truncation_strategy == TruncationStrategy.DO_NOT_TRUNCATE:
            if truncation is not None:
                self._tokenizer.no_truncation()
        else:
            target = {
                "max_length": max_length,
                "stride": stride,
                "strategy": truncation_strategy.value,
                "direction": self.truncation_side,
            }

            # _truncation might contain more keys that the target `transformers`
            # supports. Use only the target keys to trigger `enable_truncation`.
            # This should enable this code to works on various `tokenizers`
            # targets.
            if truncation is None:
                current = None
            else:
                current = {k: truncation.get(k, None) for k in target}

            if current != target:
                self._tokenizer.enable_truncation(**target)

        if padding_strategy == PaddingStrategy.DO_NOT_PAD:
            if padding is not None:
                self._tokenizer.no_padding()
        else:
            length = max_length if padding_strategy == PaddingStrategy.MAX_LENGTH else None
            target = {
                "length": length,
                "direction": self.padding_side,
                "pad_id": self.pad_token_id,
                "pad_token": self.pad_token,
                "pad_type_id": self.pad_token_type_id,
                "pad_to_multiple_of": pad_to_multiple_of,
            }
            if padding != target:
                self._tokenizer.enable_padding(**target)

    def _batch_encode_plus(
            self,
            batch_text_or_text_pairs: Union[
                List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair]
            ],
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[str] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs
    ) -> BatchEncoding:
        if not isinstance(batch_text_or_text_pairs, (tuple, list)):
            raise TypeError(
                f"batch_text_or_text_pairs has to be a list or a tuple (got {type(batch_text_or_text_pairs)})"
            )

        # Set the truncation and padding strategy and restore the initial configuration
        self.set_truncation_and_padding(
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
        )

        encodings = self._tokenizer.encode_batch(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            is_pretokenized=is_split_into_words,
        )

        # Convert encoding to dict
        # `Tokens` has type: Tuple[
        #                       List[Dict[str, List[List[int]]]] or List[Dict[str, 2D-Tensor]],
        #                       List[EncodingFast]
        #                    ]
        # with nested dimensions corresponding to batch, overflows, sequence length
        tokens_and_encodings = [
            self._convert_encoding(
                encoding=encoding,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
            )
            for encoding in encodings
        ]

        # Convert the output to have dict[list] from list[dict] and remove the additional overflows dimension
        # From (variable) shape (batch, overflows, sequence length) to ~ (batch * overflows, sequence length)
        # (we say ~ because the number of overflow varies with the example in the batch)
        #
        # To match each overflowing sample with the original sample in the batch
        # we add an overflow_to_sample_mapping array (see below)
        sanitized_tokens = {}
        for key in tokens_and_encodings[0][0].keys():
            stack = [
                e
                for item, _ in tokens_and_encodings
                for e in item[key]
            ]
            sanitized_tokens[key] = stack
        sanitized_encodings = [e
                               for _, item in tokens_and_encodings
                               for e in item]

        # If returning overflowing tokens, we need to return a mapping
        # from the batch idx to the original sample
        if return_overflowing_tokens:
            overflow_to_sample_mapping = []
            for i, (toks, _) in enumerate(tokens_and_encodings):
                overflow_to_sample_mapping += [i] * len(toks["input_ids"])
            sanitized_tokens["overflow_to_sample_mapping"] = overflow_to_sample_mapping

        for input_ids in sanitized_tokens.get("input_ids"):
            self._eventual_warn_about_too_long_sequence(input_ids, max_length, verbose)
        return BatchEncoding(sanitized_tokens, sanitized_encodings, tensor_type=return_tensors)

    def _encode_plus(
            self,
            text: Union[TextInput, PreTokenizedInput],
            text_pair: Optional[Union[TextInput, PreTokenizedInput]] = None,
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[bool] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs,
    ) -> BatchEncoding:
        batched_input = [(text, text_pair)] if text_pair else [text]
        batched_output = self._batch_encode_plus(
            batched_input,
            is_split_into_words=is_split_into_words,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

        # Return tensor is None, then we can remove the leading batch axis
        # Overflowing tokens are returned as a batch of output so we keep them in this case
        if return_tensors is None and not return_overflowing_tokens:
            batched_output = BatchEncoding(
                {
                    key: value[0] if value and isinstance(value[0], list) else value
                    for key, value in batched_output.items()
                },
                batched_output.encodings,
            )

        self._eventual_warn_about_too_long_sequence(batched_output["input_ids"], max_length, verbose)

        return batched_output

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.backend_tokenizer.decoder.decode(tokens)

    def _decode(
            self,
            token_ids: Union[int, List[int]],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = None,
            **kwargs,
    ) -> str:
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        if isinstance(token_ids, int):
            token_ids = [token_ids]
        text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        return text

    def _save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            file_names: Tuple[str],
            legacy_format: Optional[bool] = None,
            filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        """
        Save a tokenizer using the slow-tokenizer/legacy format: vocabulary + added tokens as well as in a unique JSON
        file containing {config + vocab + added-tokens}.
        """
        save_directory = str(save_directory)

        if self.slow_tokenizer_class is None and legacy_format is True:
            raise ValueError(
                "Your tokenizer does not have a legacy version defined and therefore cannot register this version. You"
                " might consider leaving the legacy_format at `None` or setting it to `False`."
            )

        save_slow = (
            (legacy_format is None or legacy_format is True)
            and self.slow_tokenizer_class is not None
            and self.can_save_slow_tokenizer
        )
        save_fast = legacy_format is None or legacy_format is False

        if save_slow:
            added_tokens_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + ADDED_TOKENS_FILE
            )
            # make sure to be forward compatible
            added_vocab = {tok: index for tok, index in self.added_tokens_encoder.items() if index >= self.vocab_size}
            if added_vocab:
                flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                with os.fdopen(os.open(added_tokens_file, flags_, FILE_PERMISSION), 'w', encoding="utf-8") as f:
                    out_str = json.dumps(added_vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                    f.write(out_str)

            vocab_files = self.save_vocabulary(save_directory, filename_prefix=filename_prefix)
            file_names = file_names + vocab_files + (added_tokens_file,)

        if save_fast:
            tokenizer_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_FILE
            )
            self.backend_tokenizer.save(tokenizer_file)
            file_names = file_names + (tokenizer_file,)

        return file_names

    def train_new_from_iterator(
            self,
            text_iterator,
            vocab_size,
            length=None,
            new_special_tokens=None,
            special_tokens_map=None,
            **kwargs,
    ):
        """
        Trains a tokenizer on a new corpus with the same defaults (in terms of special tokens or tokenization pipeline)
        as the current one.

        Args:
            text_iterator (list): The training corpus. Should be a generator of batches of texts,
                for instance a list of lists of texts if you have everything in memory.
            vocab_size (int): The size of the vocabulary you want for your tokenizer.
            length (int, optional): The total number of sequences in the iterator. This is used to provide meaningful
                progress tracking. Default: ``None``.
            new_special_tokens (Union[list, AddedToken], optional): A list of new special tokens to add to the
                tokenizer you are training. Default: ``None``.
            special_tokens_map (dict, optional): If you want to rename some of the special tokens this
                tokenizer uses, pass along a mapping old special token name to new special token name in this argument.
                Default: ``None``.
            kwargs (Any, optional): Additional keyword arguments.

        Returns:
            `PreTrainedTokenizerFast`, A new tokenizer of the same type as the original one, trained on
            `text_iterator`.

        """
        tokenizer_json = json.loads(self._tokenizer.to_str())
        # Remove added tokens for now (uses IDs of tokens)
        added_tokens = tokenizer_json.pop("added_tokens")
        # Remove post processor for now (uses IDs of tokens)
        post_processor = tokenizer_json.pop("post_processor")

        unk_token = None
        # Remove vocab
        if tokenizer_json["model"]["type"] == "BPE":
            tokenizer_json["model"]["vocab"] = {}
            tokenizer_json["model"]["merges"] = []
        elif tokenizer_json["model"]["type"] == "Unigram":
            if tokenizer_json["model"]["unk_id"] is not None:
                unk_id = tokenizer_json["model"]["unk_id"]
                unk_token = tokenizer_json["model"]["vocab"][unk_id][0]
                if special_tokens_map is not None and unk_token in special_tokens_map:
                    unk_token = special_tokens_map[unk_token]
                tokenizer_json["model"]["unk_id"] = 0
                tokenizer_json["model"]["vocab"] = [[unk_token, 0.0]]
        elif tokenizer_json["model"]["type"] in ["WordLevel", "WordPiece"]:
            tokenizer_json["model"]["vocab"] = {}
        else:
            raise ValueError(
                f"This method does not support this type of tokenizer (found {tokenizer_json['model']['type']}) "
                "only BPE, Unigram, WordLevel and WordPiece."
            )

        if (
                special_tokens_map is not None
                and "unk_token" in tokenizer_json["model"]
                and tokenizer_json["model"]["unk_token"] in special_tokens_map
        ):
            tokenizer_json["model"]["unk_token"] = special_tokens_map[tokenizer_json["model"]["unk_token"]]

        tokenizer = TokenizerFast.from_str(json.dumps(tokenizer_json))

        # Get the special tokens from the current tokenizer if none are specified.
        special_tokens = []
        for added_token in added_tokens:
            special = added_token.pop("special", None)
            _ = added_token.pop("id", None)
            if tokenizer_json["model"]["type"] != "Unigram" and not special:
                continue
            if special_tokens_map is not None and added_token["content"] in special_tokens_map:
                added_token["content"] = special_tokens_map[added_token["content"]]
            special_tokens.append(AddedToken(**added_token))

        if new_special_tokens is not None:
            special_tokens.extend(new_special_tokens)

        # Trainer needs to know the end of word / continuing subword thingies in BPE
        if (
                tokenizer_json["model"]["type"] == "BPE"
                and "continuing_subword_prefix" not in kwargs
                and tokenizer_json["model"]["continuing_subword_prefix"] is not None
        ):
            kwargs["continuing_subword_prefix"] = tokenizer_json["model"]["continuing_subword_prefix"]
        if (
                tokenizer_json["model"]["type"] == "BPE"
                and "end_of_word_suffix" not in kwargs
                and tokenizer_json["model"]["end_of_word_suffix"] is not None
        ):
            kwargs["end_of_word_suffix"] = tokenizer_json["model"]["end_of_word_suffix"]
        if tokenizer_json["model"]["type"] == "Unigram" and unk_token is not None:
            kwargs["unk_token"] = unk_token
        if tokenizer_json["pre_tokenizer"] is not None and tokenizer_json["pre_tokenizer"]["type"] == "ByteLevel":
            kwargs["initial_alphabet"] = pre_tokenizers_fast.ByteLevel.alphabet()

        trainer_class = MODEL_TO_TRAINER_MAPPING[tokenizer_json["model"]["type"]]
        trainer = trainer_class(vocab_size=vocab_size, special_tokens=special_tokens, **kwargs)
        tokenizer.train_from_iterator(text_iterator, length=length, trainer=trainer)

        if post_processor is not None:
            trained_tokenizer_json = json.loads(tokenizer.to_str())
            # Almost done, we just have to adjust the token IDs in the post processor
            if "special_tokens" in post_processor:
                for key in post_processor["special_tokens"]:
                    tokens = post_processor["special_tokens"][key]["tokens"]
                    if special_tokens_map is not None:
                        tokens = [special_tokens_map.get(token, token) for token in tokens]
                    post_processor["special_tokens"][key]["tokens"] = tokens
                    post_processor["special_tokens"][key]["ids"] = [tokenizer.token_to_id(token) for token in tokens]

            for special_token in ["cls", "sep"]:
                if special_token in post_processor:
                    token, _ = post_processor[special_token]
                    if special_tokens_map is not None and token in special_tokens_map:
                        token = special_tokens_map[token]
                    token_id = tokenizer.token_to_id(token)
                    post_processor[special_token] = [token, token_id]

            trained_tokenizer_json["post_processor"] = post_processor
            tokenizer = TokenizerFast.from_str(json.dumps(trained_tokenizer_json))

        kwargs = self.init_kwargs.copy()
        # Map pad/cls/mask token at the Transformers level
        special_tokens_list = SpecialTokensMixin.SPECIAL_TOKENS_ATTRIBUTES.copy()
        special_tokens_list.remove("additional_special_tokens")
        for token in special_tokens_list:
            # Get the private one to avoid unnecessary warnings.
            if getattr(self, f"_{token}") is not None:
                special_token = getattr(self, token)
                if special_tokens_map is not None and special_token in special_tokens_map:
                    special_token = special_tokens_map[special_token]

                special_token_full = getattr(self, f"_{token}")
                if isinstance(special_token_full, AddedToken):
                    # Create an added token with the same parameters except the content
                    kwargs[token] = AddedToken(
                        special_token,
                        single_word=special_token_full.single_word,
                        lstrip=special_token_full.lstrip,
                        rstrip=special_token_full.rstrip,
                        normalized=special_token_full.normalized,
                        special=True,
                    )
                else:
                    kwargs[token] = special_token

        additional_special_tokens = self.additional_special_tokens
        if new_special_tokens is not None:
            additional_special_tokens.extend(new_special_tokens)
        if additional_special_tokens:
            kwargs["additional_special_tokens"] = additional_special_tokens

        return self.__class__(tokenizer_object=tokenizer, **kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save only the vocabulary of the tokenizer (vocabulary + added tokens).

        This method won't save the configuration and special token mappings of the tokenizer. Use
        [`~PreTrainedTokenizerFast._save_pretrained`] to save the whole state of the tokenizer.

        Args:
            save_directory (str):
                The directory in which to save the vocabulary.
            filename_prefix (str, optional):
                An optional prefix to add to the named of the saved files. Default: ``None``.

        Returns:
            `Tuple(str)`, Paths to the files saved.
        """
        raise NotImplementedError
