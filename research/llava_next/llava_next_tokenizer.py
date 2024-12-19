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
"""llava tokenizer"""
import json
from typing import Optional, Dict, Any

from mindformers import MindFormerRegister, MindFormerModuleType
from mindformers.models.llama import LlamaTokenizer
from mindformers.models.tokenization_utils import AddedToken
from research.qwen2.qwen2_tokenizer import Qwen2Tokenizer

SUPPORTED_TOKENIZER = {
    "LlamaTokenizer": LlamaTokenizer,
    "Qwen2Tokenizer": Qwen2Tokenizer,
    "Qwen1.5Tokenizer": Qwen2Tokenizer,
}
CHAT_TEMPLATE = {
    "LlamaTokenizer": None,
    "Qwen2Tokenizer": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + "
                      "message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}"
                      "{{ '<|im_start|>assistant\n' }}{% endif %}",
    "Qwen1.5Tokenizer": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + "
                        "message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}"
                        "{{ '<|im_start|>assistant\n' }}{% endif %}"
}


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class LlavaNextTokenizer:
    r"""
    Construct a Llava-Next tokenizer. Based on byte-level Byte-Pair-Encoding. The default padding token is unset
    as there is no padding token in the original model.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.

        added_tokens_file (`str`, *optional*):
            Path to the additional tokens file
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be
            this token instead.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier
            token.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored
            by attention mechanisms or loss computation.
        image_tag (`str` or `tokenizers.AddedToken`, *optional*):
            A image token tag means the images tensor will fill in this position in input ids.
        sp_model_kwargs (`Dict[str, Any]`, `Optional`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other
            things to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add an `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of sequences.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
        spaces_between_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not to add spaces between special tokens.
        legacy (`bool`, *optional*):
            Whether or not the `legacy` behavior of the tokenizer should be used. Legacy is before the merge of
            #24622 and #25224 which includes fixes to properly handle tokens that appear after special tokens.
    """
    def __new__(
            cls,
            vocab_file,
            merges_file=None,
            added_tokens_file=None,
            tokenizer_type="LlamaTokenizer",
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            image_tag="<image>",
            sp_model_kwargs: Optional[Dict[str, Any]] = None,
            add_bos_token=True,
            add_eos_token=False,
            clean_up_tokenization_spaces=False,
            legacy=True,
            **kwargs,
    ):
        cls.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        if added_tokens_file is not None:
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                cls.added_tokens_dict = json.load(f)
            added_tokens_decoder = {}
            for k, v in cls.added_tokens_dict.items():
                added_tokens_decoder[v] = AddedToken(k, lstrip=False, rstrip=False) if isinstance(k, str) else k
                kwargs["added_tokens_decoder"] = added_tokens_decoder

        cls._instance = SUPPORTED_TOKENIZER[tokenizer_type](vocab_file=vocab_file,
                                                            merges_file=merges_file,
                                                            bos_token=bos_token,
                                                            eos_token=eos_token,
                                                            unk_token=unk_token,
                                                            pad_token=pad_token,
                                                            sp_model_kwargs=cls.sp_model_kwargs,
                                                            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                                                            add_bos_token=add_bos_token,
                                                            add_eos_token=add_eos_token,
                                                            legacy=legacy,
                                                            chat_template=CHAT_TEMPLATE[tokenizer_type],
                                                            **kwargs)
        cls._instance.image_token = image_tag
        if len(cls._instance.tokenize(image_tag)) != 1:
            cls._instance.add_tokens(AddedToken(image_tag, lstrip=False, rstrip=False, special=False),
                                     special_tokens=True)
        cls._instance.img_token_id = cls._instance.convert_tokens_to_ids(image_tag)

        return cls._instance
