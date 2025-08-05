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
"""Alpaca Dataset Handler."""

import numpy as np

from mindformers.tools.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.logger import logger
from .base_handler import BaseInstructDataHandler, BaseTemplate


class AlpacaTemplate(BaseTemplate):
    """Alpaca Conv Template."""
    end_token = "\n"
    input_token = "### Input:"
    system = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )


PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)

PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


@MindFormerRegister.register(MindFormerModuleType.DATA_HANDLER)
class AlpacaInstructDataHandler(BaseInstructDataHandler):
    """Alpaca Data Handler"""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.template = AlpacaTemplate()
        self.padding = self.packing is None and not self.is_dynamic

        if self.tokenizer is not None and getattr(self.tokenizer, 'pad_token_id', None) is not None:
            self.pad_token_id = self.tokenizer.pad_token_id
        else:
            logger.info(f"tokenizer not set or it have no pad_token_id, set 0 as `pad_token_id`.")
            self.pad_token_id = kwargs.get('pad_token_id', 0)

    def format_func(self, example):
        """Apply alpaca instruct template on samples"""
        if example.get('input'):
            usr_input = PROMPT_INPUT.format_map(example)
        else:
            usr_input = PROMPT_NO_INPUT.format_map(example)
        assis_response = example.get('output', '')
        example = [
            {
                "from": "USER",
                "value": usr_input,
            },
            {
                "from": "ASSISTANT",
                "value": assis_response,
            },
        ]
        return example

    def tokenize_func(self, messages):
        """Encode source text with tokenizer"""
        text = f"{self.template.system} "
        token = self.tokenizer(text)['input_ids']

        # ignore_token_id = -100 default
        labels = [self.ignore_token_id] * len(token)

        for message in messages:
            role = message.get("from")
            value = message.get("value")

            if role == "USER":
                cur_text = f"{role}: {value} "
                cur_token = self.tokenizer(cur_text)['input_ids']
                cur_labels = [self.ignore_token_id] * len(cur_token)
            else:
                cur_text = f"{role}: {value}</s>"  # last token is </s>
                cur_token = self.tokenizer(cur_text)['input_ids']
                cur_labels = cur_token

            text += cur_text
            token += cur_token
            labels += cur_labels

        labels = labels[1:] + [self.ignore_token_id]

        max_length = self.seq_length
        if self.use_legacy:
            max_length += 1

        if self.padding and len(token) < max_length:
            token += [self.pad_token_id] * (max_length - len(token))
            labels += [self.ignore_token_id] * (max_length - len(labels))

        token = token[:max_length]
        labels = labels[:max_length]

        return {
            "input_ids": np.array(token, dtype=np.int32),
            "labels": np.array(labels, dtype=np.int32)
        }
