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
from .base_handler import BaseInstructDataHandler


class AlpacaInstructDataHandler(BaseInstructDataHandler):
    """
    Alpaca Instruct Data Handler for formatting and tokenizing instruction-based datasets.

    This handler applies a conversational template to each example, distinguishing
    between user instructions and assistant responses, then encodes the text into
    token IDs for model training.

    Attributes:
        system (str): System prompt describing the AI assistant behavior.
        prompt_with_input (str): Template for examples that include an additional input context.
        prompt_without_input (str): Template for examples without extra input context.
        output_columns (list[str]): Columns to output after processing ('input_ids', 'labels').
    """

    system = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )

    prompt_with_input = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    )

    prompt_without_input = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )

    output_columns = ['input_ids', 'labels']

    def __init__(self, seq_length, padding=True, **kwargs):
        self.seq_length = seq_length
        self.padding = padding
        super().__init__(**kwargs)

    def __call__(self, dataset):
        """
        Process a dataset by applying the template and tokenizing each example.

        Args:
            dataset (Dataset): Hugging Face dataset to process.

        Returns:
            Dataset: Processed dataset with columns 'input_ids' and 'labels'.
        """
        columns = next(iter(dataset)).keys()
        remove_columns = list(set(columns) - set(self.output_columns))
        dataset = dataset.map(self.process, remove_columns=remove_columns)
        return dataset

    def process(self, example):
        """Apply template and tokenize example"""
        example = self.apply_template(example)
        example = self.tokenize_example(example)
        return example

    def apply_template(self, example):
        """Apply alpaca instruct template on samples"""
        if example.get('input'):
            usr_input = self.prompt_with_input.format_map(example)
        else:
            usr_input = self.prompt_without_input.format_map(example)
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

    def tokenize_example(self, example):
        """Encode source text with tokenizer"""
        text = f"{self.system} "
        token = self.tokenizer(text)['input_ids']

        # ignore_token_id = -100 default
        labels = [self.ignore_token_id] * len(token)

        for message in example:
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

        token, labels = self._process_sequence_with_length(token, labels)
        return {
            "input_ids": np.array(token, dtype=np.int32),
            "labels": np.array(labels, dtype=np.int32)
        }

    def _process_sequence_with_length(self, input_ids, labels):
        """
        Process input and label sequences to ensure they match the required maximum length.
        Handles optional padding and truncation.
        """
        # Determine the maximum sequence length
        max_length = self.seq_length
        if self.use_legacy:
            max_length += 1

        if self.padding and len(input_ids) < max_length:
            input_ids += [self.pad_token_id] * (max_length - len(input_ids))
            labels += [self.ignore_token_id] * (max_length - len(labels))

        # Truncate sequences to maximum length
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        return input_ids, labels
