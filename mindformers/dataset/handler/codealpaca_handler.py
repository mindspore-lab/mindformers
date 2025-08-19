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
"""Code Alpaca Dataset Handler."""

import numpy as np
from .alpaca_handler import AlpacaInstructDataHandler


class CodeAlpacaInstructDataHandler(AlpacaInstructDataHandler):
    """Code Alpaca Data Handler"""

    prompt = (
        "Below is an instruction that describes a task. Write a response that appropriately "
        "completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    )

    def apply_template(self, example):
        """Apply alpaca instruct template on samples"""
        usr_input = self.prompt.format_map(example)
        assis_response = example.get('output')
        example = [
            {"from": "USER", "value": usr_input},
            {"from": "ASSISTANT", "value": assis_response}
        ]
        return example

    def tokenize_example(self, example):
        """Encode source text with tokenizer"""
        input_ids = []
        labels = []
        for message in example:
            role = message.get("from")
            value = message.get("value")

            tokens = self.tokenizer(value)['input_ids']
            input_ids.extend(tokens)
            if role == "USER":
                labels.extend(len(tokens) * [self.ignore_token_id])
            else:
                labels.extend(tokens)

        input_ids, labels = self._process_sequence_with_length(input_ids, labels)
        return {
            "input_ids": np.array(input_ids, dtype=np.int32),
            "labels": np.array(labels, dtype=np.int32)
        }
