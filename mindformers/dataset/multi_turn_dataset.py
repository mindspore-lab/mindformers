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
"""Multi-turn Dataset."""
import os
import sys
import json
import ast
from copy import deepcopy
from typing import Dict, List

import astunparse
import numpy as np

from mindformers.models.tokenization_utils_base import PreTrainedTokenizerBase
from mindformers.dataset.base_dataset import BaseDataset
from mindformers.dataset.dataloader import build_dataset_loader
from mindformers.models.build_tokenizer import build_tokenizer
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister
from mindformers.version_control import get_dataset_map, is_version_python

# text constants
FUNCTION_CALL_NAME = 'tool_call'
FUNCTION_CALL_PREFIX = '```python\n'
FUNCTION_CALL_POSTFIX = '\n```'
TOOL_DEFINITION_PREFIX = 'Answer the following questions as best as you can. You have access to the following tools:\n'
CONVERSATOIN_KEY = 'conversations'
TOOL_DESC_KEY = 'tools'

@MindFormerRegister.register(MindFormerModuleType.DATASET)
class MultiTurnDataset(BaseDataset):
    """
    Multi-turn dataset.

    Args:
        dataset_config (dict): Config for dataset.

    Returns:
        A dataset for MultiTurnDataset.

    Examples:
        >>> from mindformers import MultiTurnDataset
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers import MindFormerBook
        >>> from mindformers.dataset import build_dataset, check_dataset_config
        >>> config_dict_list = MindFormerBook.get_trainer_support_task_list()
        >>> config_path = config_dict_list['text_generation']['glm3_6b']
        >>> # Initialize a MindFormerConfig instance with a specific config file of yaml.
        >>> config = MindFormerConfig(config_path)
        >>> config.train_dataset.data_loader.dataset_dir = "The required task dataset path"
        >>> # Note:
        >>> #     The detailed data setting could refer to
        >>> #     https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/glm3.md
        >>> check_dataset_config(config)
        >>> # use class to build dataset
        >>> dataset_from_class = MultiTurnDataset(config.train_dataset_task.dataset_config)
    """

    def __new__(cls, dataset_config: dict = None):
        logger.info("Now Create Multi-turn Dataset.")
        assert is_version_python(sys.version, "3.9"), \
               f"MultiTurnDataset needs python3.9 or larter, please upgrade your python."

        cls.init_dataset_config(dataset_config)

        rank_id = int(os.getenv("RANK_ID", "0"))
        device_num = int(os.getenv("RANK_SIZE", "1"))
        rank_id, device_num = cls._check_device_rank_for_parallel(rank_id, device_num)
        dataset_config.rank_id = rank_id
        dataset_config.device_num = device_num

        if isinstance(dataset_config.tokenizer, PreTrainedTokenizerBase):
            cls.tokenizer = dataset_config.tokenizer
        else:
            cls.tokenizer = build_tokenizer(dataset_config.tokenizer)

        dataset_dir = dataset_config.data_loader.pop("dataset_dir")
        shuffle = dataset_config.data_loader.pop("shuffle")
        if not os.path.isfile(dataset_dir):
            raise ValueError(f"{dataset_dir} is not existed.")

        dataset = build_dataset_loader(dataset_config.data_loader,
                                       default_args={'dataset_dir': dataset_dir,
                                                     'shuffle': shuffle})
        dataset = cls._tokenizer_map(dataset, dataset_config)

        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                num_parallel_workers=dataset_config.num_parallel_workers,
                                python_multiprocessing=dataset_config.python_multiprocessing)
        dataset = dataset.repeat(dataset_config.repeat)

        return dataset

    @classmethod
    def _tokenizer_map(cls, dataset, dataset_config):
        """Maps the tokenizer on the source and the output"""

        if isinstance(dataset_config.tokenizer, PreTrainedTokenizerBase):
            tokenizer = dataset_config.tokenizer
        else:
            tokenizer = build_tokenizer(dataset_config.tokenizer)

        train_dataset_function = cls._train_dataset_function
        input_columns = ["data"]
        train_output_columns = ["input_ids", "labels"]

        # Avoid to_json error when summary monitor is opened
        def train_dataset_func(data):
            return train_dataset_function(data, dataset_config, tokenizer)

        dataset = get_dataset_map(dataset,
                                  train_dataset_func,
                                  input_columns=input_columns,
                                  output_columns=train_output_columns)
        dataset = dataset.project(columns=train_output_columns)

        return dataset

    @classmethod
    def _format_function_call(cls, function_name: str, parameters: Dict[str, str]):
        """format function call"""
        function_name = ast.Name(id=function_name)
        keywords = [
            ast.keyword(arg=arg_name, value=ast.Constant(arg_value))
            for arg_name, arg_value in parameters.items()
        ]
        func_call = ast.Call(func=function_name, args=[], keywords=keywords)

        return astunparse.unparse(func_call).strip()

    @classmethod
    def _format_conversation(cls, item, tokenizer, conversation_key: str, tool_key: str):
        """format_conversation"""
        conversations = deepcopy(item[conversation_key])

        # Note: `loss_mask` here means whether *the prediction* of the token should take loss
        tokens, loss_masks = [tokenizer.get_command("[gMASK]"), tokenizer.get_command("sop")], [0, 0]

        def _update(conv_tokens: List[int], value: int = 1):
            value = int(value)
            tokens.extend(conv_tokens)
            loss_masks.extend([value] * len(conv_tokens))

        # insert system prompt for tools
        if tool_key in item:
            conversations.insert(0, {"role": "system",
                                     "content": TOOL_DEFINITION_PREFIX + \
                                     json.dumps(item[tool_key], ensure_ascii=False)})

        for _, conv in enumerate(conversations):

            loss = conv.get("loss", True)
            if conv['role'] in {'system', 'user'}:
                loss = False
            if conv['role'] == 'tool':
                value = FUNCTION_CALL_PREFIX + cls._format_function_call(FUNCTION_CALL_NAME, conv["parameters"]) + \
                        FUNCTION_CALL_POSTFIX
                text = tokenizer.build_single_message("assistant", conv["name"], value)
                _update(text, loss)

                # function call result
                value = conv.get('observation', None)
                if not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)
                text = tokenizer.build_single_message("observation", "", value)
                _update(text, False)
            else:
                text = tokenizer.build_single_message(conv['role'], "", conv["content"])
                _update(text, loss)

        _update([tokenizer.eos_token_id], False)

        assert len(tokens) == len(loss_masks), f"length mismatch: {len(tokens)} vs {len(loss_masks)}"

        return tokens, loss_masks

    @classmethod
    def _train_dataset_function(cls, data, dataset_config, tokenizer):
        """generates train dataset"""
        max_seq_length = dataset_config.max_seq_length
        tokens, loss_masks = cls._format_conversation(data, tokenizer, CONVERSATOIN_KEY, TOOL_DESC_KEY)

        # labels are used inside the model
        target_based_loss_mask = [False] + loss_masks[:-1]
        labels = [(t if m else -100) for t, m in zip(tokens, target_based_loss_mask)]

        # cut input_ids to max_seq_length
        input_ids = tokens[:max_seq_length]
        # cut labels to max_seq_length
        labels = labels[1:max_seq_length]
        # pad input_ids to max_seq_length
        input_ids += [tokenizer.pad_token_id] * (max_seq_length - len(input_ids))
        # pad labels to max_seq_length
        labels += [-100] * (max_seq_length - len(labels))

        assert len(input_ids) == len(labels), f"length mismatch: {len(input_ids)} vs {len(labels)}"

        input_ids = np.array(input_ids, dtype=np.int32)
        labels = np.array(labels, dtype=np.int32)

        return input_ids, labels
