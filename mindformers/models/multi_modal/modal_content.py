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

"""
ModalContent Tools
"""
import abc
from typing import Dict, Any, List

import numpy as np
import mindspore as ms
from mindspore import Tensor, ops

from mindformers.tools.logger import logger
from .utils import DataRecord


class ModalContentBuilder:
    """
    Base class of builder for modal content. Modal content builder processes the input into what the model expects in
    different phase.
    """
    def __init__(
            self, type_, context_pad_token,
            context_length,
            use_custom_token=True,
            start_token=None,
            end_token=None,
            tokenizer=None,
            need_create_context_pos=True
    ):
        self.type = type_
        self.context_pad_token = context_pad_token
        self.context_length = context_length
        self.use_custom_token = use_custom_token

        if not self.use_custom_token and (not start_token or not end_token) and type_ != "text":
            raise ValueError("start_token and end_token must be set when use_custom_token is False")

        if self.use_custom_token and not start_token:
            start_token = f"<|MF_CUSTOM_{type_.upper()}_START|>"

        if self.use_custom_token and not end_token:
            end_token = f"<|MF_CUSTOM_{type_.upper()}_END|>"

        self.start_token = start_token
        self.end_token = end_token

        self.tokenizer = tokenizer
        self.need_create_context_pos = need_create_context_pos

        self.has_add_custom_token = False
        self.start_token_id = None
        self.end_token_id = None
        self.context_pad_token_id = None

    @abc.abstractmethod
    def regular_input_for_predict(self, inputs, result_recorder: DataRecord = None, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def regular_input_for_train(self, inputs, result_recorder: DataRecord = None, **kwargs):
        raise NotImplementedError

    # pylint: disable=W0613
    def padding_context(self, input_ids, result_recorder: DataRecord = None, **kwargs):
        """padding context pad token id into the text token ids"""
        start_position = np.where(input_ids == self.start_token_id)[0]
        offset = 0
        for start_position_item in start_position:
            start_position_item = start_position_item + 1 + offset

            input_ids = np.insert(input_ids, start_position_item, [self.context_pad_token_id] * self.context_length)
            offset += self.context_length

        if self.use_custom_token:
            input_ids = np.delete(input_ids, np.where(input_ids == self.start_token_id))
            input_ids = np.delete(input_ids, np.where(input_ids == self.end_token_id))

        return input_ids

    def generate_context_positions(self, input_ids, batch_index=0, result_recorder: DataRecord = None, **kwargs):
        pos = np.where(np.array(input_ids) == self.context_pad_token_id)[0]
        pos = np.expand_dims(pos, axis=0)
        pos = np.insert(pos, 0, batch_index, axis=0)
        pos = np.transpose(pos).reshape((-1, self.context_length, 2))
        return Tensor(pos, dtype=ms.int32)


class BaseTextContentBuilder(ModalContentBuilder):
    """
    Base text modal builder. It returns the original input.
    """

    def __init__(self):
        super().__init__(type_="text",
                         context_pad_token="",
                         context_length=-1,
                         use_custom_token=False,
                         start_token="",
                         end_token="",
                         need_create_context_pos=False)

    def regular_input_for_predict(self, inputs: Dict, result_recorder: DataRecord = None, **kwargs):
        if "text" in inputs:
            return inputs["text"]
        return ""

    def regular_input_for_train(self, inputs, result_recorder: DataRecord = None, **kwargs):
        raise NotImplementedError


class ModalContentTransformTemplate:
    """
    Base class of modal content transform template. It should be implemented by the specific model. The child class can
    override the methods `build_conversion_input_text`, `update_result_before_output`, `batch`, `post_process` to
    achieve the model's expectations

    Args:
        output_columns(List[str]): Specify which columns will be output.
        tokenizer: tokenizer.
        mode(str): running mode, predict or train.
        vstack_columns(List[str]): Specify which columns will be vstack when batching data.
    """
    _DEFAULT_OUTPUT_COLUMNS = ["input_ids"]

    # pylint: disable=W0613
    def __init__(self, output_columns: List[str] = None, tokenizer=None, mode="predict",
                 vstack_columns: List[str] = None, **kwargs):
        if output_columns is None:
            self.output_columns = self._DEFAULT_OUTPUT_COLUMNS
        else:
            self.output_columns = output_columns

        self.tokenizer = tokenizer
        self.mode = mode

        if vstack_columns is not None:
            new_vstack_columns = [column for column in vstack_columns if column in self.output_columns]

            if len(new_vstack_columns) != len(vstack_columns):
                logger.warning("there are columns in vstack_columns not in output_columns, it will be ignored.")
            vstack_columns = new_vstack_columns
        else:
            vstack_columns = []
        self.vstack_columns = vstack_columns

        self.history = []
        self.modal_builders: Dict[str, ModalContentBuilder] = {}
        self._supported_modal = []

        self.has_init_modal_builder_tokens = False
        if self.tokenizer is not None:
            self.check_modal_builder_tokens(self.tokenizer)

        self.tensor_stack = ops.Stack(axis=0)
        self.tensor_vstack = ops.vstack

    def process_predict_query(self, query_ele_list: List[Dict], result_recorder: DataRecord):
        """Find the corresponding modal builder by traversing and process it"""
        text_list = []
        for query_ele in query_ele_list:
            modal_type = ""
            for supported_modal_type in self.supported_modal:
                if supported_modal_type in query_ele:
                    modal_type = supported_modal_type
                    break

            if not modal_type:
                logger.warning(f"The query {query_ele} is not supported, it will be ignore.")
                continue

            text_list.append(
                self.modal_builders[modal_type].regular_input_for_predict(inputs=query_ele,
                                                                          result_recorder=result_recorder))
        return text_list

    @property
    def supported_modal(self):
        if not self._supported_modal:
            self._supported_modal = [modal_builder.type for modal_builder in self.modal_builders.values()]
        return self._supported_modal

    @abc.abstractmethod
    def build_conversion_input_text(self, raw_inputs, result_recorder: DataRecord):
        if self.mode == "predict":
            return "".join(raw_inputs)
        raise ValueError(f"building {self.mode} mode conversion inputs is not supported.")

    def build_modal_context(self, input_ids, result_recorder: DataRecord, **kwargs):
        if isinstance(input_ids, list):
            input_ids = np.array(input_ids)

        for modal_builder in self.modal_builders.values():
            input_ids = modal_builder.padding_context(input_ids, result_recorder, **kwargs)
        return input_ids

    def generate_modal_context_positions(self, input_ids, batch_index: int = 0, **kwargs):
        context_positions = {}
        for modal_builder in self.modal_builders.values():
            if not modal_builder.need_create_context_pos:
                continue

            context_pos = modal_builder.generate_context_positions(input_ids, batch_index=batch_index, **kwargs)
            context_positions[f"{modal_builder.type}_context_pos"] = context_pos
        return context_positions

    def check_modal_builder_tokens(self, tokenizer):
        """check modal builder tokens status, it will be assigned when not inited."""
        for modal_builder in self.modal_builders.values():
            if modal_builder.use_custom_token:
                tokenizer.add_tokens([modal_builder.start_token, modal_builder.end_token], special_tokens=True)

            if modal_builder.type == "text":
                continue

            if modal_builder.start_token_id is None:
                modal_builder.start_token_id = \
                    tokenizer([modal_builder.start_token], add_special_tokens=False)["input_ids"][0][0]

            if modal_builder.end_token_id is None:
                modal_builder.end_token_id = \
                    tokenizer([modal_builder.end_token], add_special_tokens=False)["input_ids"][0][0]

            if modal_builder.context_pad_token_id is None:
                modal_builder.context_pad_token_id = \
                    tokenizer([modal_builder.context_pad_token], add_special_tokens=False)["input_ids"][0][0]

            if modal_builder.tokenizer is None:
                modal_builder.tokenizer = tokenizer
            self.has_init_modal_builder_tokens = True

    def get_need_update_output_items(self, result: DataRecord) -> Dict[str, Any]:
        update_items = {}
        return update_items

    def batch_input_ids(self, input_ids_list, max_length):
        """batch the input_ids_list with max_length, if the length of input_ids is greater than max_length,
        it will be truncated to the max_length"""
        padded_input_ids = []
        for input_ids in input_ids_list:
            if len(input_ids) < max_length:
                input_ids = np.pad(input_ids, (0, max_length - len(input_ids)), "constant",
                                   constant_values=self.tokenizer.pad_token_id)
            else:
                input_ids = input_ids[:max_length]

            padded_input_ids.append(input_ids)
        return np.stack(padded_input_ids, axis=0)

    def stack_data(self, data, need_vstack: bool = False):
        """stack data"""
        if isinstance(data[0], Tensor):
            if need_vstack:
                stacked_data = self.tensor_vstack(data)
            else:
                stacked_data = self.tensor_stack(data)
        else:
            if need_vstack:
                stacked_data = np.vstack(data)
            else:
                stacked_data = np.stack(data, axis=0)
        return stacked_data

    def try_to_batch(self, data_list, column_name):
        """try to batch data list"""
        batched_data = data_list
        need_vstack = column_name in self.vstack_columns
        try:
            batched_data = self.stack_data(data_list, need_vstack)
            batch_result = True
        # pylint: disable=W0703
        except Exception as e:
            logger.warning("batching %s failed, Error: %s.", column_name, e)
            batch_result = False

        return batch_result, batched_data

    def batch(self, data_list, token_padding_length, **kwargs):
        """bath the column data in the output_names"""
        batched_data = {}
        for column_name in self.output_columns:
            column_data_list = [data[column_name] for data in data_list if
                                data[column_name] is not None and len(data[column_name]) > 0]

            if column_name == "input_ids":
                batched_data[column_name] = self.batch_input_ids(column_data_list, token_padding_length)
                continue

            if column_data_list:
                batch_success, batched_column_data = self.try_to_batch(column_data_list, column_name)
                batched_data[column_name] = batched_column_data
                if not batch_success:
                    logger.warning("batching %s failed, try to create a separate process for this data.", column_name)
            else:
                logger.warning(f"the data {column_name} is empty when batching data, please check the query data.")
        return batched_data

    def post_process(self, output_ids, **kwargs):
        skip_special_tokens = kwargs.get("skip_special_tokens", True)
        output = []
        for output_ids_item in output_ids:
            output.append(self.tokenizer.decode(output_ids_item, skip_special_tokens))
        return output
