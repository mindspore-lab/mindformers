# Copyright 2023 Huawei Technologies Co., Ltd
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
"""visualglm text generator adaptor for lite inference."""
import time
from typing import List, Union, Optional

import numpy as np
from mindspore_lite import Model

from mindformers.generation.streamers import BaseStreamer
from mindformers.inference import InferTask
from mindformers.inference.infers.text_generator_infer import TextGeneratorInfer, InputOfInfer, BaseInputsOfInfer


def register_task():
    """register task for visualglm. """
    InputOfInfer.MAPPING["visualglm"] = VisualGlmInputsOfInfer
    InferTask.task_mapping["visualglm_generation"] = VisualGlmGeneratorInfer


class VisualGlmInputsOfInfer(BaseInputsOfInfer):
    """
    VisualGlmInputsOfInfer
    """

    def get_inputs(self, model: Model, input_ids=None, input_embeddings=None, current_index=None, valid_length=None,
                   init_reset=None, tokenizer=None, is_first_iteration=True,
                   attention_mask=None, position_ids=None, **kwargs):
        """
        get input for lite
        """
        del tokenizer, kwargs
        position_ids = position_ids.asnumpy()
        attention_mask = attention_mask.asnumpy()
        if not is_first_iteration:
            inputs_tmp = []
            position_ids_tmp = []
            attention_mask_tmp = []
            for i in range(len(current_index)):
                current_index_tmp = int(current_index[i]) - i * input_ids.shape[1]  # multibatch
                # use numpy to slice array to avoid complie ascend slice op
                inputs_tmp.append(input_ids[i][current_index_tmp:current_index_tmp + 1])
                position_ids_tmp.append(position_ids[i][:, current_index_tmp:current_index_tmp + 1])
                attention_mask_tmp.append(attention_mask[i][:, current_index_tmp:current_index_tmp + 1, :])

            input_ids = np.array(inputs_tmp, np.int32)
            position_ids = np.array(position_ids_tmp, np.int32)
            attention_mask = np.array(attention_mask_tmp, np.int32)
            inputs = [input_ids, None, position_ids, attention_mask,
                      current_index, init_reset, valid_length]
        else:
            input_embeddings = input_embeddings.asnumpy()
            inputs = [input_embeddings, input_ids, None, position_ids, attention_mask,
                      current_index, init_reset, valid_length]

        lite_inputs = self.get_lite_tensor_list(inputs, model)
        return lite_inputs


class VisualGlmGeneratorInfer(TextGeneratorInfer):
    """
    VisualGlm generator infer implement class.
    """

    # pylint: disable=W0221

    def infer(self,
              inputs: Optional[Union[List[int], List[List[int]]]],
              do_sample: bool = False,
              top_k: int = 1,
              top_p: float = 1.0,
              temperature: float = 1.0,
              repetition_penalty: float = 1.0,
              eos_token_id: int = 2,
              pad_token_id: int = 0,
              max_length: int = 256,
              is_sample_acceleration: bool = False,
              add_special_tokens: bool = False,
              streamer: Optional[BaseStreamer] = None,
              **kwargs):
        """
        text generator inference api

        Args:
            inputs(List(str), List(List(str))): The token id list or a list of token id list.
            do_sample(bool): Whether to do sampling on the candidate ids.
                If set True it will be enabled, and set it to be False to disable the sampling, equivalent to topk 1.
                If set None, it follows the setting in the configureation in the model. Default None.
            top_k(int): Determine the topK numbers token id as candidate. This should be a positive number.
                If set None, it follows the setting in the configureation in the model. Default 1.
            top_p(float): The accumulation probability of the candidate token ids below the top_p will be select as the
                condaite ids. The valid value of top_p is between (0, 1]. If the value is larger than 1,
                top_K algorithm will be enabled. If set None, it follows the setting in the configureation in the model.
                Default 1.
            temperature (`float`, *optional*, defaults to 1.0): The value used to modulate the next token probabilities.
            eos_token_id(int): The end of sentence token id. If set None, it follows the setting in the configureation
                in the model. Default 2.
            pad_token_id(int): The padding of sentence token id. If set None, it follows the setting in the
                configureation in the model. Default 0.
            repetition_penalty(float): The penalty factor of the frequency that generated words. The If set 1,
                the repetition_penalty will not be enabled. If set None, it follows the setting in the configureation in
                the model. Default 1.
            max_length: The maximum length of the generated words. If set None, it follows the setting in the
                configureation in the model. Default 256.
            is_sample_acceleration: The postprocess are processing in model. Default False.
            add_special_tokens: Add special tokens for preprocess.
            streamer: The streamer that generator uses.

        Returns:
            outputs of model infer
        """
        del add_special_tokens
        start_generate_time = time.time()
        output_ids = self.generate(inputs, do_sample, top_k, top_p, temperature,
                                   repetition_penalty, eos_token_id, pad_token_id,
                                   max_length, is_sample_acceleration, streamer, **kwargs)

        generate_time = time.time() - start_generate_time
        outputs = self.postprocess(output_ids)
        return outputs, output_ids, generate_time
