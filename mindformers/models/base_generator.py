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
# ============================================================================

"""
TopK for text generation
"""
import logging
import copy

import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P

__all__ = ['GeneratorMinMax']

def topk_fun(logits, topk=5):
    """Get topk"""
    target_column = logits[0].tolist()
    sorted_array = [(k, v) for k, v in enumerate(target_column)]
    sorted_array.sort(key=lambda x: x[1], reverse=True)
    topk_array = sorted_array[:topk]
    index, value = zip(*topk_array)
    index = np.array([index])
    value = np.array([value])
    return value, index


def sampler(log_probs_revised, top_p, top_k, use_pynative=False):
    """Convert the log_probs to probability"""
    if use_pynative:
        logits = P.Pow()(np.e, Tensor(log_probs_revised, mstype.float32))
    else:
        logits = np.power(np.e, np.array(log_probs_revised, np.float32))

    # If top_p is less than 1.0, use top_p sampling
    if top_p < 1.0:
        # Only consider the 5000 largest logits to reduce computation
        if use_pynative:
            sorted_logits, index = P.TopK(sorted=True)(logits, 5000)
            cumsum_logits = P.CumSum()(sorted_logits, 1)
            cumsum_logits = cumsum_logits.asnumpy()
            index = index.asnumpy()
            sorted_logits = sorted_logits.asnumpy()
        else:
            sorted_logits, index = topk_fun(logits, 5000)
            cumsum_logits = np.cumsum(sorted_logits, 1)
        cumsum_logits = cumsum_logits[0]
        index = index[0]
        sorted_logits = sorted_logits[0]
        top_p_num = sum(cumsum_logits < top_p) + 1
        # In case the probability is smooth, the sum of 5000 largest probabilities are not large enough
        if top_p_num == 0:
            top_p_num = 5000
        # Get the corresponding probs and indices
        probs = sorted_logits[:top_p_num]
        p_args = index[:top_p_num]
        p = probs / sum(probs)
        # if top_p is set to 1.0, use top_k sampling
    else:
        # Get the corresponding probs and indices
        if use_pynative:
            probs, p_args = P.TopK(sorted=True)(logits, top_k)
            probs = probs.asnumpy()
            p_args = p_args.asnumpy()
        else:
            probs, p_args = topk_fun(logits, top_k)
        probs = probs[0]
        p_args = p_args[0]
        # Avoid rounding error
        if sum(probs) == 0:
            probs = np.array([1 / top_k for _ in range(top_k)])
        p = probs / sum(probs)
    return p, p_args


class GeneratorMinMax:
    """Generator For the nlp models"""
    def __init__(self):
        pass

    def _forward(self,
                 origin_inputs,
                 top_k,
                 top_p,
                 repetition_penalty,
                 max_length,
                 do_sample,
                 eos_token_id):
        """
        Text generation given the model and origin inputs

        Inputs:
            model: The model to run the prediction
            end_token(int): The model will stop generating the words when it reaches the end_token.
            origin_inputs(list): The prompt for generation, should be a list of ids.
            model_origin_max_length(int): The sequence length of the model trained.
            max_length(int):  The maximum of generated length.
            vocab_size(int): The vocabulary length of the model.
            config: Inference configurations.

        Returns:
            outputs: the ids for the generated text
        """
        # Get configurations for inference
        use_pynative = False
        is_encoder_decoder = self.config.is_encoder_decoder
        _, valid_length = origin_inputs.shape
        logging.info("The input shape is: %s", origin_inputs.shape)
        logging.info("Valid length is: %s", valid_length)
        # If target length exceeds model_max_length, use model_max_length instead
        target_length = valid_length + max_length

        target_length = self.config.seq_length if target_length > self.config.seq_length else target_length
        logging.info("target_length is: %s", target_length)
        # A list of the frequency of each token
        frequency_list = None
        pad_length = self.config.seq_length - origin_inputs.shape[-1]
        if pad_length < 0:
            raise ValueError(f"origin_inputs size is {origin_inputs.shape}, you should increase the "
                             f"seq_length of the model {self.config.seq_length}.")
        # Pad original inputs to model_origin_max_length
        input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, 0))
        logging.info("pad the origin inputs into shape: %s", input_ids.shape)
        input_mask = np.zeros_like(input_ids)
        input_mask[0][:valid_length] = 1
        encoder_output = None
        encoder_mask = None
        if is_encoder_decoder:
            if target_length > self.config.max_decode_length:
                target_length = self.config.max_decode_length
            logging.info("target_length is: %s", target_length)
            # When do encoder and decoder prediction, the encoder can be cached to speed up the inference
            inputs = Tensor(input_ids, mstype.int32)
            encoder_mask = copy.deepcopy(input_mask)
            encoder_output = self.construct(inputs, Tensor(encoder_mask, mstype.float32))
            input_ids = [[0]]
            input_ids = np.pad(input_ids, ((0, 0), (0, self.config.max_decode_length - 1)),
                               'constant', constant_values=(0, 0))

            logging.info("Decoder: pad the origin inputs into shape: %s", input_ids.shape)
            target_mask = np.zeros_like(input_ids)
            target_mask[0, 0] = 1
            # As the decoder is generating from [START] token
            valid_length = 1
        # A single loop generates one token, loop until reaching target model_origin_max_length or generating eod token
        while valid_length < target_length:
            inputs = Tensor(input_ids, mstype.int32)
            # Indicate the exact token position
            current_index = valid_length - 1 if valid_length - 1 > 0 else 0
            current_index = Tensor([current_index], mstype.int32)
            # Call a single inference
            if is_encoder_decoder:
                # view inputs as target_ids
                log_probs = self.construct(None, Tensor(encoder_mask, mstype.float32), current_index,
                                           encoder_output, inputs,
                                           Tensor(target_mask, mstype.float32))
            else:
                log_probs = self.construct(inputs, Tensor(input_mask, mstype.float32))
            # Get the revised log_probs considering frequency and presence penalty to eliminate duplicate
            # in generated results
            vocab_size = log_probs.shape[-1]
            if frequency_list is None:
                frequency_list = np.array([[0 for _ in range(vocab_size)]])
            log_probs_revised = log_probs.asnumpy().reshape(1, vocab_size)
            if repetition_penalty != 1:
                log_probs_revised = log_probs - frequency_list * repetition_penalty - \
                                    (frequency_list > 0) * repetition_penalty

            p, p_args = sampler(log_probs_revised, top_p, top_k, use_pynative)
            # Random select a token as final output for this round
            target_index = np.random.choice(len(p), p=p)
            # Stop judgment
            if p_args[target_index] == eos_token_id or valid_length == target_length - 1:
                outputs = input_ids
                break

            # update frequency list
            target = p_args[target_index]
            if not do_sample:
                frequency_list[0][target] = frequency_list[0][target] + 1
            # Modify input_ids with newly generated token
            input_ids[0][valid_length] = p_args[target_index]
            if is_encoder_decoder:
                target_mask[0][valid_length] = 1
            valid_length += 1
            input_mask[0][valid_length - 1] = 1
        # Return valid outputs out of padded outputs
        length = np.sum(outputs != 0)
        outputs = outputs[0][:length]
        return outputs

    def generate(self,
                 input_ids,
                 do_sample=None,
                 top_k=None,
                 top_p=None,
                 eos_token_id=None,
                 repetition_penalty=None,
                 max_length=None):
        """
        Generate the word given the input prompt, model and configs

        Args:


        Inputs:
            input_ids: the tokenized inputs
            attention_mask: the attention mask with [bs, seq_length]. 1 means effective and 0 mean it should be masked.
            labels:
        Returns:
            output: Tensor, the loss of the network
        """
        # Tokenize input sentence to ids
        input_ids = np.array(input_ids).reshape(1, -1)
        config = self.config
        top_p = config.top_p if top_p is None else top_p
        top_k = config.top_k if top_k is None else top_k
        repetition_penalty = config.repetition_penalty if repetition_penalty is None else repetition_penalty
        max_length = config.max_decode_length if max_length is None else max_length
        eos_token_id = config.eos_token_id if eos_token_id is None else eos_token_id
        do_sample = config.do_sample if do_sample is None else do_sample

        if not do_sample:
            top_p = 1
            top_k = 1
        # eval ops
        output_ids = self._forward(origin_inputs=input_ids,
                                   top_k=top_k,
                                   top_p=top_p,
                                   repetition_penalty=repetition_penalty,
                                   max_length=max_length,
                                   eos_token_id=eos_token_id,
                                   do_sample=do_sample)
        return output_ids
