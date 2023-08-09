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

"""
For text generation
"""
import copy
from typing import Optional, List, Union

import numpy as np
from mindspore.ops import functional as F
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindformers.generation.streamers import BaseStreamer

from .generation_config import GenerationConfig
from .streamers import BaseStreamer
from .utils import softmax
from ..tools import logger

__all__ = ["GeneratorMixin"]


def topk_fun(logits, topk=5):
    """Get topk"""
    batch_value = []
    batch_index = []
    for i in range(logits.shape[0]):
        target_column = logits[i].tolist()
        sorted_array = [(k, v) for k, v in enumerate(target_column)]
        sorted_array.sort(key=lambda x: x[1], reverse=True)
        topk_array = sorted_array[:topk]
        index, value = zip(*topk_array)
        batch_value.append(value)
        batch_index.append(index)
    return np.array(batch_value), np.array(batch_index)


def batch_select(data, index):
    """bathc operation to sorted_logits[:, :top_p_num]"""
    output = []
    for i in range(data.shape[0]):
        res = data[i, :index[i]]
        output.append(res.reshape(1, -1))
    return np.concatenate(output, 0)


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
        cumsum_logits = cumsum_logits
        index = index
        sorted_logits = sorted_logits
        top_p_num = np.sum(cumsum_logits < top_p, axis=-1) + 1
        # Get the corresponding probs and indices
        probs = batch_select(sorted_logits, top_p_num)
        p_args = batch_select(index, top_p_num)
        p = probs / np.sum(probs, -1, keepdims=True)
        # if top_p is set to 1.0, use top_k sampling
    else:
        # Get the corresponding probs and indices
        if use_pynative:
            probs, p_args = P.TopK(sorted=True)(logits, top_k)
            probs = probs.asnumpy()
            p_args = p_args.asnumpy()
        else:
            probs, p_args = topk_fun(logits, top_k)
        probs = probs
        p_args = p_args
        # Avoid rounding error
        for i in range(probs.shape[0]):
            if np.sum(probs[i]) == 0:
                probs[i] = np.array([1 / top_k for _ in range(top_k)])
        p = probs / np.sum(probs, -1, keepdims=True)
    return p, p_args


class GeneratorMixin:
    """Generator For the nlp models"""

    def __init__(self):
        pass

    # pylint: disable=W0613
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        prepare inputs for generation.
        A model class needs to define a `prepare_inputs_for_generation` method
        in order to use `.generate()`

        Raises:
            RuntimeError: Not implemented in model but call `.generate()`
        """
        raise RuntimeError(
            "A model class needs to define a `prepare_inputs_for_generation`"
            " method in order to use `.generate()`."
        )

    # pylint: disable=W0613
    def update_model_kwargs_before_generate(self, input_ids, model_kwargs: dict):
        """
        update model kwargs before generate.
        If your model needs to update model kwargs before generate, implement
        this method in your model, else do nothing.
        """
        return

    def slice_incremental_inputs(self, model_inputs: dict, current_index):
        """used for non-first iterations, slice the inputs to length 1."""
        input_ids = model_inputs.pop("input_ids")
        if isinstance(input_ids, Tensor):
            input_ids = input_ids.asnumpy()
        inputs_tmp = []
        for i, index_value in enumerate(current_index):
            current_index_tmp = (
                int(index_value) - i * input_ids.shape[1]
            )  # multibatch
            # use numpy to slice array to avoid complie ascend slice op
            inputs_tmp.append(input_ids[i][current_index_tmp : current_index_tmp + 1])
        inputs_tmp = np.array(inputs_tmp, dtype=np.int32)
        model_inputs["input_ids"] = Tensor(inputs_tmp, mstype.int32)

    def process_logits(self, logits, current_index=None, keep_all=False):
        """Process the logits"""
        logits = logits.reshape(-1, logits.shape[-1])
        if not keep_all and current_index is not None:
            index = current_index.view(-1,)
            logits = P.Gather()(logits, index, 0)
        outputs = P.LogSoftmax(-1)(logits)
        outputs = F.tensor_pow(np.e, outputs)
        return outputs

    def _prepare_model_inputs_for_decoder(self, input_ids, input_mask):
        """generate the inputs for the decoder"""
        batch_size = input_ids.shape[0]

        encoder_mask = Tensor(input_mask, mstype.float32)

        encoder_output = self.encoder_forward(
            Tensor(input_ids, mstype.int32), encoder_mask
        )

        input_ids = np.zeros((batch_size, self.config.max_decode_length))
        logger.debug("Decoder: pad the origin inputs into shape: %s", input_ids.shape)
        target_mask = np.zeros_like(input_ids)
        target_mask[:, 0] = 1

        # As the decoder is generating from [START] token
        return encoder_output, encoder_mask, input_ids, target_mask

    def _pad_inputs_using_max_length(self, origin_inputs, pad_token_id=0):
        """pad the input_ids to the max_length"""
        pad_length = self.config.seq_length - origin_inputs.shape[-1]
        if pad_length < 0:
            raise ValueError(
                f"origin_inputs size is {origin_inputs.shape}, you should"
                f"increase the seq_length of the model {self.config.seq_length}."
            )
        # Pad original inputs to model_origin_max_length
        input_ids = np.pad(
            origin_inputs,
            ((0, 0), (0, pad_length)),
            "constant",
            constant_values=(0, pad_token_id),
        )
        return input_ids

    def _incremental_infer(self, model_inputs, current_index, valid_length_each_example):
        """model forward for incremental infer."""
        # Claim the first graph
        if self.is_first_iteration:
            self.add_flags_recursive(is_first_iteration=True)
            # pylint: disable=E1102
            res = self(
                **model_inputs,
                input_position=Tensor(current_index, mstype.int32),
                init_reset=Tensor([False], mstype.bool_),  # init_reset (1,) bool False
                batch_valid_length=Tensor([valid_length_each_example], mstype.int32),
            )
            # first iter done, go to other iters
            self.is_first_iteration = False
        else:
            self.add_flags_recursive(is_first_iteration=False)
            # slice model inputs for incremental infer
            self.slice_incremental_inputs(model_inputs, current_index)
            # pylint: disable=E1102
            res = self(
                **model_inputs,
                input_position=Tensor(current_index, mstype.int32),
                init_reset=Tensor([True], mstype.bool_),  # init_reset (1,) bool True
                batch_valid_length=Tensor([valid_length_each_example], mstype.int32),
            )

        return res

    def _forward(self,
                 origin_inputs,
                 generation_config: GenerationConfig,
                 streamer: BaseStreamer = None,
                 **model_kwargs):
        """
        Text generation given the model and origin inputs

        Inputs:
            origin_inputs(list): The prompt for generation, should be a list of ids.
            generation_config(GenerationConfig): The controlling config for text generation.
            streamer: Streamer object that will be used to stream the generated sequences.
            model_kwargs: dict of model input kwargs, when be passed to model construct
                during the generation forward.

        Returns:
            outputs: the ids for the generated text
        """
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = 0
        use_pynative = True

        if streamer is not None:
            streamer.put(origin_inputs[0])

        batch_size = origin_inputs.shape[0]
        is_encoder_decoder = self.config.is_encoder_decoder
        logger.debug("The input shape is: %s", origin_inputs.shape)
        valid_length_each_example = []
        for i in range(batch_size):
            # As the nonzero returns the index and we need length
            valid_length_each_example.append(
                np.max(np.argwhere(origin_inputs[i] != generation_config.pad_token_id))
                + 1
            )
        valid_length_each_example = np.array(valid_length_each_example)
        logger.debug("Get the valid for each example is: %s", valid_length_each_example)
        if not is_encoder_decoder and np.max(valid_length_each_example) > generation_config.max_length:
            raise ValueError(
                "The max_length set is smaller than the length in the input_ids."
                f"You shout set max_length to {np.max(valid_length_each_example)}"
            )
        target_length = (
            self.config.seq_length
            if generation_config.max_length > self.config.seq_length
            else generation_config.max_length
        )
        logger.debug("max target_length is: %s", target_length)
        # A list of the frequency of each token
        frequency_list = None
        input_ids = self._pad_inputs_using_max_length(
            origin_inputs=origin_inputs, pad_token_id=generation_config.pad_token_id
        )

        logger.debug(
            "pad the origin inputs from %s into shape: %s",
            origin_inputs.shape,
            input_ids.shape,
        )

        input_mask = np.zeros_like(input_ids)
        for i in range(valid_length_each_example.shape[0]):
            input_mask[i, :valid_length_each_example[i]] = 1
        encoder_output = None
        encoder_mask = None
        if is_encoder_decoder:
            if target_length > self.config.max_decode_length:
                target_length = self.config.max_decode_length
            logger.debug("target_length is: %s", target_length)

            # When do encoder and decoder prediction, the encoder can be cached
            # to speed up the inference
            (
                encoder_output,
                encoder_mask,
                input_ids,
                target_mask,
            ) = self._prepare_model_inputs_for_decoder(input_ids, input_mask)
            valid_length_each_example = np.ones((batch_size, 1)).astype(np.int32)
        # A single loop generates one token, loop until reaching target
        # model_origin_max_length or generating eod token
        is_finished = [False] * batch_size

        # update model kwargs once, before go into generate loop.
        self.update_model_kwargs_before_generate(input_ids, model_kwargs)

        # setup is_first_iteration flag for incremental infer
        if generation_config.use_past:
            self.is_first_iteration = True
        keep_all = False
        while np.sum(is_finished) != batch_size:
            if is_encoder_decoder:
                inputs = Tensor(input_ids, mstype.int32)
                seq_length = inputs.shape[1]
                current_index = [
                    valid_length_each_example[i] - 1 + i * seq_length
                    for i in range(batch_size)
                ]
                # current_index = Tensor(valid_length_each_example - 1, mstype.int32)
                current_index = Tensor(current_index, mstype.int32)
                logger.debug("validate length: %s", valid_length_each_example)
                # pylint: disable=E1102
                logits = self(
                    input_ids=None,
                    attention_mask=encoder_mask,
                    encoder_outputs=encoder_output,
                    decoder_input_ids=inputs,
                    decoder_attention_mask=Tensor(target_mask, mstype.float32),
                )
                log_probs = self.process_logits(logits, current_index)
            else:
                # model prepare input dict
                model_inputs = self.prepare_inputs_for_generation( # pylint: disable=E1111
                    input_ids, **model_kwargs
                )
                seq_length = input_ids.shape[1]
                current_index = [
                    valid_length_each_example[i] - 1 + i * seq_length
                    for i in range(batch_size)
                ]
                logger.debug("validate length: %s", valid_length_each_example)
                # incremental generate
                if generation_config.use_past:
                    # when first iteration, keep last logits; others keep all logits.
                    keep_all = not self.is_first_iteration
                    # incremental generate
                    res = self._incremental_infer(
                        model_inputs=model_inputs,
                        current_index=current_index,
                        valid_length_each_example=valid_length_each_example,
                    )
                # auto-aggressive generate
                else:
                    res = self(**model_inputs)  # pylint: disable=E1102
                if self.config.is_sample_acceleration:
                    p, p_args = res
                    if isinstance(p, Tensor):
                        p = p.asnumpy()
                    if isinstance(p_args, Tensor):
                        p_args = p_args.asnumpy()
                else:
                    logits = res[0]
                    log_probs = self.process_logits(
                        logits, Tensor(current_index, mstype.int32), keep_all
                    )

            # Sample
            if not self.config.is_sample_acceleration:
                log_probs = log_probs.asnumpy()
                vocab_size = log_probs.shape[-1]
                if generation_config.repetition_penalty != 1 and frequency_list is None:
                    frequency_list = np.array([[0 for _ in range(vocab_size)]])
                log_probs_revised = log_probs.reshape(batch_size, vocab_size)
                if generation_config.repetition_penalty != 1:
                    log_probs_revised = (
                        log_probs
                        - frequency_list * generation_config.repetition_penalty
                        - (frequency_list > 0) * generation_config.repetition_penalty
                    )
                p, p_args = sampler(
                    log_probs_revised,
                    generation_config.top_p,
                    generation_config.top_k,
                    use_pynative,
                )

            # Random select a token as final output for this round
            for i in range(batch_size):
                if is_finished[i]:
                    continue
                p_norm = softmax(p[i])
                target_index = np.random.choice(len(p[i]), p=p_norm)

                # update frequency list
                target = p_args[i][target_index]

                if generation_config.repetition_penalty != 1:
                    frequency_list[0][target] = frequency_list[0][target] + 1
                input_ids[i, valid_length_each_example[i]] = p_args[i, target_index]

                if streamer is not None:
                    streamer.put(np.asarray([target]))

                if is_encoder_decoder:
                    target_mask[i][valid_length_each_example[i]] = int(1)

                valid_length_each_example[i] += int(1)
                input_mask[i][valid_length_each_example[i] - 1] = 1

                # Stop judgment
                if p_args[i][target_index] == generation_config.eos_token_id \
                    or valid_length_each_example[i] == target_length:
                    is_finished[i] = True
                    continue

        # Return valid outputs out of padded outputs
        output_ids = []
        for i in range(batch_size):
            output_ids.append(
                input_ids[i, : int(valid_length_each_example[i])].astype(np.int32)
            )
        logger.debug("The output is: %s", output_ids)
        if streamer is not None:
            streamer.end()
        return output_ids

    def generate(self,
                 input_ids: Optional[Union[List[int], List[List[int]]]],
                 generation_config: Optional[GenerationConfig] = None,
                 streamer: Optional[BaseStreamer] = None,
                 seed: Optional[int] = None,
                 **kwargs):
        """
        Generate the words according to the given the input ids.

        Most generation-controlling parameters are set in `generation_config` which, if not passed,
        will be set to the model's default generation configuration. You can override any
        `generation_config` by passing the corresponding parameters to generate(),
        e.g. `.generate(inputs, top_k=3, do_sample=True)`.

        Args:
            input_ids(List(str), List(List(str))): The token id list or a batch of token id list.
                When input a batch of token id list, the length of each token id list
                should be same.
            generation_config (`GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation
                call. `**kwargs` passed to generate matching the attributes of `generation_config`
                will override them. If `generation_config` is not provided, the default config
                from the model configuration will be used. Please note that unspecified parameters
                will inherit [`GenerationConfig`]'s default values, whose documentation should be
                checked to parameterize generation.
            streamer: The streamer that generator uses.
            seed: Random seed used in sample.
            kwargs:
                Specific parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. Supported `generate_config` keywords can be
                checked in [`GenerationConfig`]'s documentation. Mainly used Keywords are shown below:

                max_length(int): The maximum length the generated tokens can have.
                do_sample(bool): Whether to do sampling on the candidate ids.
                    If set True it will be enabled, and set it to be False to disable the sampling,
                    equivalent to topk 1.
                    If set None, it follows the setting in the configureation in the model.
                top_k(int): Determine the topK numbers token id as candidate. This should be a positive number.
                    If set None, it follows the setting in the configureation in the model.
                top_p(float): The accumulation probability of the candidate token ids below the top_p
                    will be select as the condaite ids. The valid value of top_p is between (0, 1]. If the value
                    is larger than 1, top_K algorithm will be enabled. If set None, it follows the setting in the
                    configureation in the model.
                eos_token_id(int): The end of sentence token id. If set None, it follows the setting in the
                    configureation in the model.
                pad_token_id(int): The pad token id. If set None, it follows the setting in the configureation
                    in the model.
                repetition_penalty(float): The penalty factor of the frequency that generated words. The If set 1,
                    the repetition_penalty will not be enabled. If set None, it follows the setting in the
                    configureation in the model. Default None.

        Examples:
            >>> from mindformers import T5ForConditionalGeneration, T5Tokenizer
            >>> t5 = T5ForConditionalGeneration.from_pretrained("t5_small")
            >>> tokenizer = T5Tokenizer.from_pretrained("t5_small")
            >>> words = "translate the English to the Romanian: UN Chief Says There Is No Military Solution in Syria"
            >>> words = tokenizer(words, max_length=21, padding='max_length')['input_ids']
            >>> output = t5.generate(words, do_sample=True)
            >>> output = tokenizer.decode(output[0], skip_special_tokens=True)
            >>> print(output)
            eful ONU declară că nu există o soluţie militară în Siria
            >>> # Enable the top p sampling
            >>> output = t5.generate(words, do_sample=True, top_p=0.4)
            >>> output = tokenizer.decode(output[0], skip_special_tokens=True)
            >>> print(output)
            eful ONU declară că nu există o soluţie militară în Siria
            >>> # Enable the top k sampling.
            >>> output = t5.generate(words, do_sample=True, top_k=10, top_p=1)
            >>> output = tokenizer.decode(output[0], skip_special_tokens=True)
            >>> print(output)
            Este comist de stat ale stateului membre nai uzusepa şi ONU

        Returns:
            A list of the generated token ids
        """
        origin_phase = self.phase
        self.set_train(False)
        input_ids = np.array(input_ids).reshape(-1, np.shape(input_ids)[-1])
        seed = 0 if seed is None else seed
        np.random.seed(seed)

        # Handle `generation_config` and kwargs that might update it
        # priority: `generation_config` argument > `model.generation_config` (default config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation
            # model attribute accordingly, if it was created from the model config
            generation_config = GenerationConfig.from_model_config(self.config)
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(
            **kwargs
        )  # All unused kwargs must be model kwargs

        if not generation_config.do_sample:
            generation_config.top_p = 1
            generation_config.top_k = 1
        logger.debug("Generation Config is: %s", generation_config)

        output_ids = self._forward(
            origin_inputs=input_ids,
            generation_config=generation_config,
            streamer=streamer,
            **model_kwargs,
        )
        # set to original phase
        self.set_train(origin_phase == "train")
        return output_ids
