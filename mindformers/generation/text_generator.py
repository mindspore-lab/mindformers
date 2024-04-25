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
import time
from typing import Optional, List, Union

import numpy as np
import mindspore as ms
from mindspore.ops import functional as F
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor

from mindformers.generation.beam_search import BeamSearchScorer
from mindformers.generation.generation_config import GenerationConfig
from mindformers.generation.logits_process import (LogitNormalization, LogitsProcessorList,
                                                   RepetitionPenaltyLogitsProcessor,
                                                   TemperatureLogitsWarper, TopKLogitsWarper,
                                                   TopPLogitsWarper, MinLengthLogitsProcessor,
                                                   MinNewTokensLengthLogitsProcessor)
from mindformers.generation.streamers import BaseStreamer
from mindformers.generation.utils import softmax_with_threads, topk
from mindformers.modules.block_tables import BlockTables
from mindformers.tools import logger

__all__ = ["GenerationMixin"]


def get_valid_length_each_example(input_ids, pad_token_id):
    """get valid length and max length in a batch"""
    batch_size = input_ids.shape[0]
    valid_length_each_example = []
    for i in range(batch_size):
        # As the nonzero returns the index and we need length
        valid_length_each_example.append(
            np.max(np.argwhere(input_ids[i] != pad_token_id))
            + 1
        )
    valid_length_each_example = np.array(valid_length_each_example)
    logger.debug("Get the valid for each example is: %s", valid_length_each_example)
    max_length = np.max(valid_length_each_example)
    return valid_length_each_example, max_length


class GenerationMode:
    """
    Possible generation modes.
    """

    # Non-beam methods
    GREEDY_SEARCH = "greedy_search"
    SAMPLE = "sample"
    # Beam methods
    BEAM_SEARCH = "beam_search"


class GenerationMixin:
    """Generator For the nlp models"""

    def __init__(self):
        self.block_mgr = None
        self.use_kbk_infer = False

    def _set_block_mgr(self, batch_size):
        """ Set model block table mgr function. """

        if not self.block_mgr:
            self.block_mgr = BlockTables(self.config.num_blocks, self.config.block_size, self.config.seq_length)

        if self.block_mgr:
            self.block_mgr.init_cache_engine(batch_size)

    def _set_kbk_infer(self):
        jit_level = self.jit_config_dict.get("jit_level")
        infer_boost = self.jit_config_dict.get("infer_boost")
        self.use_kbk_infer = (jit_level == "O0" and infer_boost == "on")
        logger.info(
            "Set kbk infer :{}".format(self.use_kbk_infer))

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

    def add_flags_custom(self, is_first_iteration):
        """
        Add customized attributes for specific cells in the model. If the model does not implement this method,
        this will add customized attributes for all cells in the model recursively.

        Args:
            is_first_iteration (bool): Network configuration information.
            Indicate whether current iteration is the first iteration in prediction.
        """
        self.add_flags_recursive(is_first_iteration=is_first_iteration)

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
            if input_ids.shape[-1] == 1:
                model_inputs["input_ids"] = input_ids
                return
            input_ids = input_ids.asnumpy()
        inputs_tmp = []
        for i, index_value in enumerate(current_index):
            current_index_tmp = (
                int(index_value) - i * input_ids.shape[1]
            )  # multibatch
            # use numpy to slice array to avoid complie ascend slice op
            inputs_tmp.append(input_ids[i][current_index_tmp: current_index_tmp + 1])
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

    def _get_logits_processor(self,
                              generation_config: GenerationConfig,
                              input_ids_seq_length: int,
                              logits_processor: Optional[LogitsProcessorList]):
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        # instantiate processors list
        processors = LogitsProcessorList()

        if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty=generation_config.repetition_penalty))
        if (
                generation_config.min_length is not None
                and generation_config.eos_token_id is not None
                and generation_config.min_length > 0
        ):
            processors.append(
                MinLengthLogitsProcessor(
                    generation_config.min_length,
                    generation_config.eos_token_id,
                    generation_config.pad_token_id
                )
            )
        if (
                generation_config.min_new_tokens is not None
                and generation_config.eos_token_id is not None
                and generation_config.min_new_tokens > 0
        ):
            processors.append(
                MinNewTokensLengthLogitsProcessor(
                    input_ids_seq_length,
                    generation_config.min_new_tokens,
                    generation_config.eos_token_id,
                    generation_config.pad_token_id
                )
            )
        processors = self._merge_processor_list(processors, logits_processor)
        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            processors.append(LogitNormalization())
        return processors

    def _merge_processor_list(self,
                              default_list: LogitsProcessorList,
                              custom_list: LogitsProcessorList):
        """merge custom processor list with default list."""
        if not custom_list:
            return default_list
        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = "logits processor"
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `.generate()`, but it has already been created with the values {default}."
                        f" {default} has been created by passing the corresponding arguments to generate or"
                        f" by the model's config default values. If you just want to change the default values"
                        f" of {object_type} consider passing them as arguments to `.generate()`"
                        f" instead of using a custom {object_type}."
                    )
        default_list.extend(custom_list)
        return default_list

    def _get_logits_warper(self, generation_config: GenerationConfig):
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
        used for multinomial sampling.
        """

        # instantiate warpers list
        warpers = LogitsProcessorList()

        # all samplers can be found in `generation_utils_samplers.py`
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(generation_config.temperature))
        min_tokens_to_keep = 1
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep))
        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            warpers.append(LogitNormalization())
        return warpers

    def _get_generation_mode(self, generation_config: GenerationConfig):
        """determine the generation mode by config"""
        if generation_config.num_beams == 1:
            if generation_config.do_sample:
                logger.info("The generation mode will be **SAMPLE**.")
                return GenerationMode.SAMPLE
            logger.info("The generation mode will be **GREEDY_SEARCH**.")
            return GenerationMode.GREEDY_SEARCH
        logger.info("The generation mode will be **BEAM_SEARCH**.")
        return GenerationMode.BEAM_SEARCH

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

    def _incremental_infer(self, model_inputs: dict, prefill, current_index, valid_length_each_example, block_tables,
                           slot_mapping):
        """model forward for incremental infer."""
        # Claim the first graph
        if prefill:
            self.phase = "prefill"
            self.add_flags_custom(is_first_iteration=True)
            model_inputs["input_position"] = Tensor(current_index, mstype.int32)
            model_inputs["init_reset"] = Tensor([False], mstype.bool_)  # init_reset (1,) bool False
            model_inputs["batch_valid_length"] = Tensor([valid_length_each_example], mstype.int32)
            if block_tables is not None:
                model_inputs["block_tables"] = Tensor(block_tables, mstype.int32)
                model_inputs["slot_mapping"] = Tensor(slot_mapping, mstype.int32)
            # pylint: disable=E1102
            res = self(
                **model_inputs,
            )
            ms.hal.synchronize()
            self.phase = "increment"
            # first iter done, go to other iters
            self.add_flags_custom(is_first_iteration=False)
        else:
            # slice model inputs for incremental infer
            self.slice_incremental_inputs(model_inputs, current_index)
            model_inputs["input_position"] = Tensor(current_index, mstype.int32)
            model_inputs["init_reset"] = Tensor([True], mstype.bool_)  # init_reset (1,) bool True
            model_inputs["batch_valid_length"] = Tensor([valid_length_each_example], mstype.int32)
            if block_tables is not None:
                model_inputs["block_tables"] = Tensor(block_tables, mstype.int32)
                model_inputs["slot_mapping"] = Tensor(slot_mapping, mstype.int32)
            # pylint: disable=E1102
            res = self(
                **model_inputs,
            )
            ms.hal.synchronize()

        return res

    def _beam_search(self,
                     origin_inputs,
                     beam_scorer: BeamSearchScorer,
                     generation_config: GenerationConfig,
                     logits_processor: Optional[LogitsProcessorList] = None,
                     streamer: BaseStreamer = None,
                     **model_kwargs):
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            origin_inputs (`List(str), List(List(str))`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            generation_config (`GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation
                call. `**kwargs` passed to generate matching the attributes of `generation_config`
                will override them. If `generation_config` is not provided, the default config
                from the model configuration will be used. Please note that unspecified parameters
                will inherit [`GenerationConfig`]'s default values, whose documentation should be
                checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            streamer (`TextStreamer, *optional*`):
                The streamer that generator uses.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            A list of the generated token ids
        """
        if streamer is not None:
            raise ValueError("Streamer does not support in beam search method yet!")
        if generation_config.use_past:
            raise ValueError("Beam search does not support incremental inference yet! Please set use_past to False.")
        if self.config.is_sample_acceleration:
            raise ValueError("Beam search does not support sample acceleration yet! "
                             "Please set is_sample_acceleration to False.")

        total_time = time.time()
        prepare_time = time.time()
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()

        batch_size = len(beam_scorer._beam_hyps)  # pylint: disable=W0212
        num_beams = beam_scorer.num_beams
        batch_beam_size = origin_inputs.shape[0]
        logger.debug("The input shape is: %s", origin_inputs.shape)
        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        valid_length_each_example, _ = \
            get_valid_length_each_example(origin_inputs, generation_config.pad_token_id)

        target_length = (
            self.config.seq_length
            if generation_config.max_length > self.config.seq_length
            else generation_config.max_length
        )
        logger.debug("max target_length is: %s", target_length)
        input_ids = self._pad_inputs_using_max_length(
            origin_inputs=origin_inputs, pad_token_id=generation_config.pad_token_id
        )

        logger.debug(
            "pad the origin inputs from %s into shape: %s",
            origin_inputs.shape,
            input_ids.shape,
        )

        beam_scores = np.zeros((batch_size, num_beams), dtype=np.float64)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.reshape((batch_size * num_beams,))

        input_mask = np.zeros_like(input_ids)
        for i in range(valid_length_each_example.shape[0]):
            input_mask[i, :valid_length_each_example[i]] = 1
        encoder_output = None
        encoder_mask = None
        if self.config.is_encoder_decoder:
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
            valid_length_each_example = np.ones((batch_beam_size, 1)).astype(np.int32)

        # update model kwargs once, before go into generate loop.
        self.update_model_kwargs_before_generate(input_ids, model_kwargs)

        need_gather_logits = True

        is_first_token = True

        origin_len = np.sum(valid_length_each_example) / num_beams
        prepare_time = time.time() - prepare_time
        logger.debug("forward prepare time: %s s", prepare_time)

        while True:
            forward_time = time.time()
            seq_length = input_ids.shape[1]
            current_index = [
                valid_length_each_example[i] - 1 + i * seq_length
                for i in range(batch_beam_size)
            ]
            logger.debug("validate length: %s", valid_length_each_example)
            if self.config.is_encoder_decoder:
                inputs = Tensor(input_ids, mstype.int32)
                # pylint: disable=E1102
                res = self(
                    input_ids=None,
                    attention_mask=encoder_mask,
                    encoder_outputs=encoder_output,
                    decoder_input_ids=inputs,
                    decoder_attention_mask=Tensor(target_mask, mstype.float32),
                )
            else:
                model_kwargs["current_index"] = current_index
                # model prepare input dict
                model_inputs = self.prepare_inputs_for_generation(  # pylint: disable=E1111
                    input_ids, **model_kwargs
                )
                # incremental generate
                if generation_config.use_past:
                    logger.warning("Beam search currently not support incremental, "
                                   "auto-aggressive generate will be performed.")
                # auto-aggressive generate
                res = self(**model_inputs)  # pylint: disable=E1102
            forward_time = time.time() - forward_time

            search_time = time.time()
            # post process logits
            # convert to numpy for post process
            logits = res[0] if isinstance(res, tuple) else res
            if isinstance(logits, Tensor):
                logits = logits.asnumpy().astype(np.float32)
            logits = np.reshape(logits, (-1, logits.shape[-1]))  # (batch_size * num_beams * seq_length, vocab_size)
            # need gather last seq logits using current_index
            # compare length to determine if need gather; if not, gather should be done in model construct
            if need_gather_logits and logits.shape[0] > len(current_index):
                logits = logits[current_index]  # (total_batch_size, vocab_size)
            logits_processor.append(LogitNormalization())

            # post process logits, without changing logits shape and order
            next_token_scores = logits_processor(input_ids, logits)  # (batch_size * num_beams, vocab_size)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = np.reshape(next_token_scores, (batch_size, -1))  # (batch_size, num_beams * vocab_size)

            if is_first_token:
                next_token_scores = next_token_scores[:, :vocab_size]
                is_first_token = False

            # sample 2 next tokens for each beam, so we have at least 1 non eos token per beam
            next_token_scores, next_tokens = topk(
                next_token_scores, 2 * num_beams, axis=1, largest=True, sort=True
            )

            next_indices = np.floor_divide(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            beam_outputs = beam_scorer.process(
                input_ids,  # (batch_size * num_beams, seq_length)
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            search_time = time.time() - search_time

            update_time = time.time()
            # reorder model inputs
            old_input_ids = input_ids.copy()
            for i in range(batch_beam_size):
                input_ids[i] = old_input_ids[beam_idx[i], :]

            # add new tokens to input_ids
            for i in range(batch_beam_size):
                input_ids[i, valid_length_each_example[i]] = beam_next_tokens[i]
                if self.config.is_encoder_decoder:
                    target_mask[i][valid_length_each_example[i]] = int(1)

                input_mask[i][valid_length_each_example[i]] = 1
                valid_length_each_example[i] += int(1)

            update_time = time.time() - update_time
            logger.debug("forward time: %s s; beam search time: %s s; update time: %s s; total count: %s s",
                         forward_time, search_time, update_time, forward_time + search_time + update_time)

            if beam_scorer.is_done or np.min(valid_length_each_example) >= generation_config.max_length:
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            max_length=generation_config.max_length
        )

        generate_len = np.sum(valid_length_each_example) / num_beams - origin_len
        total_time = time.time() - total_time
        logger.info("total time: %s s; generated tokens: %s tokens; generate speed: %s tokens/s",
                    total_time, generate_len, generate_len / total_time)

        return sequence_outputs["sequences"]

    def generate(self,
                 input_ids: Optional[Union[List[int], List[List[int]]]],
                 generation_config: Optional[GenerationConfig] = None,
                 logits_processor: Optional[LogitsProcessorList] = None,
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
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            streamer: The streamer that generator uses.
            seed: Random seed used in sample.
            kwargs:
                Specific parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. Supported `generate_config` keywords can be
                checked in [`GenerationConfig`]'s documentation. Mainly used Keywords are shown below:

                max_length(int): The maximum length the generated tokens can have. Corresponds to the length of
                    the input prompt + `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
                max_new_tokens (int): The maximum numbers of tokens to generate, ignoring the number of
                    tokens in the prompt.
                min_length (int): The minimum length of the sequence to be generated. Corresponds to the length of the
                    input prompt + `min_new_tokens`. Its effect is overridden by `min_new_tokens`, if also set.
                min_new_tokens (int): The minimum numbers of tokens to generate, ignoring the number of tokens
                    in the prompt.
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
                num_beams(int): Number of beams for beam search. 1 means no beam search. If larger than 1, do_sample
                    will be set to false.

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
            >>> from mindformers import T5ForConditionalGeneration, T5Tokenizer
            >>> t5 = T5ForConditionalGeneration.from_pretrained("t5_small")
            >>> tokenizer = T5Tokenizer.from_pretrained("t5_small")
            >>> words = "translate the English to the Romanian: UN Chief Says There Is No Military Solution in Syria"
            >>> words = tokenizer(words, max_length=21, padding='max_length')['input_ids']
            >>> output = t5.generate(words, num_beams=3)
            >>> output = tokenizer.decode(output[0], skip_special_tokens=True)
            >>> print(output)
            eful ONU declară că nu există o soluţie militară în Siria

        Returns:
            A list of the generated token ids
        """
        origin_phase = self.phase
        self.set_train(False)
        try:
            input_ids = np.array(input_ids)
        except ValueError as e:
            raise ValueError(str(e) + " Please check your inputs of model.generate(),"
                                      " and make sure the inputs are padded to same length.")
        input_ids = np.reshape(input_ids, (-1, np.shape(input_ids)[-1]))
        batch_size = input_ids.shape[0]

        seed = 0 if seed is None else seed
        np.random.seed(seed)

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()

        # use_past should be defined in model config
        use_past_tmp = kwargs.pop("use_past", None)
        if use_past_tmp is not None:
            logger.warning("use_past should be defined in model config, it will not take effect when passed to "
                           ".generate() method.")

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

        if generation_config.num_beams > 1:
            logger.warning("When num_beams is set to a value greater than 1, do_sample will be set to False, "
                           "due to the current beam search does not support sampling.")
            generation_config.do_sample = False
        if not generation_config.do_sample:
            logger.warning("When do_sample is set to False, top_k will be set to 1 and top_p will be set to 0, "
                           "making them inactive.")
            generation_config.top_p = 1.0
            generation_config.top_k = 0
        logger.info("Generation Config is: %s", generation_config)

        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = 0

        if generation_config.max_length > self.config.seq_length:
            logger.warning("max_length %s can not exceeds model seq_length %s, set max_length = seq_length.",
                           generation_config.max_length, self.config.seq_length)
            generation_config.max_length = self.config.seq_length

        logger.debug("max length is: %s", generation_config.max_length)

        _, input_ids_length = get_valid_length_each_example(input_ids, generation_config.pad_token_id)

        if generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        if not self.config.is_encoder_decoder and input_ids_length > generation_config.max_length:
            raise ValueError(
                "The max_length set is smaller than the length in the input_ids."
                f"You shout set max_length to {input_ids_length}"
            )

        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            logits_processor=logits_processor,
        )

        # determine generation mode
        generation_config.generation_mode = self._get_generation_mode(generation_config)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search yet. Make sure that `num_beams` is set to 1."
            )
        self._set_kbk_infer()

        if generation_config.use_past and self.use_kbk_infer:
            self._set_block_mgr(batch_size)
            if self.config.is_dynamic:
                self.set_dynamic_inputs()

        # beam search
        if generation_config.generation_mode == GenerationMode.BEAM_SEARCH:
            # prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                max_length=generation_config.max_length
            )
            # interleave input_ids with `num_beams` additional sequences per batch
            input_ids = np.repeat(input_ids, generation_config.num_beams, 0)

            # run beam search
            output_ids = self._beam_search(
                origin_inputs=input_ids,
                beam_scorer=beam_scorer,
                generation_config=generation_config,
                logits_processor=logits_processor,
                streamer=streamer,
                **model_kwargs
            )
        # greedy search or sample
        else:
            total_time = time.time()
            prepare_time = time.time()

            origin_inputs = input_ids
            logits_warper = self._get_logits_warper(generation_config) \
                if generation_config.generation_mode == GenerationMode.SAMPLE else None

            if streamer is not None:
                streamer.put(origin_inputs)

            batch_size = origin_inputs.shape[0]
            logger.debug("The input shape is: %s", origin_inputs.shape)

            valid_length_each_example, _ = \
                get_valid_length_each_example(origin_inputs, generation_config.pad_token_id)

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
            target_mask = None
            if self.config.is_encoder_decoder:
                if generation_config.max_length > self.config.max_decode_length:
                    generation_config.max_length = self.config.max_decode_length
                logger.debug("max decode length is: %s", generation_config.max_length)

                # When do encoder and decoder prediction, the encoder can be cached
                # to speed up the inference
                (
                    encoder_output,
                    encoder_mask,
                    input_ids,
                    target_mask,
                ) = self._prepare_model_inputs_for_decoder(input_ids, input_mask)
                valid_length_each_example = [1 for _ in range(batch_size)]
            # A single loop generates one token, loop until reaching target
            # model_origin_max_length or generating eod token
            is_finished = [False] * batch_size

            # update model kwargs once, before go into generate loop.
            self.update_model_kwargs_before_generate(input_ids, model_kwargs)

            origin_len = np.sum(valid_length_each_example)
            prepare_time = time.time() - prepare_time
            logger.debug("forward prepare time: %s s", prepare_time)

            prefill = True
            model_kwargs["origin_inputs"] = origin_inputs
            while np.sum(is_finished) != batch_size:
                block_tables = None
                slot_mapping = None
                if generation_config.use_past and self.use_kbk_infer:
                    if prefill:
                        if self.config.is_dynamic:
                            max_input_length = len(origin_inputs[0])
                        else:
                            max_input_length = self.config.seq_length
                        block_tables, slot_mapping = self.block_mgr.assemble_pa_full_inputs(max_input_length,
                                                                                            valid_length_each_example,
                                                                                            is_finished)
                    else:
                        block_tables, slot_mapping = self.block_mgr.assemble_pa_inc_inputs(valid_length_each_example,
                                                                                           is_finished)

                target_list, is_finished = self.infer(input_ids=input_ids,
                                                      valid_length_each_example=valid_length_each_example,
                                                      generation_config=generation_config,
                                                      logits_processor=logits_processor,
                                                      logits_warper=logits_warper,
                                                      block_tables=block_tables,
                                                      slot_mapping=slot_mapping,
                                                      prefill=prefill,
                                                      is_finished=is_finished,
                                                      encoder_mask=encoder_mask,
                                                      encoder_output=encoder_output,
                                                      target_mask=target_mask,
                                                      **model_kwargs)
                if generation_config.use_past:
                    if prefill and "origin_inputs" in model_kwargs:
                        model_kwargs.pop("origin_inputs")
                    prefill = False
                for i in range(batch_size):
                    if is_finished[i]:
                        continue
                    input_ids[i, valid_length_each_example[i]] = target_list[i]

                    if self.config.is_encoder_decoder:
                        target_mask[i][valid_length_each_example[i]] = int(1)

                    valid_length_each_example[i] += int(1)
                    input_mask[i][valid_length_each_example[i] - 1] = 1

                if streamer is not None:
                    if batch_size == 1:
                        streamer.put(target_list[0])
                    else:
                        streamer.put(target_list)

            # Return valid outputs out of padded outputs
            output_ids = []
            for i in range(batch_size):
                output_ids.append(
                    input_ids[i, : int(valid_length_each_example[i])].astype(np.int32)
                )
            logger.debug("The output is: %s", output_ids)
            if streamer is not None:
                streamer.end()

            generate_len = np.sum(valid_length_each_example) - origin_len
            total_time = time.time() - total_time
            logger.info("total time: %s s; generated tokens: %s tokens; generate speed: %s tokens/s",
                        total_time, generate_len, generate_len / total_time)

        # set to original phase
        self.set_train(origin_phase == "train")

        if self.block_mgr:
            self.block_mgr.clear_cache()

        return output_ids

    def infer(self,
              input_ids: [Union[List[int], List[List[int]]]],
              valid_length_each_example: [List[int]],
              generation_config: [GenerationConfig] = None,
              logits_processor: Optional[LogitsProcessorList] = None,
              logits_warper: Optional[LogitsProcessorList] = None,
              block_tables: Optional[Tensor] = None,
              slot_mapping: Optional[Tensor] = None,
              prefill: bool = True,
              is_finished: List[bool] = None,
              encoder_mask: Optional[Tensor] = None,
              encoder_output: Optional[Tensor] = None,
              target_mask: Optional[Tensor] = None,
              **model_kwargs):
        """
        do infer and return logits on next position, can choose do prefill or decode predict.

        Args:
            input_ids (List(List(int))):
                Input ids after padding.
            valid_length_each_example (List(int)):
                Valid input length except padding.
            generation_config (`GenerationConfig`):
                The generation configuration to be used as base parametrization for the generation call.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            block_tables (Tensor):
                Params for page attention
            slot_mapping (Tensor):
                Params for page attention
            prefill (bool):
                Whether to do prefill predict or decode predict
            is_finished (List(bool)):
                Whether each sequence is finished its generation.
            encoder_mask (Tensor):
                Use for encoder-decoder construct, do not need for decoder only construct
            encoder_output (Tensor):
                Use for encoder-decoder construct, do not need for decoder only construct
            target_mask (Tensor):
                Use for encoder-decoder construct, do not need for decoder only construct

        Returns:
            next_token, is_finished
        """

        start_time = time.time()

        input_ids = np.array(input_ids)
        res, current_index = self.forward(input_ids=input_ids,
                                          valid_length_each_example=valid_length_each_example,
                                          generation_config=generation_config,
                                          block_tables=block_tables,
                                          slot_mapping=slot_mapping,
                                          prefill=prefill,
                                          encoder_mask=encoder_mask,
                                          encoder_output=encoder_output,
                                          target_mask=target_mask,
                                          **model_kwargs)

        forward_time = time.time() - start_time
        sample_time = time.time()

        need_gather_logits = True
        if not self.config.is_encoder_decoder and generation_config.use_past:
            need_gather_logits = prefill

        next_id, is_finished = self.postprocess(input_ids=input_ids,
                                                is_finished=is_finished,
                                                res=res,
                                                generation_config=generation_config,
                                                valid_length_each_example=valid_length_each_example,
                                                current_index=current_index,
                                                logits_processor=logits_processor,
                                                logits_warper=logits_warper,
                                                need_gather_logits=need_gather_logits)

        sample_time = time.time() - sample_time
        infer_time = time.time() - start_time
        logger.debug("forward time: %s s; sample time: %s s; total count: %s s",
                     forward_time, sample_time, infer_time)

        return next_id, is_finished

    def forward(self,
                input_ids: [Union[List[int], List[List[int]]]],
                valid_length_each_example: [List[int]],
                generation_config: [GenerationConfig] = None,
                block_tables: Optional[Tensor] = None,
                slot_mapping: Optional[Tensor] = None,
                prefill: bool = None,
                encoder_mask: Optional[Tensor] = None,
                encoder_output: Optional[Tensor] = None,
                target_mask: Optional[Tensor] = None,
                **model_kwargs):
        """
        Model forward process.

        Args:
            input_ids (List(List(int))):
                Input ids after padding.
            valid_length_each_example (List(int)):
                Valid input length except padding.
            generation_config (`GenerationConfig`):
                The generation configuration to be used as base parametrization for the generation call.
            block_tables (Tensor):
                Params for page attention
            slot_mapping (Tensor):
                Params for page attention
            prefill (bool):
                Whether to do prefill predict or decode predict
            encoder_mask (Tensor):
                Use for encoder-decoder construct, do not need for decoder only construct
            encoder_output (Tensor):
                Use for encoder-decoder construct, do not need for decoder only construct
            target_mask (Tensor):
                Use for encoder-decoder construct, do not need for decoder only construct

        Returns:
            res, current_index
        """
        input_ids = np.reshape(input_ids, (-1, np.shape(input_ids)[-1]))
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        current_index = [
            valid_length_each_example[i] - 1 + i * seq_length
            for i in range(batch_size)
        ]

        if self.config.is_encoder_decoder:
            inputs = Tensor(input_ids, mstype.int32)
            # pylint: disable=E1102
            res = self(
                input_ids=None,
                attention_mask=encoder_mask,
                encoder_outputs=encoder_output,
                decoder_input_ids=inputs,
                decoder_attention_mask=Tensor(target_mask, mstype.float32),
            )
        else:
            model_kwargs["current_index"] = current_index
            # pylint: disable=E1111
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            real_input_ids = model_inputs["input_ids"]
            batch_size = real_input_ids.shape[0]
            seq_length = real_input_ids.shape[1]
            current_index = [
                valid_length_each_example[i] - 1 + i * seq_length
                for i in range(batch_size)
            ]
            model_kwargs["current_index"] = current_index

            if generation_config.use_past:
                res = self._incremental_infer(
                    model_inputs=model_inputs,
                    prefill=prefill,
                    current_index=current_index,
                    valid_length_each_example=valid_length_each_example,
                    block_tables=block_tables,
                    slot_mapping=slot_mapping
                )
            else:
                res = self(**model_inputs)  # pylint: disable=E1102

        return res, current_index

    def postprocess(self,
                    input_ids,
                    is_finished,
                    res,
                    generation_config,
                    valid_length_each_example,
                    current_index: Optional[Union[List[int], List[List[int]]]],
                    logits_processor: Optional[LogitsProcessorList] = None,
                    logits_warper: Optional[LogitsProcessorList] = None,
                    need_gather_logits: bool = True):
        """
        postprocess of the output from model generation.

        Args:
            input_ids (List(List(int))):
                Input ids after padding.
            res (List(List(int))):
                Logits after infer.
            is_finished (List(bool)):
                Whether each sequence is finished its generation.
            generation_config (`GenerationConfig`):
                The generation configuration to be used as base parametrization for the generation call.
            valid_length_each_example (List(int)):
                Valid input length except padding.
            current_index (List(int)):
                Current index of sequence.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            need_gather_logits (bool):
                whether gather result, when decode predict and is first iteration, set True.
        """
        batch_size = input_ids.shape[0]
        target_list = [[] for _ in range(batch_size)]

        if not hasattr(generation_config, "generation_mode"):
            generation_config.generation_mode = self._get_generation_mode(generation_config)
        if generation_config.generation_mode == GenerationMode.GREEDY_SEARCH:
            if not self.config.is_sample_acceleration:
                logits = res[0] if isinstance(res, tuple) else res
                logits = P.Reshape()(logits, (-1, logits.shape[-1]))
                if need_gather_logits and logits.shape[0] > len(current_index):
                    logits = logits[Tensor(current_index, dtype=mstype.int32)]
                target_list = P.Argmax()(logits)
                target_list = target_list.asnumpy().tolist()
            else:
                probs, p_args = res
                if isinstance(p_args, Tensor):
                    p_args = p_args.asnumpy()
                target_index_list = P.Argmax()(probs)
                target_index_list = target_index_list.asnumpy().tolist()
            # run greedy search
            for i in range(batch_size):
                if is_finished[i]:
                    continue
                if self.config.is_sample_acceleration:
                    target_index = target_index_list[i]
                    target = p_args[i][target_index]
                    target_list[i] = target
                # Stop judgment
                if target_list[i] == generation_config.eos_token_id \
                        or valid_length_each_example[i] == generation_config.max_length - 1:
                    is_finished[i] = True

        elif generation_config.generation_mode == GenerationMode.SAMPLE:
            if not self.config.is_sample_acceleration:
                # convert to numpy for post process
                logits = res[0] if isinstance(res, tuple) else res
                if isinstance(logits, Tensor):
                    logits = logits.asnumpy()
                logits = np.reshape(logits, (-1, logits.shape[-1]))
                # need gather last seq logits using current_index
                # compare length to determine if need gather; if not, gather should be done in model construct
                if need_gather_logits and logits.shape[0] > len(current_index):
                    logits = logits[current_index]

                probs = logits_processor(input_ids, logits, is_finished)
                p_args = np.tile(np.arange(logits.shape[-1]), (batch_size, 1))
                probs = logits_warper(input_ids, probs, is_finished)
            else:
                probs, p_args = res
                if isinstance(probs, Tensor):
                    probs = probs.asnumpy()
                if isinstance(p_args, Tensor):
                    p_args = p_args.asnumpy()
            p_norms = softmax_with_threads(probs, is_finished)

            for i in range(batch_size):
                if is_finished[i]:
                    continue
                p_norm = p_norms[i]
                target_index = np.random.choice(len(probs[i]), p=p_norm)
                # get target token id
                target = p_args[i][target_index]
                target_list[i] = target
                # Stop judgment
                if p_args[i][target_index] == generation_config.eos_token_id \
                        or valid_length_each_example[i] == generation_config.max_length - 1:
                    is_finished[i] = True

        elif generation_config.generation_mode == GenerationMode.BEAM_SEARCH:
            raise ValueError("sampler method doesn't support BEAM_SEARCH. ")

        return target_list, is_finished
