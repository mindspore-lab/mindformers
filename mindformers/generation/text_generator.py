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
import os
import copy
import time
from typing import Optional, List, Union, Dict

import numpy as np
import mindspore as ms
from mindspore import mint, mutable, ops
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
from mindformers.core.context import is_legacy_model
from mindformers.version_control import is_310p
from mindformers.models.utils import format_type
from mindformers.models.tokenization_utils import PreTrainedTokenizer
from mindformers.generation.streamers import BaseStreamer
from mindformers.generation.utils import softmax_with_threads, topk, GenerateOutput, InferOutput, convert_pin
from mindformers.modules.block_tables import BlockTables
from mindformers.tools.logger import logger
from mindformers.tools.utils import is_pynative
from mindformers.tools.debug_info import DetailedLatency, Profiling
from mindformers.generation.parallel_decoding import parallel_decoding_control, parallel_decoding_process
from mindformers.parallel_core.inference.parallel_state import get_tensor_model_parallel_world_size
from mindformers.generation.parallel_decoding_mcore import la_pre_process
from mindformers.parallel_core.inference.utils import (
    get_tp_world_size,
    divide,
)

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
    """A class providing all functions for autoregressive text generation, used as a mixin with PreTrainedModel."""

    def __init__(self):
        self.detailed_latency = DetailedLatency()
        self.profile = Profiling()
        self.block_mgr = None
        self.is_pynative = is_pynative()
        self.argmax = mint.argmax
        self._pre_set_phase = None
        self._exec_add_flags = True
        self.gather = P.Gather()
        if is_310p():
            self.hard_mask = Tensor([0], dtype=ms.float16).reshape(1, 1)
        else:
            self.hard_mask = Tensor([0], dtype=ms.bfloat16).reshape(1, 1)
        self.lower_triangle_mask = None
        self.key_cache = None
        self.value_cache = None

    def _set_network_phase(self, phase):
        self._pre_set_phase = phase
        self._exec_add_flags = True

    def _set_block_mgr(self, batch_size, seq_length):
        """ Set model block table mgr function. """
        if not self.block_mgr:
            self.block_mgr = BlockTables(self.config.num_blocks, self.config.block_size, seq_length)

        if self.block_mgr:
            self.block_mgr.init_cache_engine(batch_size)

    def _set_kv_cache(self):
        """Initial key cache and value cache."""
        if self.key_cache is None and self.value_cache is None:
            tansformer_config = self.get_gpt_model().config

            num_heads = tansformer_config.num_attention_heads
            num_query_groups = tansformer_config.num_query_groups or num_heads

            tp_group_size = get_tp_world_size()
            # When kv heads < tp size, will replicate kv heads
            if num_query_groups < tp_group_size:
                num_query_groups = tp_group_size

            if hasattr(tansformer_config, 'kv_channels'):
                hidden_size_per_attention_head = getattr(tansformer_config, 'kv_channels')
            else:
                hidden_size_per_attention_head = divide(tansformer_config.hidden_size, num_heads)

            if num_heads != num_query_groups:
                num_query_groups_per_partition = divide(num_query_groups, tp_group_size)
            else:
                num_query_groups_per_partition = divide(num_heads, tp_group_size)

            if tansformer_config.multi_latent_attention:
                kv_cache_shape = (
                    tansformer_config.num_blocks,
                    tansformer_config.block_size,
                    1,
                    tansformer_config.kv_lora_rank + tansformer_config.qk_pos_emb_head_dim
                    )
            else:
                kv_cache_shape = (
                    tansformer_config.num_blocks,
                    tansformer_config.block_size,
                    num_query_groups_per_partition,
                    hidden_size_per_attention_head
                    )
            use_ringmla = getattr(self, 'use_fused_mla', False) and get_tensor_model_parallel_world_size() < 16
            key_cache, value_cache = self._generate_kv_cache(tansformer_config, kv_cache_shape, use_ringmla)
            self.key_cache = mutable(key_cache)
            self.value_cache = mutable(value_cache) if not tansformer_config.multi_latent_attention \
                                                       or use_ringmla else None

    def _generate_kv_cache(self, tansformer_config, kv_cache_shape, use_ringmla):
        """generate empty kv_cache"""
        fa3_quant = self.model.quant_config.fa3_quant if self.model.quant_config else False
        fa3_quant_layer = self.model.quant_config.fa3_quant_layer if self.model.quant_config else set()
        compute_dtype = tansformer_config.compute_dtype
        if fa3_quant and not use_ringmla:
            raise ValueError(f'For fa3_quant, it is necessary to set use_ringmla to True.')
        key_cache = []
        value_cache = []
        if is_310p():
            num_blocks, block_size, num_groups, head_dim = kv_cache_shape
            merge_dim = num_groups * head_dim
            kv_cache_shape = (num_blocks, block_size, merge_dim)
        for num_layer in range(tansformer_config.num_layers):
            if fa3_quant:
                import ms_custom_ops
                k_cache_dtype = mstype.int8 if num_layer in fa3_quant_layer else compute_dtype
                k_cache = mint.zeros(kv_cache_shape[:-2] + (tansformer_config.kv_lora_rank,), dtype=k_cache_dtype)
                v_cache = mint.zeros(kv_cache_shape[:-2] + (tansformer_config.qk_pos_emb_head_dim,),
                                     dtype=compute_dtype)
                k_cache = ms_custom_ops.trans_data(k_cache, transdata_type=1)
                v_cache = ms_custom_ops.trans_data(v_cache, transdata_type=1)
            elif use_ringmla:
                k_cache = mint.zeros(kv_cache_shape[:-1] + (tansformer_config.kv_lora_rank,), dtype=compute_dtype)
                v_cache = mint.zeros(kv_cache_shape[:-1] + (tansformer_config.qk_pos_emb_head_dim,),
                                     dtype=compute_dtype)
            else:
                k_cache = mint.zeros(kv_cache_shape, dtype=compute_dtype)
                v_cache = mint.zeros(kv_cache_shape, dtype=compute_dtype)
            # 310p nd-to-nz precess only supports format_cast
            if is_310p():
                k_cache = ops.auto_generate.format_cast(k_cache, format_type['nz'])
                v_cache = ops.auto_generate.format_cast(v_cache, format_type['nz'])
            key_cache.append(k_cache)
            value_cache.append(v_cache)
        return key_cache, value_cache

    def _set_lower_triangle_mask(self):
        """Initial attention mask."""
        if self.lower_triangle_mask is None:
            compute_dtype = self.get_gpt_model().config.compute_dtype
            mask_coeff = 1.0 if compute_dtype is mstype.bfloat16 else -10000.0
            self.lower_triangle_mask = Tensor(
                np.triu(np.ones(shape=(128, 128), dtype=np.float16), 1) * mask_coeff, dtype=compute_dtype
                )

    @staticmethod
    def _prepare_inputs_for_prefill_flatten(input_ids, batch_valid_length, slot_mapping, model_inputs):
        """prepare inputs ids for prefill flatten"""
        batch_valid_length_bs = batch_valid_length.shape[0]  # [bs,]
        input_ids_list = []
        for i in range(batch_valid_length_bs):
            context_len = batch_valid_length[i]
            input_ids_list.append(input_ids[i][:context_len])
        input_ids = np.concatenate(input_ids_list, 0)
        input_ids = input_ids.reshape((1, -1))
        slot_mapping = np.delete(slot_mapping, np.where(slot_mapping == -1))
        model_inputs["input_ids"] = Tensor.from_numpy(input_ids.astype(np.int32))
        model_inputs["slot_mapping"] = Tensor.from_numpy(slot_mapping)
        return model_inputs

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        prepare inputs for generation.
        A model class needs to define a `prepare_inputs_for_generation` method
        in order to use `.generate()`

        """
        model_inputs = {"input_ids": Tensor.from_numpy(input_ids.astype(np.int32))}
        if self.is_pynative:
            model_inputs = {}
            if self.config.is_dynamic and "origin_inputs" in kwargs and self.use_past:
                input_ids = kwargs["origin_inputs"]
            model_inputs["input_ids"] = Tensor.from_numpy(input_ids.astype(np.int32))
        else:
            if self.config.is_dynamic:
                prefill = kwargs.get("prefill")
                if prefill and "origin_inputs" in kwargs:
                    origin_inputs = kwargs["origin_inputs"]
                    batch_valid_length = kwargs.get("valid_length_each_example")
                    slot_mapping = kwargs.get("slot_mapping")
                    model_inputs = self._prepare_inputs_for_prefill_flatten(origin_inputs,
                                                                            batch_valid_length,
                                                                            slot_mapping,
                                                                            model_inputs)
        return model_inputs

    def update_padding_index_to_inputs(self, model_inputs):
        return model_inputs

    def add_flags_custom(self, is_first_iteration):
        """
        Add customized attributes for specific cells in the model. If the model does not implement this method,
        this will add customized attributes for all cells in the model recursively.

        Args:
            is_first_iteration (bool): Network configuration information.
                Indicate whether current iteration is the first iteration in prediction.
        """
        self.add_flags_recursive(is_first_iteration=is_first_iteration)

    def add_flags_custom_mcore(self, is_prefill):
        """
        Add customized attributes for specific cells in the model. If the model does not implement this method,
        this will add customized attributes for all cells in the model recursively.

        Args:
            is_first_iteration (bool): Network configuration information.
                Indicate whether current iteration is the first iteration in prediction.
        """
        self.add_flags_recursive(is_prefill=is_prefill)

    @staticmethod
    def update_model_kwargs_before_generate(input_ids, model_kwargs: dict):
        """
        update model kwargs before generate.
        If your model needs to update model kwargs before generate, implement
        this method in your model, else do nothing.
        """
        return

    @staticmethod
    def slice_incremental_inputs(model_inputs: dict, current_index, need_flatten: bool = False):
        """used for non-first iterations, slice the inputs to length 1."""
        input_ids = model_inputs.pop("input_ids")
        if isinstance(input_ids, Tensor):
            if input_ids.shape[-1] == 1:
                shape = (-1,) if need_flatten else (-1, 1)
                model_inputs["input_ids"] = input_ids.reshape(shape)
                return
            input_ids = input_ids.asnumpy()

        current_index_tmp = current_index - np.arange(input_ids.size, step=input_ids.shape[1])
        arg = np.arange(input_ids.shape[0])
        if need_flatten:
            inputs_tmp = input_ids[arg, current_index_tmp].reshape(-1)
        else:
            inputs_tmp = input_ids[arg, current_index_tmp].reshape(-1, 1)
        model_inputs["input_ids"] = Tensor.from_numpy(inputs_tmp.astype(np.int32))

    @staticmethod
    def process_logits(logits, current_index=None, keep_all=False):
        """Process the logits"""
        logits = logits.reshape(-1, logits.shape[-1])
        if not keep_all and current_index is not None:
            index = current_index.view(-1,)
            logits = P.Gather()(logits, index, 0)
        outputs = P.LogSoftmax(-1)(logits)
        outputs = F.tensor_pow(np.e, outputs)
        return outputs

    def get_logits_processor(self,
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

    @staticmethod
    def _merge_processor_list(default_list: LogitsProcessorList,
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

    @staticmethod
    def get_logits_warper(generation_config: GenerationConfig):
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
        used for multinomial sampling.
        """

        # instantiate warpers list
        warpers = LogitsProcessorList()

        # all samplers can be found in `generation_utils_samplers.py`
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(generation_config.temperature))
        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            warpers.append(LogitNormalization())

        if not generation_config.do_sample:
            return warpers

        min_tokens_to_keep = 1
        if generation_config.top_k is not None and generation_config.top_k > 0:
            warpers.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.top_p is not None:
            warpers.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep))
        return warpers

    @staticmethod
    def _get_generation_mode(generation_config: GenerationConfig):
        """determine the generation mode by config"""
        if generation_config.num_beams == 1:
            if generation_config.do_sample:
                return GenerationMode.SAMPLE
            return GenerationMode.GREEDY_SEARCH
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

    def _incremental_infer(self, model_inputs: dict, prefill, current_index, key_cache=None, value_cache=None,
                           need_flatten=False):
        """model forward for incremental infer."""
        # Claim the first graph
        if key_cache is not None:
            model_inputs = {**model_inputs, 'key_cache': key_cache, 'value_cache': value_cache}
        if prefill:
            self.phase = "prefill"
            if self._pre_set_phase:
                self.phase = f"prefill_{self._pre_set_phase}"
            # In dynamic shape scenarios, only the first execution of the prefill process will trigger this.
            if self._exec_add_flags:
                self.add_flags_custom(is_first_iteration=True)
            self.detailed_latency.start_predict_timer()
            if need_flatten:
                model_inputs["input_ids"] = model_inputs["input_ids"].reshape(-1)
            model_inputs["batch_valid_length"] = Tensor.from_numpy(model_inputs["batch_valid_length"])
            model_inputs = self.convert_pin_model_inputs(model_inputs)
            # pylint: disable=E1102
            res = self(
                **model_inputs,
            )
            self.phase = "increment"
            # first iter done, go to other iters, in dynamic shape scenarios, only the first execution
            # of the increment process will trigger this.
            if self._exec_add_flags:
                self.add_flags_custom(is_first_iteration=False)
                if self.config.is_dynamic and not self.is_pynative:
                    self._exec_add_flags = False
        else:
            # slice model inputs for incremental infer
            if self._pre_set_phase:
                self.phase = f"increment_{self._pre_set_phase}"
            if not (hasattr(self.config, 'parallel_decoding_params') and self.config.parallel_decoding_params):
                self.slice_incremental_inputs(model_inputs, current_index, need_flatten)
            self.detailed_latency.start_predict_timer()
            model_inputs["batch_valid_length"] = Tensor.from_numpy(model_inputs["batch_valid_length"])
            model_inputs = self.convert_pin_model_inputs(model_inputs)
            # pylint: disable=E1102
            res = self(
                **model_inputs,
            )

        return res

    def _incremental_infer_mcore(self,
                                 model_inputs: dict,
                                 prefill,
                                 gather_decode=True):
        r"""
        mcore model forward for incremental infer.

        Args:
            model_inputs: infer model inputs.
            prefill: flag to distinguish prefill and decode.
            gather_decode: whether to gather decode logits.

        Returns:
            res: the output logits.

        """
        # Claim the first graph
        if prefill:
            self.phase = "prefill"
            if self._pre_set_phase:
                self.phase = f"prefill_{self._pre_set_phase}"
            # In dynamic shape scenarios, only the first execution of the prefill process will trigger this.
            if self._exec_add_flags:
                self.add_flags_custom_mcore(is_prefill=True)
            self.detailed_latency.start_predict_timer()
            # pylint: disable=E1102
            res = self(
                **model_inputs,
            )
            self.phase = "increment"
            # first iter done, go to other iters, in dynamic shape scenarios, only the first execution
            # of the increment process will trigger this.
            if self._exec_add_flags:
                self.add_flags_custom_mcore(is_prefill=False)
                self._exec_add_flags = False

        else:
            # slice model inputs for incremental infer
            if self._pre_set_phase:
                self.phase = f"increment_{self._pre_set_phase}"
            self.detailed_latency.start_predict_timer()
            # pylint: disable=E1102
            res = self(
                **model_inputs,
            )
            q_seq_lens = model_inputs.get("q_seq_lens", None)
            if gather_decode and q_seq_lens is not None:
                if q_seq_lens.max() > 1 and q_seq_lens.sum() == res.shape[0]:
                    res = self.gather(res, mint.cumsum(q_seq_lens, dim=0) - 1, 0)
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
        r"""
        Generate the words according to the given the input ids.

        Most generation-controlling parameters are set in `generation_config` which, if not passed,
        will be set to the model's default generation configuration. You can override any
        `generation_config` by passing the corresponding parameters to generate(),
        e.g. `.generate(inputs, top_k=3, do_sample=True)`.

        Args:
            input_ids (List[str], List[List[str]]): The token id list or a batch of token id list.
                When input a batch of token id list, the length of each token id list should be same.
            generation_config (`GenerationConfig`, optional): The generation configuration to be used as
                base parametrization for the generation call.
                `**kwargs` passed to generate matching the attributes of `generation_config` will override them.
                If `generation_config` is not provided, the default config from the model configuration will be used.
                Please note that unspecified parameters will inherit [`GenerationConfig`]'s default values,
                whose documentation should be checked to parameterize generation. Default: ``None``.
            logits_processor (`LogitsProcessorList`, optional): Custom logits processors that
                complement the default logits processors built from arguments and generation config.
                If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users. Default: ``None``.
            streamer (TextStreamer, optional): The streamer that generator uses. Default: ``None``.
            seed (int, optional): Random seed used in sample. Default: ``None``.
            kwargs:
                Specific parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. Supported `generate_config` keywords can be
                checked in [`GenerationConfig`]'s documentation. Mainly used Keywords are shown below:

                - max_length (int): The maximum length the generated tokens can have. Corresponds to the length of
                  the input prompt + `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
                - max_new_tokens (int): The maximum numbers of tokens to generate, ignoring the number of
                  tokens in the prompt.
                - min_length (int): The minimum length of the sequence to be generated.
                  Corresponds to the length of the input prompt + `min_new_tokens`.
                  Its effect is overridden by `min_new_tokens`, if also set.
                - min_new_tokens (int): The minimum numbers of tokens to generate, ignoring the number of tokens
                  in the prompt.
                - do_sample (bool): Whether to do sampling on the candidate ids.
                  If set True it will be enabled, and set it to be False to disable the sampling,
                  equivalent to top-k 1.
                  If set None, it follows the setting in the configureation in the model.
                - top_k (int): Determine the top-k numbers token id as candidate. This should be a positive number.
                  If set None, it follows the setting in the configureation in the model.
                - top_p (float): The accumulation probability of the candidate token ids below the top-p
                  will be select as the condaite ids. The valid value of top-p is between (0, 1]. If the value
                  is larger than 1, top-k algorithm will be enabled. If set None, it follows the setting in the
                  configureation in the model.
                - eos_token_id (int): The end of sentence token id. If set None, it follows the setting in the
                  configureation in the model.
                - pad_token_id (int): The pad token id. If set None, it follows the setting in the configureation
                  in the model.
                - repetition_penalty (float): The penalty factor of the frequency that generated words. The If set 1,
                  the repetition_penalty will not be enabled. If set None, it follows the setting in the
                  configureation in the model. Default: ``None``.
                - num_beams (int): Number of beams for beam search. 1 means no beam search. If larger than 1, do_sample
                  will be set to false.

        Returns:
            A list of the generated token ids.
        """
        self.detailed_latency.clear()
        origin_phase = self.phase
        self.set_train(False)
        try:
            input_ids = np.array(input_ids)
        except ValueError as e:
            raise ValueError(str(e) + " Please check your inputs of model.generate(),"
                                      " and make sure the inputs are padded to same length.") from e
        input_ids = np.reshape(input_ids, (-1, np.shape(input_ids)[-1]))
        batch_size = input_ids.shape[0]

        if seed is not None:
            if not isinstance(seed, int):
                raise ValueError(f"Invalid seed type: {type(seed)}. Seed must be an integer.")
            if not 0 <= seed < 2**64:
                raise ValueError(f"Invalid seed value: {seed}. Seed must be in the range [0, 2**64 - 1].")
            np.random.seed(seed)

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()

        # use_past should be defined in model config
        use_past_tmp = kwargs.pop("use_past", None)
        if use_past_tmp is not None:
            logger.warning("use_past should be defined in model config, it will not take effect when passed to "
                           ".generate() method.")
        use_legacy = is_legacy_model()
        # Handle `generation_config` and kwargs that might update it
        # priority: `generation_config` argument > `model.generation_config` (default config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation
            # model attribute accordingly, if it was created from the model config
            if use_legacy:
                generation_config = GenerationConfig.from_model_config(self.config)
            else:
                generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(
            **kwargs
        )  # All unused kwargs must be model kwargs

        if generation_config.num_beams > 1:
            logger.warning("When num_beams is set to a value greater than 1, do_sample will be set to False, "
                           "due to the current beam search does not support sampling.")
            generation_config.do_sample = False
        logger.info("Generation Config is: %s", generation_config)

        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = 0

        valid_length_each_example, input_ids_length = \
            get_valid_length_each_example(input_ids, generation_config.pad_token_id)
        if hasattr(self.config, "extend_method") and self.config.extend_method == "DYNAMIC_NTK":
            if not self.config.is_dynamic:
                raise ValueError("Dynamic NTK predict mode only support is_dynamic=True, but get is_dynamic=False")

        if generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        if generation_config.max_length > self.config.seq_length:
            logger.warning("max_length %s can not exceeds model seq_length %s, set max_length = seq_length.",
                           generation_config.max_length, self.config.seq_length)
            generation_config.max_length = self.config.seq_length

        logger.debug("max length is: %s", generation_config.max_length)

        if not self.config.is_encoder_decoder and input_ids_length > generation_config.max_length:
            raise ValueError(
                "The max_length set is smaller than the length in the input_ids."
                f"You shout set max_length to {input_ids_length}"
            )

        if generation_config.max_new_tokens is not None:
            max_length_each_example = [valid_length + generation_config.max_new_tokens \
                for valid_length in valid_length_each_example]
        else:
            max_length_each_example = [generation_config.max_length] * len(valid_length_each_example)

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            logger.warning(f"Unfeasible length constraints: `min_length` ({generation_config.min_length}) is "
                           f"larger than the maximum possible length ({generation_config.max_length})."
                           f" Generation will stop at the defined maximum length. "
                           f"You should decrease the minimum length and/or increase the maximum length.")

        if generation_config.min_new_tokens is not None:
            min_length = generation_config.min_new_tokens + input_ids_length
            if min_length > generation_config.max_length:
                logger.warning(
                    f"Unfeasible length constraints: `min_new_tokens` ({generation_config.min_new_tokens}), when "
                    f"added to the prompt length ({input_ids_length}), is larger than"
                    f" the maximum possible length ({generation_config.max_length}). "
                    f"Generation will stop at the defined maximum length. "
                    f"You should decrease the minimum length and/or increase the maximum length."
                )

        logits_processor = self.get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            logits_processor=logits_processor,
        )

        # determine generation mode
        generation_config.generation_mode = self._get_generation_mode(generation_config)
        logger.info(f"The generation mode will be **{generation_config.generation_mode.upper()}**.")
        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search yet. Make sure that `num_beams` is set to 1."
            )

        if not use_legacy:
            self._set_block_mgr(batch_size, self.config.seq_length)
            self._set_kv_cache()
            self._set_lower_triangle_mask()
            self.set_dynamic_inputs()
        elif generation_config.use_past:
            self._set_block_mgr(batch_size, self.config.seq_length)
            if self.config.is_dynamic:
                self.set_dynamic_inputs()

        # prepare dict outputs
        if generation_config.return_dict_in_generate and generation_config.output_logits \
                and self.config.is_sample_acceleration:
            logger.warning("When `is_sample_acceleration` is True, logits can not be fetched. "
                           "Set `output_logits` to False.")
            generation_config.output_logits = False
        scores = () if generation_config.return_dict_in_generate and generation_config.output_scores else None
        raw_logits = () if generation_config.return_dict_in_generate and generation_config.output_logits else None

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
            logits_warper = self.get_logits_warper(generation_config) \
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
                valid_length_each_example = np.array([1 for _ in range(batch_size)])
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

            if (hasattr(self.config, 'pet_config') and self.config.pet_config is not None
                    and self.config.pet_config.pet_type == "slora"):
                adapter_id = kwargs.pop("adapter_id", None)
                if adapter_id is not None and len(adapter_id) > 1:
                    if len(adapter_id) != batch_size:
                        raise ValueError("adapter_ids has different length with inputs.")
                    model_kwargs["adapter_ids"] = adapter_id
                else:
                    model_kwargs["adapter_ids"] = adapter_id * batch_size if adapter_id is not None else None

            while np.sum(is_finished) != batch_size:
                self.detailed_latency.start_preprocess_timer()
                block_tables = None
                slot_mapping = None
                if not use_legacy or generation_config.use_past:
                    if prefill:
                        if (use_legacy and self.is_pynative and self.config.is_dynamic):
                            max_input_length = len(origin_inputs[0])
                        else:
                            max_input_length = self.config.seq_length
                        block_tables, slot_mapping = self.block_mgr.assemble_pa_full_inputs(max_input_length,
                                                                                            valid_length_each_example,
                                                                                            is_finished)
                    else:
                        block_tables, slot_mapping = self.block_mgr.assemble_pa_inc_inputs(valid_length_each_example,
                                                                                           is_finished)
                self.profile.start_profiling(valid_length_each_example[0] - input_ids_length)
                if use_legacy:
                    infer_output, is_finished = self.infer(input_ids=input_ids,
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
                else:
                    infer_output, is_finished = self.infer_mcore(input_ids=input_ids,
                                                                 valid_length_each_example=valid_length_each_example,
                                                                 generation_config=generation_config,
                                                                 logits_processor=logits_processor,
                                                                 logits_warper=logits_warper,
                                                                 block_tables=block_tables,
                                                                 slot_mapping=slot_mapping,
                                                                 prefill=prefill,
                                                                 is_finished=is_finished,
                                                                 **model_kwargs)
                self.profile.stop_profiling(valid_length_each_example[0] - input_ids_length)
                if generation_config.return_dict_in_generate:
                    target_list = infer_output["target_list"]
                    if generation_config.output_scores:
                        scores += (infer_output["probs"],)
                    if generation_config.output_logits:
                        raw_logits += (infer_output["logits"],)
                else:
                    target_list = infer_output
                if not use_legacy or generation_config.use_past:
                    if prefill and "origin_inputs" in model_kwargs:
                        model_kwargs.pop("origin_inputs")
                    prefill = False

                for i in range(batch_size):
                    if is_finished[i]:
                        continue
                    input_ids[i, valid_length_each_example[i]] = target_list[i]

                    if self.config.is_encoder_decoder:
                        target_mask[i][valid_length_each_example[i]] = int(1)

                    # Stop judgment
                    if target_list[i] in generation_config.eos_token_id \
                            or valid_length_each_example[i] + 1 == generation_config.max_length \
                            or valid_length_each_example[i] + 1 == max_length_each_example[i]:
                        is_finished[i] = True
                    else:
                        valid_length_each_example[i] += 1
                        input_mask[i][valid_length_each_example[i] - 1] = 1

                if streamer is not None:
                    if batch_size == 1:
                        streamer.put(target_list[0])
                    else:
                        streamer.put(target_list)
                self.detailed_latency.end_postprocess_timer()

            # Return valid outputs out of padded outputs
            valid_length_each_example += 1
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
            self.detailed_latency.print_info()

        # set to original phase
        self.set_train(origin_phase == "train")

        if self.block_mgr:
            self.block_mgr.clear_cache()

        if generation_config.return_dict_in_generate:
            result = GenerateOutput(
                sequences=output_ids,
                scores=scores,
                logits=raw_logits
            )
            return result
        return output_ids

    def infer(self,
              input_ids: Union[List[int], List[List[int]]],
              valid_length_each_example: np.ndarray,
              generation_config: GenerationConfig = None,
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
        Do infer and return logits on next position, can choose do prefill or decode predict.

        Args:
            input_ids (List(List(int))): Input ids after padding.
            valid_length_each_example (np.ndarray): Valid input length except padding.
            generation_config (`GenerationConfig`, optional): The generation configuration to be used
                as base parametrization for the generation call. Default: ``None``.
            logits_processor (`LogitsProcessorList`, optional): An instance of [`LogitsProcessorList`].
                List of instances of class derived from [`LogitsProcessor`] used to modify the prediction scores
                of the language modeling head applied at each generation step. Default: ``None``.
            logits_warper (`LogitsProcessorList`, optional): An instance of [`LogitsProcessorList`].
                List of instances of class derived from [`LogitsWarper`] used to warp the prediction score
                distribution of the language modeling head applied before multinomial sampling
                at each generation step. Default: ``None``.
            block_tables (Tensor, optional): Store mapping tables for each sequence. Default: ``None``.
            slot_mapping (Tensor, optional): Token cache physical slot index. Default: ``None``.
            prefill (bool, optional): Whether to do prefill predict or decode predict. Default: ``True``.
            is_finished (List(bool), optional): Whether each sequence is finished its generation. Default: ``None``.
            encoder_mask (Tensor, optional): Use for encoder-decoder construct, do not need for decoder only
                construct. Default: ``None``.
            encoder_output (Tensor, optional): Use for encoder-decoder construct, do not need for decoder only
                construct. Default: ``None``.
            target_mask (Tensor, optional): Use for encoder-decoder construct, do not need for decoder only
                construct. Default: ``None``.
            **model_kwargs (Any): Keyword arguments of the model.

        Returns:
            next_token, the next token to be generated.
            is_finished, whether the sequence has completed its generation task.
        """
        max_valid_length = max(valid_length_each_example)
        if not self.config.is_encoder_decoder and max_valid_length > self.config.seq_length:
            raise ValueError(
                f"The input length:{max_valid_length} is longer than the seq_length:{self.config.seq_length}, "
                "which is not allowed."
            )

        start_time = time.time()

        input_ids = np.array(input_ids)
        res, current_index = self.forward(input_ids=input_ids,
                                          valid_length_each_example=valid_length_each_example,
                                          block_tables=block_tables,
                                          slot_mapping=slot_mapping,
                                          prefill=prefill,
                                          use_past=generation_config.use_past,
                                          encoder_mask=encoder_mask,
                                          encoder_output=encoder_output,
                                          target_mask=target_mask,
                                          **model_kwargs)

        self.detailed_latency.start_postprocess_timer()
        forward_time = time.time() - start_time
        sample_time = time.time()

        need_gather_logits = True
        if not self.config.is_encoder_decoder and generation_config.use_past:
            need_gather_logits = prefill

        target_list, probs, logits, is_finished = self.postprocess(
            input_ids=input_ids,
            is_finished=is_finished,
            res=res,
            generation_config=generation_config,
            valid_length_each_example=valid_length_each_example,
            current_index=current_index,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            need_gather_logits=need_gather_logits
        )

        sample_time = time.time() - sample_time
        infer_time = time.time() - start_time
        logger.debug("forward time: %s s; sample time: %s s; total count: %s s",
                     forward_time, sample_time, infer_time)

        if generation_config.return_dict_in_generate:
            infer_output_dict = InferOutput(
                target_list=target_list,
                probs=probs,
                logits=logits
            )
            return infer_output_dict, is_finished

        return target_list, is_finished

    def forward(self,
                input_ids: [Union[List[int], List[List[int]]]],
                valid_length_each_example: np.ndarray,
                block_tables: Optional[Tensor] = None,
                slot_mapping: Optional[Tensor] = None,
                prefill: bool = None,
                use_past: bool = False,
                encoder_mask: Optional[Tensor] = None,
                encoder_output: Optional[Tensor] = None,
                target_mask: Optional[Tensor] = None,
                key_cache: Optional[List[Tensor]] = None,
                value_cache: Optional[List[Tensor]] = None,
                **model_kwargs):
        r"""
        Model forward process.

        Args:
            input_ids (List[List[int]]): Input ids after padding.
            valid_length_each_example (np.ndarray): Valid input length except padding.
            block_tables (Tensor, optional): Params for page attention. Default: ``None``.
            slot_mapping (Tensor, optional): Params for page attention. Default: ``None``.
            prefill (bool, optional): Whether to do prefill predict or decode predict. Default: ``None``.
            use_past (bool, optional): Whether to use past. Default: ``False``.
            encoder_mask (Tensor, optional): Use for encoder-decoder construct, do not need for decoder only
                construct. Default: ``None``.
            encoder_output (Tensor, optional): Use for encoder-decoder construct, do not need for decoder only
                construct. Default: ``None``.
            target_mask (Tensor, optional): Use for encoder-decoder construct, do not need for decoder only
                construct. Default: ``None``.
            key_cache (List[Tensor], optional): A group of tensors used for kvcache.
                Default: ``None``.
            value_cache (List[Tensor], optional): A group of tensors used for kvcache.
                Default: ``None``.
            **model_kwargs (Any): Keyword arguments of the model.

        Returns:
            res, the result after the forward process.
            current_index, records the current index of the sequence.
        """
        input_ids = np.reshape(input_ids, (-1, np.shape(input_ids)[-1]))
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
            if parallel_decoding_control(self.config):
                current_index = None
            else:
                current_index = valid_length_each_example - 1 + np.arange(input_ids.size, step=input_ids.shape[1])
            model_kwargs["current_index"] = current_index
            model_kwargs["prefill"] = prefill if use_past else None
            model_kwargs["valid_length_each_example"] = valid_length_each_example
            model_kwargs["block_tables"] = block_tables
            model_kwargs["slot_mapping"] = slot_mapping
            # pylint: disable=E1111
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            real_input_ids = model_inputs["input_ids"]
            if parallel_decoding_control(self.config):
                model_inputs, block_tables, slot_mapping = parallel_decoding_process(
                    self.config, input_ids, model_inputs, **model_kwargs
                )
            else:
                current_index = valid_length_each_example - 1 + np.arange(real_input_ids.numel(),
                                                                          step=real_input_ids.shape[-1])
            if use_past:
                need_flatten = False
                if "need_flatten" in model_inputs:
                    need_flatten = model_inputs["need_flatten"]
                    model_inputs.pop("need_flatten")
                if "batch_valid_length" not in model_inputs:
                    model_inputs["batch_valid_length"] = np.array(valid_length_each_example, dtype=np.int32) \
                                                         if need_flatten else \
                                                         np.array([valid_length_each_example], dtype=np.int32)
                if block_tables is not None and "block_tables" not in model_inputs:
                    model_inputs["block_tables"] = Tensor.from_numpy(block_tables)
                if slot_mapping is not None and "slot_mapping" not in model_inputs:
                    model_inputs["slot_mapping"] = Tensor.from_numpy(slot_mapping)

                model_inputs = self.update_padding_index_to_inputs(model_inputs)

                res = self._incremental_infer(
                    model_inputs=model_inputs,
                    prefill=prefill,
                    current_index=current_index,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    need_flatten=need_flatten
                )
            else:
                if self._pre_set_phase:
                    self.phase = f"predict_{self._pre_set_phase}"
                res = self(**model_inputs)  # pylint: disable=E1102
        return res, current_index

    def gen_attention_mask(self, is_prefill):
        if is_prefill:
            attention_mask = self.lower_triangle_mask
        else:
            attention_mask = self.hard_mask
        return attention_mask

    def prepare_inputs_for_generation_mcore(self,
                                            input_ids: [Union[List[int], List[List[int]]]],
                                            valid_length_each_example: np.ndarray,
                                            block_tables: Optional[Tensor] = None,
                                            slot_mapping: Optional[Tensor] = None,
                                            prefill: bool = None,
                                            **model_kwargs):
        """prepare inputs for mcore"""
        model_inputs = dict()
        seq_lens = np.array(valid_length_each_example)

        q_seq_lens = model_kwargs.get("q_seq_lens", None)
        positions = model_kwargs.get("position_ids", None)
        attention_mask = model_kwargs.get("attention_mask", None)
        if q_seq_lens is None or np.size(q_seq_lens) == 0:
            if len(input_ids) == len(seq_lens):
                q_seq_lens = np.ones_like(seq_lens)
            else:
                q_seq_lens = np.array(valid_length_each_example)
        q_seq_lens = np.array(q_seq_lens)
        if prefill and len(input_ids) != q_seq_lens.sum():
            q_seq_lens = np.array(valid_length_each_example)
        context_lens = seq_lens - q_seq_lens
        if positions is None:
            positions = np.zeros_like(input_ids, dtype=np.int32)
            start = 0
            for i in range(seq_lens.size):
                positions[start:start + q_seq_lens[i]] = np.arange(context_lens[i], seq_lens[i])
                start += q_seq_lens[i]
        if prefill and context_lens.max() > 0:
            prefill = False
        model_inputs["input_ids"] = Tensor.from_numpy(input_ids.astype(np.int32))
        model_inputs["batch_valid_length"] = Tensor.from_numpy(seq_lens.astype(np.int32))
        model_inputs["context_lens_tensor"] = Tensor.from_numpy(context_lens.astype(np.int32))
        model_inputs["q_seq_lens"] = Tensor.from_numpy(q_seq_lens.astype(np.int32))
        model_inputs["positions"] = Tensor.from_numpy(positions.astype(np.int32))
        model_inputs["block_tables"] = Tensor.from_numpy(block_tables)
        model_inputs["slot_mapping"] = Tensor.from_numpy(slot_mapping)
        if attention_mask is not None:
            if isinstance(attention_mask, np.ndarray):
                attention_mask = Tensor.from_numpy(attention_mask)
            model_inputs["attention_mask"] = attention_mask.astype(self.config.compute_dtype)
        else:
            model_inputs["attention_mask"] = self.gen_attention_mask(prefill)
        model_inputs["attn_metadata"] = None
        model_inputs["key_cache"] = self.key_cache
        model_inputs["value_cache"] = self.value_cache
        model_inputs = self.get_gpt_model().update_padding_index_to_inputs(model_inputs)
        return model_inputs, prefill

    def forward_mcore(self,
                      input_ids: [Union[List[int], List[List[int]]]],
                      valid_length_each_example: np.ndarray,
                      block_tables: Optional[Tensor] = None,
                      slot_mapping: Optional[Tensor] = None,
                      prefill: bool = None,
                      **model_kwargs):
        r"""
        Model forward process.

        Args:
            input_ids (List(List(int))): Input ids after padding.
            valid_length_each_example (np.ndarray): Valid input length except padding.
            block_tables (Tensor, optional): Params for page attention. Default: ``None``.
            slot_mapping (Tensor, optional): Params for page attention. Default: ``None``.
            prefill (bool, optional): Whether to do prefill predict or decode predict. Default: ``None``.
            **model_kwargs (Any): Keyword arguments of the model.

        Returns:
            res, the result after the forward process.
            current_index, records the current index of the sequence.
        """
        attention_mask = None
        gather_decode = True
        if isinstance(self.config.parallel_decoding_params, Dict):
            plugin_type = self.config.parallel_decoding_params.get("plugin_type")
        else:
            plugin_type = None
        if plugin_type == "la":
            slot_mapping, attention_mask = la_pre_process(input_ids,
                                                          slot_mapping,
                                                          **model_kwargs)
            model_kwargs["attention_mask"] = attention_mask
            # lookahead should not gather decode logits
            gather_decode = False
        model_inputs, prefill = self.prepare_inputs_for_generation_mcore(
            input_ids=input_ids,
            valid_length_each_example=valid_length_each_example,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            prefill=prefill,
            **model_kwargs,
        )
        res = self._incremental_infer_mcore(
            model_inputs=model_inputs,
            prefill=prefill,
            gather_decode=gather_decode
        )
        return res, None

    def infer_mcore(self,
                    input_ids: Union[List[int], List[List[int]]],
                    valid_length_each_example: np.ndarray,
                    generation_config: GenerationConfig = None,
                    logits_processor: Optional[LogitsProcessorList] = None,
                    logits_warper: Optional[LogitsProcessorList] = None,
                    block_tables: Optional[Tensor] = None,
                    slot_mapping: Optional[Tensor] = None,
                    prefill: bool = True,
                    is_finished: List[bool] = None,
                    **model_kwargs):
        """
        Do infer and return logits on next position, can choose do prefill or decode predict.

        Args:
            input_ids (List(List(int))): Input ids after padding.
            valid_length_each_example (np.ndarray): Valid input length except padding.
            generation_config (`GenerationConfig`, optional): The generation configuration to be used
                as base parametrization for the generation call. Default: ``None``.
            logits_processor (`LogitsProcessorList`, optional): An instance of [`LogitsProcessorList`].
                List of instances of class derived from [`LogitsProcessor`] used to modify the prediction scores
                of the language modeling head applied at each generation step. Default: ``None``.
            logits_warper (`LogitsProcessorList`, optional): An instance of [`LogitsProcessorList`].
                List of instances of class derived from [`LogitsWarper`] used to warp the prediction score
                distribution of the language modeling head applied before multinomial sampling
                at each generation step. Default: ``None``.
            block_tables (Tensor, optional): Store mapping tables for each sequence. Default: ``None``.
            slot_mapping (Tensor, optional): Token cache physical slot index. Default: ``None``.
            prefill (bool, optional): Whether to do prefill predict or decode predict. Default: ``True``.
            is_finished (List(bool), optional): Whether each sequence is finished its generation. Default: ``None``.
            **model_kwargs (Any): Keyword arguments of the model.

        Returns:
            next_token, the next token to be generated.
            is_finished, whether the sequence has completed its generation task.
        """
        max_valid_length = max(valid_length_each_example)
        if max_valid_length > self.config.seq_length:
            raise ValueError(
                f"The input length:{max_valid_length} is longer than the seq_length:{self.config.seq_length}, "
                "which is not allowed."
            )

        start_time = time.time()
        flatten_input_ids, slot_mapping = self._prepare_inputs_for_flatten(
            input_ids, valid_length_each_example, slot_mapping, prefill
        )
        res, current_index = self.forward_mcore(
            input_ids=flatten_input_ids,
            valid_length_each_example=valid_length_each_example,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            prefill=prefill,
            **model_kwargs,
        )

        self.detailed_latency.start_postprocess_timer()
        forward_time = time.time() - start_time
        sample_time = time.time()
        target_list, probs, logits, is_finished = self.postprocess(
            input_ids=input_ids,
            is_finished=is_finished,
            res=res,
            current_index=current_index,
            generation_config=generation_config,
            valid_length_each_example=valid_length_each_example,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            need_gather_logits=False,
        )

        sample_time = time.time() - sample_time
        infer_time = time.time() - start_time
        logger.debug("forward time: %s s; sample time: %s s; total count: %s s",
                     forward_time, sample_time, infer_time)

        if generation_config.return_dict_in_generate:
            infer_output_dict = InferOutput(
                target_list=target_list,
                probs=probs,
                logits=logits
            )
            return infer_output_dict, is_finished

        return target_list, is_finished

    @staticmethod
    def _prepare_inputs_for_flatten(input_ids, valid_length_each_example, slot_mapping, prefill=True):
        """prepare inputs ids for prefill flatten"""
        input_ids = np.array(input_ids)
        batch_valid_length_bs = valid_length_each_example.shape[0]
        if prefill:
            input_ids_list = []
            for i in range(batch_valid_length_bs):
                input_ids_list.append(input_ids[i][:valid_length_each_example[i]])
            input_ids = np.concatenate(input_ids_list, 0)
            slot_mapping = np.delete(slot_mapping, np.where(slot_mapping == -1))
        else:
            batch_valid_length_bs = valid_length_each_example.shape[0]
            input_ids_list = []
            for i in range(batch_valid_length_bs):
                input_ids_list.append(input_ids[i][valid_length_each_example[i] - 1])
            input_ids = np.array(input_ids_list)
        input_ids = input_ids.reshape((-1))
        return input_ids, slot_mapping

    # pylint: disable=E1102
    def chunk_prefill_infer(self,
                            input_ids: [Union[List[int], List[List[int]]]],
                            batch_valid_length: np.ndarray,
                            block_tables: np.ndarray,
                            slot_mapping: np.ndarray,
                            attention_mask: Optional[np.ndarray] = None,
                            **model_kwargs):
        """
        Preprocessing of chunk prefill inference

        Args:
            input_ids (List(List(int))): Input ids.
            batch_valid_length (np.ndarray): Valid input length.
            block_tables (np.ndarray): Params for page attention.
            slot_mapping (np.ndarray): Params for page attention.
            attention_mask (np.ndarray): Params for page attention.
            q_seq_lens (np.ndarray): Params for page attention.
            gather_index (np.ndarray): Used to obtain the last latent vector of each sequence.
            seq_range (np.ndarray): Used to obtain Mask and positional encoding of valid tokens for each sequence.
        """
        if not (self.use_past and self.chunk_prefill):
            raise ValueError(f"chunk prefill infer can be called only when use_past=true and chunk_prefill=true, \
                but use_past={self.use_past}, chunk_prefill={self.chunk_prefill}")
        # decode
        if "gather_index" not in model_kwargs or "seq_range"not in model_kwargs \
            or "q_seq_lens" not in model_kwargs:
            model_kwargs["gather_index"] = None
            model_kwargs["seq_range"] = None
            model_kwargs["q_seq_lens"] = None
            self.add_flags_custom(is_first_iteration=False)
        else: # decode + chunk
            input_ids = np.reshape(input_ids, (1, -1))
            model_kwargs["gather_index"] = Tensor(model_kwargs["gather_index"], ms.int32)
            model_kwargs["seq_range"] = Tensor(model_kwargs["seq_range"], ms.int32)
            model_kwargs["q_seq_lens"] = Tensor(model_kwargs["q_seq_lens"], ms.int32)
            self.add_flags_custom(is_first_iteration=True)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = Tensor(attention_mask, ms.float16)

        model_kwargs["input_ids"] = Tensor(input_ids, ms.int32)
        model_kwargs["batch_valid_length"] = Tensor(batch_valid_length, ms.int32)
        model_kwargs["block_tables"] = Tensor(block_tables, ms.int32)
        model_kwargs["slot_mapping"] = Tensor(slot_mapping, ms.int32)

        logits = self(**model_kwargs)

        return logits

    def postprocess(self,
                    input_ids,
                    is_finished,
                    res,
                    generation_config: GenerationConfig,
                    valid_length_each_example,
                    current_index: Optional[Union[List[int], List[List[int]]]],
                    logits_processor: Optional[LogitsProcessorList] = None,
                    logits_warper: Optional[LogitsProcessorList] = None,
                    need_gather_logits: bool = True):
        r"""
        Postprocess of the output from model generation.

        Args:
            input_ids (List(List(int))): Input ids after padding.
            res (List(List(int))): Logits after infer.
            is_finished (List(bool)): Whether each sequence is finished its generation.
            generation_config (`GenerationConfig`): The generation configuration to be used
                as base parametrization for the generation call.
            valid_length_each_example (np.ndarray): Valid input length except padding.
            current_index (List(int)): Current index of sequence.
            logits_processor (`LogitsProcessorList`, optional): An instance of [`LogitsProcessorList`].
                List of instances of class derived from [`LogitsProcessor`] used to modify the prediction scores
                of the language modeling head applied at each generation step. Default: ``None``.
            logits_warper (`LogitsProcessorList`, optional): An instance of [`LogitsProcessorList`]. List of
                instances of class derived from [`LogitsWarper`] used to warp
                the prediction score distribution of the language modeling head applied
                before multinomial sampling at each generation step. Default: ``None``.
            need_gather_logits (bool, optional): whether gather result, when decode predict and is first iteration.
                Default: ``True``.

        Returns:
            target_list, contains the target values generated in each batch.
            next_probs_cache, cache for probs, if needed in output.
            next_logits_cache, cache for logits, if needed in output.
            is_finished, whether the sequence has completed its generation task.
        """
        if not self.is_pynative:
            from mindspore.common.api import _pynative_executor
            _pynative_executor.set_async_for_graph(True)
        batch_size = input_ids.shape[0]
        target_list = [[] for _ in range(batch_size)]

        # cache for logits and probs, if needed in output
        next_logits_cache = None
        next_probs_cache = None

        generation_config.generation_mode = self._get_generation_mode(generation_config)
        if generation_config.generation_mode == GenerationMode.GREEDY_SEARCH:
            if not self.config.is_sample_acceleration:
                logits = res[0] if isinstance(res, tuple) else res
                logits = logits.reshape(-1, logits.shape[-1])
                if need_gather_logits and logits.shape[0] > len(current_index):
                    logits = logits[Tensor(current_index, dtype=mstype.int32)]
                # store caced logits
                if generation_config.return_dict_in_generate and generation_config.output_logits:
                    if isinstance(logits, Tensor):
                        next_logits_cache = logits.asnumpy().copy()
                    else:
                        next_logits_cache = logits.copy()
                if logits_processor:
                    if isinstance(logits, Tensor):
                        logits = logits.asnumpy()
                    logits = Tensor(logits_processor(input_ids, logits, is_finished))
                # store caced probs
                if generation_config.return_dict_in_generate and generation_config.output_scores:
                    if isinstance(logits, Tensor):
                        next_probs_cache = logits.asnumpy().copy()
                    else:
                        next_probs_cache = logits.copy()
                target_list = self.argmax(logits, -1)
                target_list = target_list.asnumpy().tolist()
            else:
                probs, p_args = res
                if isinstance(p_args, Tensor):
                    p_args = p_args.asnumpy()
                # store caced probs
                if generation_config.return_dict_in_generate and generation_config.output_scores:
                    if isinstance(probs, Tensor):
                        next_probs_cache = probs.asnumpy().copy()
                    else:
                        next_probs_cache = probs.copy()
                target_index_list = P.Argmax()(probs)
                target_index_list = target_index_list.asnumpy().tolist()
                # run greedy search
                for i in range(batch_size):
                    if is_finished[i]:
                        continue
                    target_index = target_index_list[i]
                    target = p_args[i][target_index]
                    target_list[i] = target

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
                # store caced logits
                if generation_config.return_dict_in_generate and generation_config.output_logits:
                    next_logits_cache = logits.copy()
                probs = logits_processor(input_ids, logits, is_finished)
                p_args = np.tile(np.arange(logits.shape[-1]), (batch_size, 1))
                probs = logits_warper(input_ids, probs, is_finished)
            else:
                probs, p_args = res
                if isinstance(probs, Tensor):
                    probs = probs.asnumpy()
                if isinstance(p_args, Tensor):
                    p_args = p_args.asnumpy()
            # store caced probs
            if generation_config.return_dict_in_generate and generation_config.output_scores:
                next_probs_cache = probs.copy()
            p_norms = softmax_with_threads(probs, is_finished)

            for i in range(batch_size):
                if is_finished[i]:
                    continue
                p_norm = p_norms[i]
                target_index = np.random.choice(len(probs[i]), p=p_norm)
                # get target token id
                target = p_args[i][target_index]
                target_list[i] = target

        elif generation_config.generation_mode == GenerationMode.BEAM_SEARCH:
            raise ValueError("sampler method doesn't support BEAM_SEARCH. ")
        if not self.is_pynative:
            from mindspore.common.api import _pynative_executor
            _pynative_executor.sync()
            _pynative_executor.set_async_for_graph(False)
        return target_list, next_probs_cache, next_logits_cache, is_finished

    def chat(self,
             tokenizer: PreTrainedTokenizer,
             query: str,
             history: Optional[List[Dict[str, str]]] = None,
             system_role_name: Optional[str] = "system",
             user_role_name: Optional[str] = "user",
             assistant_role_name: Optional[str] = "assistant",
             instruction: Optional[str] = "",
             max_length: Optional[int] = 512,
             max_new_tokens: Optional[int] = None,
             min_length: Optional[int] = 0,
             min_new_tokens: Optional[int] = None,
             do_sample: Optional[bool] = True,
             temperature: Optional[float] = 1.0,
             top_k: Optional[int] = 50,
             top_p: Optional[float] = 1.0,
             repetition_penalty: Optional[float] = 1.0):
        r"""
        Dia-logical text generation inference with large language models. The query from the user will be inference
        using generate() after adding the chat template via the provided tokenizer.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenized used to decode the tokens.
            query (str): User input for inference.
            history (List[Dict[str, str]], optional): A Conversation object or list of dicts with "role"
                and "content" keys, representing the chat history so far. Default: ``None``.
            system_role_name (str, optional): The name of system role. Default: ``"system"``.
            user_role_name (str, optional): The name of user role. Default: ``"user"``.
            assistant_role_name (str, optional): The name of assistant role. Default: "assistant".
            instruction (str, optional): Instruction message to the model. Default: ``""``.
            max_length (int, optional): The maximum length the generated tokens can have.
                Corresponds to the length of the input prompt + `max_new_tokens`.
                Its effect is overridden by `max_new_tokens`, if also set. Default: ``512``.
            max_new_tokens (int, optional): The maximum numbers of tokens to generate, ignoring the number of
                tokens in the prompt. Default: ``None``.
            min_length (int, optional): The minimum length of the sequence to be generated.
                Corresponds to the length of the input prompt + `min_new_tokens`.
                Its effect is overridden by `min_new_tokens`, if also set. Default: 0.
            min_new_tokens (int, optional): The minimum numbers of tokens to generate,
                ignoring the number of tokens in the prompt. Default: ``None``.
            do_sample (bool, optional): Whether to do sampling on the candidate ids. If set True it will be enabled,
                and set it to be False to disable the sampling, equivalent to top-k 1.
                If set None, it follows the setting in the configuration in the model. Default: ``True``.
            temperature (float, optional): The value used to modulate the next token probabilities. Default: ``1.0``.
            top_k (int, optional): Determine the top-k numbers token id as candidate. This should be a positive number.
                If set None, it follows the setting in the configuration in the model. Default: ``50``.
            top_p (float, optional): The accumulation probability of the candidate token ids below the top-p
                will be select as the candidate ids. The valid value of top-p is between (0, 1].
                If the value is larger than 1, top-k algorithm will be enabled.
                If set None, it follows the setting in the configuration in the model. Default: ``1.0``.
            repetition_penalty (float, optional): The penalty factor of the frequency that generated words.
                If set 1, the repetition_penalty will not be enabled.
                If set None, it follows the setting in the configuration in the model. Default: ``1.0``.

        Returns:
            response, the reply from the LLM in this session.
            history, the conversation history.
        """
        if history is None:
            history = []
            if instruction:
                history.append({"role": system_role_name, "content": instruction})

        history.append({"role": user_role_name, "content": query})
        input_ids = tokenizer.apply_chat_template(conversation=history, add_generation_prompt=True)
        output_ids = self.generate(input_ids=input_ids,
                                   max_length=max_length,
                                   max_new_tokens=max_new_tokens,
                                   min_length=min_length,
                                   min_new_tokens=min_new_tokens,
                                   do_sample=do_sample,
                                   temperature=temperature,
                                   top_k=top_k,
                                   top_p=top_p,
                                   repetition_penalty=repetition_penalty)
        output_ids = output_ids[0][len(input_ids):]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        history.append({"role": assistant_role_name, "content": response})
        return response, history

    def convert_pin_model_inputs(self, model_inputs):
        if os.environ.get("EXPERIMENTAL_KERNEL_LAUNCH_GROUP", None):
            model_inputs["input_ids"] = convert_pin(model_inputs["input_ids"])
            model_inputs["batch_valid_length"] = convert_pin(model_inputs["batch_valid_length"])
            model_inputs["block_tables"] = convert_pin(model_inputs["block_tables"])
            model_inputs["slot_mapping"] = convert_pin(model_inputs["slot_mapping"])
        return model_inputs
