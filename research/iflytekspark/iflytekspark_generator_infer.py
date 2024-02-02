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
"""Text Generator Infer."""
from typing import Union, List, Optional
from threading import Thread

import numpy as np
from mindformers.inference.infer_config import InferConfig
from mindformers.tools.logger import logger
from mindformers.models import BaseImageProcessor, BaseTokenizer
from mindformers.generation import GenerationConfig, LogitsProcessorList
from mindformers.generation.logits_process import RepetitionPenaltyLogitsProcessor, LogitNormalization, \
    TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from mindformers.generation.streamers import BaseStreamer
from mindformers.generation.utils import softmax
from mindformers.inference.infers.base_infer import BaseInfer
import mindspore_lite as mslite


__all__ = ['IFlytekSparkGeneratorInfer']


class DynSeqInputsOfInfer():
    """
    infer inputs of IFlytekSpark models.
    """
    def get_inputs(self, input_ids=None, current_index=None, valid_length=None,
                   init_reset=None, is_first_iteration=True, kv_cache_size=0):
        """Get inputs"""
        if not is_first_iteration:
            inputs_tmp = []
            for i in range(len(current_index)):
                current_index_tmp = int(current_index[i]) - i * input_ids.shape[1]  # multibatch
                inputs_tmp.append(input_ids[i][current_index_tmp:current_index_tmp + 1])
            input_ids = np.array(inputs_tmp, dtype=np.int32) # 只取上一个生成的id
        elif 0 < kv_cache_size < input_ids.shape[1]:
            pass

        input_ids = input_ids.astype(np.int32)
        current_index = current_index.astype(np.int32)
        init_reset = init_reset.astype(np.bool_)
        valid_length = valid_length.astype(np.int32)
        inputs = [input_ids, current_index, init_reset, valid_length]

        lite_inputs = []
        for input_np in inputs:
            lite_inputs.append(mslite.Tensor(input_np))
        return lite_inputs

class IFlytekSparkGeneratorInfer(BaseInfer):
    """
    Text generator infer implement class.
    """
    def __init__(self,
                 config: InferConfig = None,
                 tokenizer: Optional[BaseTokenizer] = None,
                 image_processor: Optional[BaseImageProcessor] = None):
        super().__init__(config, tokenizer, image_processor)
        self.input_prepare = DynSeqInputsOfInfer()
        self.is_dynamic = config.is_dynamic

    def infer(self,
              inputs: Union[str, List[str]],
              do_sample: bool = False,
              top_k: int = 1,
              top_p: float = 1.0,
              temperature: float = 1.0,
              repetition_penalty: float = 1.0,
              repetition_penalty_increase: float = 0.1,
              eos_token_id: int = 2,
              pad_token_id: int = 0,
              max_length: int = 256,
              add_special_tokens: bool = False,
              streamer: Optional[BaseStreamer] = None,
              **kwargs):
        """Inference."""
        input_ids = self.preprocess(inputs, add_special_tokens)
        output_ids = self.generate(input_ids, do_sample, top_k, top_p, temperature,
                                   repetition_penalty, repetition_penalty_increase, eos_token_id, pad_token_id,
                                   max_length, streamer, **kwargs)
        outputs = self.postprocess(output_ids)
        return outputs

    # pylint: disable=W0613
    def preprocess(self, input_data, add_special_tokens=False, **kwargs):
        """preprocess."""
        tokens = self.tokenizer(input_data, add_special_tokens=add_special_tokens)
        input_ids = tokens["input_ids"]
        input_list = []
        if isinstance(input_data, str):
            input_list.append(input_ids)
        else:
            input_list = input_ids
        return input_list

    # pylint: disable=W0613
    def postprocess(self, predict_data, **kwargs):
        """postprocess."""
        outputs = self.tokenizer.batch_decode(predict_data, skip_special_tokens=False)
        return outputs

    def _get_logits_processor(self,
                              generation_config: GenerationConfig,
                              logits_processor: Optional[LogitsProcessorList]):
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        # instantiate processors list
        processors = LogitsProcessorList()

        if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty=generation_config.repetition_penalty))
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

    def generate(self,
                 input_ids,
                 do_sample,
                 top_k,
                 top_p,
                 temperature,
                 repetition_penalty,
                 repetition_penalty_increase,
                 eos_token_id,
                 pad_token_id,
                 max_length,
                 streamer,
                 kv_cache_size,
                 **kwargs):
        """token generator."""
        sampler_dict = {"do_sample": do_sample,
                        "top_k": top_k,
                        "top_p": top_p,
                        "temperature": temperature,
                        "repetition_penalty": repetition_penalty,
                        "repetition_penalty_increase": repetition_penalty_increase,
                        "max_length": max_length, **kwargs}
        generation_config = GenerationConfig(**sampler_dict)
        if not generation_config.do_sample:
            generation_config.top_p = 1.0
            generation_config.top_k = 0
        generation_config.repetition_penalty_increase = repetition_penalty_increase
        self.is_sample_model = generation_config.top_p == 1.0 and generation_config.do_sample

        batch_size = len(input_ids)
        valid_length = []
        for i in range(batch_size):
            # As the nonzero returns the index and we need length
            valid_length.append(np.max(np.argwhere(np.array(input_ids[i]) != pad_token_id)) + 1)
        valid_length = np.array(valid_length, np.int32)
        batch_max_valid_len = np.max(valid_length)

        if int(batch_max_valid_len) >= self.seq_length:
            print("error: input length is greater than seq_length, please input a shorter string", flush=True)
            return []
        max_length += int(batch_max_valid_len)

        target_length = self.seq_length-1 if max_length > self.seq_length else max_length
        # pad original input ids to seq_length
        if self.is_dynamic:
            pad_length = batch_max_valid_len - valid_length
        else:
            pad_length = self.seq_length - valid_length
        pad_input_ids = np.array([
            np.pad(input_ids[i], (0, pad_length[i]),
                   'constant', constant_values=(pad_token_id)) for i in range(len(input_ids))
        ], np.int32)
        input_ids = pad_input_ids

        # repetition_penalty update cache
        repetition_penalty_cache = None

        # setup is_first_iteration flag for incremental infer
        is_first_iteration = True
        is_finished = [False] * batch_size
        while np.sum(is_finished) != batch_size:
            seq_length = input_ids.shape[1] # batch_max_valid_len
            current_index = [valid_length[i] - 1 + i * seq_length for i in range(batch_size)] # [5, 5+2048]
            current_index = np.array(current_index, np.int32)
            logger.debug("validate length: %s", valid_length)

            outputs = self._inc_infer(input_ids, current_index, valid_length,
                                      is_first_iteration, kv_cache_size)

            logits = outputs[0].get_data_to_numpy()
            vocab_size = logits.shape[-1]
            if len(logits.shape) < 3:
                logits = logits.reshape(batch_size, -1, vocab_size)

            # init repetition_penalty_cache with vocab_size
            if repetition_penalty_cache is None:
                repetition_penalty_cache = np.zeros(shape=(batch_size, vocab_size), dtype=np.float32)

            # rept + tempr + topk + topp
            logits = logits.reshape(-1, vocab_size)

            # using cpu parallel for top_p != 1.0 and no sample
            if not self.is_sample_model:
                batch_target = np.zeros((batch_size), dtype=int)
                self._parallel_sampler(generation_config, input_ids, logits, repetition_penalty_cache, batch_target)
                p_args = np.tile(np.arange(logits.shape[-1]), (batch_size, 1))
            else:
                # using cpu parallel to do repetition only, using lite infer to do temperature、topk, currently the sample model only support temperature、topk
                batch_prob = np.zeros((batch_size, vocab_size), dtype=np.float16)
                temperature = np.array([generation_config.temperature], dtype=np.float16)

                generation_config.temperature = 1.0 # remove temperature process for _parallel_sampler
                generation_config.top_k = 0 # remove topk process for _parallel_sampler
                self._parallel_sampler(generation_config, input_ids, logits, repetition_penalty_cache, batch_prob) # using parallel to do repetition only
                sample_inputs = [mslite.Tensor(batch_prob), mslite.Tensor(temperature)]
                sample_outputs = self.sample_model.predict(sample_inputs) # todo lite input, using lite to do temperature、topk
                p, p_args = sample_outputs[0].get_data_to_numpy(), sample_outputs[1].get_data_to_numpy()

            # pad one position for putting this generated token
            if self.is_dynamic:
                input_ids = np.pad(input_ids, ((0, 0), (0, 1)), 'constant', constant_values=(pad_token_id))
            stream_list = [[] for _ in range(batch_size)]
            for i in range(batch_size):
                if is_finished[i]:
                    continue

                if not self.is_sample_model:
                    target_index = batch_target[i]
                else:
                    if generation_config.do_sample:
                        # Random select a token as final output for this round, multinomial sample
                        target_index = np.random.choice(len(p[i]), p=p[i])
                    else:
                        # greedy
                        target_index = p[i, 0]

                # Stop judgment
                if valid_length[i] == target_length or p_args[i][target_index] == eos_token_id:
                    is_finished[i] = True
                    target = eos_token_id
                    stream_list[i] = [target]
                    valid_length[i] += int(1)
                    continue

                # update next iter input id
                target = p_args[i][target_index]
                input_ids[i, valid_length[i]] = target
                stream_list[i] = [target]
                valid_length[i] += int(1)

                # update repetition_penalty cache based on target
                if generation_config.repetition_penalty != 1.0:
                    old_penalty = repetition_penalty_cache[i][target]
                    # this word first appears
                    if old_penalty == 0:
                        repetition_penalty_cache[i][target] = generation_config.repetition_penalty
                    # appears more than one time
                    else:
                        repetition_penalty_cache[i][target] = old_penalty + \
                            generation_config.repetition_penalty_increase

            if streamer:
                streamer.put(stream_list)
            stream_list[i] = [[] for _ in range(batch_size)]
            is_first_iteration = False

        # Return valid outputs out of padded outputs
        if streamer:
            streamer.end()

        output_ids = []
        for i in range(batch_size):
            output_ids.append(input_ids[i, : int(valid_length[i])].astype(np.int32))
        logger.debug("The output is: %s", output_ids)
        return output_ids

    def _inc_infer(self, input_ids, current_index, valid_length, is_first_iteration, kv_cache_size):
        """kvcache infer"""
        if is_first_iteration:
            init_reset = np.array([False])
            lite_inputs = self.input_prepare.get_inputs(input_ids, current_index, valid_length, init_reset,
                                                        is_first_iteration, kv_cache_size)
            outputs = self.full_model.predict(lite_inputs)
        else:
            init_reset = np.array([True])
            lite_inputs = self.input_prepare.get_inputs(input_ids, current_index, valid_length, init_reset,
                                                        is_first_iteration, kv_cache_size)
            outputs = self.cache_model.predict(lite_inputs)
        return outputs

    # pylint: disable=W0108
    def _sampler(self, generation_config, input_id, logits, repetition_penalty_cache):
        """sampler"""
        if not generation_config.do_sample:
            generation_config.top_p = 1.0
            generation_config.top_k = 0

        # apply repetition_penalty
        log_probs = logits
        if generation_config.repetition_penalty != 1.0:
            # both shape are (1, vocab_size)
            log_probs -= repetition_penalty_cache

        if self.is_sample_model:
            return log_probs

        logits_warper = self._get_logits_warper(generation_config)
        p = logits_warper(input_id, log_probs)
        if generation_config.do_sample:
            p = softmax(p)
            target_index = np.apply_along_axis(
                lambda x: np.random.choice(len(x), p=x), axis=1, arr=p)
        else:
            target_index = np.apply_along_axis(
                lambda x: np.argmax(x), axis=1, arr=p)

        return target_index

    def _single_batch(self, i, generation_config, input_id, logits, repetition_penalty_cache, target):
        target[i] = self._sampler(generation_config, input_id, logits, repetition_penalty_cache)

    def _parallel_sampler(self, generation_config, input_ids, logits, repetition_penalty_cache, target):
        """parallel sampler"""
        all_threads = []
        for i in range(0, input_ids.shape[0]):
            thread = Thread(target=self._single_batch,
                            args=(i,
                                  generation_config,
                                  input_ids[i:i + 1],
                                  logits[i:i + 1],
                                  repetition_penalty_cache[i:i + 1],
                                  target)
                            )
            all_threads.append(thread)
            thread.start()
        for thread in all_threads:
            thread.join()
