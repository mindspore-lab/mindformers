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
""" LLama2 Model """

import mindspore.common.dtype as mstype
from mindformers.experimental.parallel_core.pynative.transformer.module import Module
from mindformers.experimental.parallel_core.pynative.transformer.language_model import get_language_model
from mindformers.experimental.parallel_core.pynative.transformer import ParallelLMLogits
from mindformers.experimental.parallel_core.pynative.training.loss_func import LossWithMask
from mindformers.experimental.parallel_core.pynative.tensor_parallel import VocabParallelCrossEntropy


def post_language_model_processing(parallel_lm_logits, loss_fn, lm_output, labels, logit_weights,
                                   parallel_output, fp16_lm_cross_entropy, loss_mask):
    """ gpt model post process forward """
    output = parallel_lm_logits(lm_output, logit_weights, parallel_output)

    if labels is None:
        # [s b h] -> [b s h]
        return output.swapaxes(0, 1)

    # [b s] -> [s b]
    labels = labels.swapaxes(0, 1)
    loss_mask = loss_mask.reshape(-1)

    if fp16_lm_cross_entropy:
        if output.dtype != mstype.float16:
            raise ValueError(f"When fp16_lm_cross_entropy=True, output should be float16, but got {output.dtype}")
        loss = loss_fn(output, labels, loss_mask)
    else:
        loss = loss_fn(output.astype(mstype.float32), labels, loss_mask)

    return loss


class GPTModel(Module):
    """
    GPT Model

    Args:
        - **config** : model config
        - **num_tokentypes** : if > 0, using tokentypes embedding
        - **parallel_output** : Specifies whether return paralleled output on each tensor parallel rank.
        - **pre_process** : when using pipeline parallel, indicate whether it's the first stage
        - **post_process** : when using pipeline parallel, indicate whether it's the last stage

    Supported Platforms:
        ``Ascend``

    """
    # pylint: disable=W0613
    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True):
        super().__init__(config=config,\
                         share_embeddings_and_output_weights=not config.untie_embeddings_and_output_weights)

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.untie_embeddings_and_output_weights = config.untie_embeddings_and_output_weights
        self.fp16_lm_cross_entropy = config.fp16_lm_cross_entropy

        # set model key
        self.set_model_key()

        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=None,
            pre_process=self.pre_process,
            post_process=self.post_process)

        if self.post_process:
            self.parallel_lm_logits = ParallelLMLogits(config=config,
                                                       bias=False,
                                                       compute_dtype=config.compute_dtype)
            self.loss = LossWithMask(VocabParallelCrossEntropy())

        if not config.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        """ set input_tensor to model """
        self.language_model.set_input_tensor(input_tensor)

    def set_model_key(self):
        """ set model key for differentiate PipelineCell process """
        self.model_key = "gpt3"

    def construct(self, input_ids, position_ids, attention_mask, loss_mask,
                  retriever_input_ids=None,
                  retriever_position_ids=None,
                  retriever_attn_mask=None,
                  labels=None, tokentype_ids=None, inference_params=None):
        """ gpt model forward """
        # use RoPE
        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            retriever_input_ids=retriever_input_ids,
            retriever_position_ids=retriever_position_ids,
            retriever_attn_mask=retriever_attn_mask,
            inference_params=inference_params)
        # pylint: disable=R1705
        if self.post_process:
            # pylint: disable=E1121
            return post_language_model_processing(
                self.parallel_lm_logits, self.loss,
                lm_output, labels,
                self.language_model.output_layer.weight if\
                    self.untie_embeddings_and_output_weights else self.shared_embedding_or_output_weight(),
                self.parallel_output,
                self.fp16_lm_cross_entropy,
                loss_mask)
        else:
            return lm_output
