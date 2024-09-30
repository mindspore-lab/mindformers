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
from mindformers.experimental.parallel_core.pynative.transformer.enums import AttnMaskType

attn_mask_type_mapping = {
    "padding": AttnMaskType.padding,
    "causal": AttnMaskType.causal,
}

class Llama2Model(Module):
    """
    Llama2 Model

    Args:
        - **config** : model config
        - **num_tokentypes** : if > 0, using tokentypes embedding
        - **parallel_output** : Specifies whether return paralleled output on each tensor parallel rank.
        - **pre_process** : when using pipeline parallel, indicate whether it's the first stage
        - **post_process** : when using pipeline parallel, indicate whether it's the last stage
        - **loss_func** : loss function

    Supported Platforms:
        ``Ascend``

    """
    # pylint: disable=W0613
    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 loss_func=None,
                 **kwargs):
        super().__init__(config, **kwargs)
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.untie_embeddings_and_output_weights = config.untie_embeddings_and_output_weights
        self.fp16_lm_cross_entropy = config.fp16_lm_cross_entropy
        self.seq_length = config.seq_length
        self.pad_token_id = config.dataset_config.pad_token_id
        self.compute_dtype = config.compute_dtype
        encoder_attn_mask_type = None
        if config.encoder_attn_mask_type is not None:
            encoder_attn_mask_type = attn_mask_type_mapping.get(config.encoder_attn_mask_type)
            if encoder_attn_mask_type is None:
                raise ValueError(f"encoder_attn_mask_type must be one of {attn_mask_type_mapping.keys()}, but got"
                                 f"{config.encoder_attn_mask_type}")

        self.language_model, _ = get_language_model(config,
                                                    encoder_attn_mask_type=encoder_attn_mask_type,
                                                    num_tokentypes=num_tokentypes,
                                                    pre_process=self.pre_process,
                                                    post_process=self.post_process,
                                                    add_pooler=False)
        if self.post_process:
            self.head = ParallelLMLogits(config=config,
                                         bias=False,
                                         compute_dtype=config.compute_dtype)
            self.loss = LossWithMask(VocabParallelCrossEntropy())

        if not self.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        """ set input_tensor to model """
        self.language_model.set_input_tensor(input_tensor)

    def construct(self, input_ids, attention_mask,
                  labels=None, loss_mask=None, tokentype_ids=None, inference_params=None):
        """ llama model forward """
        if inference_params is not None:
            raise NotImplementedError("inference_params is not supported for now.")
        # use RoPE
        position_ids = None
        hidden_states = self.language_model(input_ids,
                                            position_ids,
                                            attention_mask,
                                            tokentype_ids=tokentype_ids,
                                            inference_params=inference_params)
        # pylint: disable=R1705
        if self.post_process:
            if self.untie_embeddings_and_output_weights:
                logit_weights = self.language_model.output_layer.weight
            else:
                logit_weights = self.shared_embedding_or_output_weight()
            # pylint: disable=E1121
            loss, logits, tokens_nums = self.post_language_model_processing(hidden_states,
                                                                            labels,
                                                                            logit_weights,
                                                                            loss_mask)
            return loss, logits, tokens_nums
        else:
            return hidden_states

    def post_language_model_processing(self,
                                       lm_output,
                                       labels,
                                       logit_weights,
                                       loss_mask):
        """ pangu model post process forward """
        logits = self.head(lm_output, logit_weights, self.parallel_output)

        # flatten logits
        logits = logits.reshape(-1, logits.shape[-1])

        if labels is None:
            return logits

        if self.fp16_lm_cross_entropy:
            logits = logits.astype(mstype.float16)
            loss_mask = loss_mask.astype(mstype.float16)
        else:
            logits = logits.astype(mstype.float32)
        labels = labels.reshape(-1).astype(mstype.int32)
        loss_mask = loss_mask.reshape(-1)
        tokens_nums = loss_mask.sum()
        loss = self.loss(logits, labels, loss_mask)
        return loss, logits, tokens_nums
