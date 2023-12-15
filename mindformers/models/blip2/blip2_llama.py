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
# This file was refer to project:
# https://github.com/salesforce/LAVIS/tree/main/lavis/models/blip2_models
# ============================================================================
"""Blip2 second stage pretrain and infer with Llama model"""

import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops import operations as P

from mindformers.models.blip2.layers import ImageTextEmbeddingPreparationMixIn
from mindformers.models.llama.llama import LlamaForCausalLM, LlamaModel
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class LlamaModelForBlip2(LlamaModel):
    r"""
        Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`],
        it inherits from LlamaModel and is used to adapt blip2.
        Args:
            config(LlamaConfig): the config of network

        Inputs:
            input_ids: the tokenized inputs with datatype int32

        Returns:
            output: Tensor, the output of llama decoder layer
        """

    def __init__(self, config):
        super().__init__(config)

    # pylint: disable=W0221
    def construct(self, input_embeddings: Tensor, batch_valid_length=None):

        bs, seq_len, _ = input_embeddings.shape
        if not self.use_past:
            freqs_cis = self.freqs_mgr()
            mask = self.casual_mask(input_embeddings) # mask: [bs, seq, seq]
            mask = self.casual_mask.post_process(mask)
            kvcache_inputs = None
        else:
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr(bs, seq_len)
                mask = self.casual_mask(input_embeddings) # mask: [bs, seq, seq]
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length, bs)
                if self.is_dynamic and self.is_flexible_shape and not self.use_kvcache_op:
                    mask = self.casual_mask.increment_slice(self.kvcache_preprocess.range,
                                                            self.kvcache_preprocess.max_cache_length // bs,
                                                            batch_valid_length)
                else:
                    mask = self.casual_mask.increment(self.kvcache_preprocess.range, batch_valid_length)
            mask = self.casual_mask.post_process(mask)

            kvcache_inputs = self.kvcache_preprocess(bs, batch_valid_length)

        # tokens: [bs, seq/1]
        h = input_embeddings
        h = self.reshape(h, (bs, seq_len, self.hidden_size))
        # h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            h = self.layers[i](h, freqs_cis, mask, kvcache_inputs=kvcache_inputs)
        output = self.norm_out(h)
        return output


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class LlamaForBlip2(LlamaForCausalLM, ImageTextEmbeddingPreparationMixIn):
    r"""
        Provide llama inference loss or logits through network for blip2.
        Args:
            config (LlamaConfig): The config of llama model.`

        Inputs:
            input_embeddings(Tensor): the embeddings of inputs, shape :math:`(batch, seq\_length, hidden\_size)`
            input_ids(Tensor): the input ids of inputs, shape: math:`(batch, seq\_length)`
            label_ids(Tensor): Labels for computing loss. Tensor of shape :math:`(batch, seq\_length)`
            attention_mask(Tensor): attention_mask, used by model.construct

        Returns:
            Tensor, the loss of the network.

        Examples:
            >>> from mindformers.models.llama import LlamaConfig
            >>> from mindformers.models.blip2.blip2_llama import LlamaForBlip2
            >>> config = LlamaConfig(batch_size=1)
            >>> network = LlamaForBlip2(config=config)
            >>> type(network)
            <class 'mindformers.models.blip2.blip2_llama.LlamaForBlip2'>
        """

    def __init__(self, config: LlamaConfig = None):
        checkpoint_name_or_path = config.checkpoint_name_or_path
        config.checkpoint_name_or_path = ""

        LlamaForCausalLM.__init__(self, config=config)
        ImageTextEmbeddingPreparationMixIn.__init__(self, config=config)

        self.model = LlamaModelForBlip2(config)

        self.config.checkpoint_name_or_path = checkpoint_name_or_path
        self.load_checkpoint(config)

        dp = config.parallel_config.data_parallel

        self.mul_1d = P.Mul().shard(((1,), (1,)))
        self.concat_2d = P.Concat(axis=1).shard(((dp, 1), (dp, 1)))
        self.concat_3d = P.Concat(axis=1).shard(((dp, 1, 1), (dp, 1, 1)))
        self.not_equal_1d = P.NotEqual().shard(((1,), ()))
        self.ones = P.Ones().shard(((dp, 1),))
        self.cast_1d = P.Cast().shard(((1,), (1,)))
        self.cast = P.Cast()
        self.gather = P.Gather().shard(((1, 1), (1,)))

        self.is_first_iteration = True
        self.use_past = self.config.use_past

    def to_text_embeddings(self, text_input_ids):
        return self.model.tok_embeddings(text_input_ids)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.prepare_image_text_embedding(input_ids, **kwargs)

    # pylint: disable=W0221
    def construct(self, input_embeddings=None,
                  input_ids=None,
                  labels=None,
                  input_position=None,
                  attention_mask=None,
                  init_reset=True,
                  batch_valid_length=None):
        """LlamaForBlip2 forward."""
        if input_embeddings is None and input_ids is not None:  # for incremental infer
            input_embeddings = self.model.tok_embeddings(input_ids)

        output = self.model(input_embeddings=input_embeddings,
                            input_attention_masks=attention_mask,
                            input_position=input_position,
                            init_reset=init_reset,
                            batch_valid_length=batch_valid_length)
        logits = self.lm_head(output)
        logits = self.cast(logits, mstype.float32)

        if labels is None:
            # inference
            logits = self.reshape(logits, (-1, logits.shape[-1]))
            if (self.is_first_iteration or not self.use_past) and input_position is not None:
                logits = self.gather(logits, input_position, 0)
            return logits

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))

        labels = self.reshape(labels, (-1,))
        label_mask = self.cast_1d(self.not_equal_1d(labels, self.ignore_token_id), mstype.float32)
        attention_mask = self.reshape(attention_mask, (-1,))
        input_mask = self.mul_1d(attention_mask, label_mask)
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss
