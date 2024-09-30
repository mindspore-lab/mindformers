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
""" Pangu Model """

import copy
from mindspore import mint
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindformers.experimental.parallel_core.pynative.tensor_parallel import VocabParallelCrossEntropy
from mindformers.experimental.parallel_core.pynative.transformer.module import Module
from mindformers.experimental.parallel_core.pynative.transformer import TransformerLanguageModel, ParallelLMLogits, \
                                                            VocabParallelEmbedding, ParallelTransformerLayer, \
                                                            ParallelAttention

class PanguModel(Module):
    """
    Pangu Model

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
                 post_process=True,
                 **kwargs):
        super().__init__(config, **kwargs)
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.untie_embeddings_and_output_weights = config.untie_embeddings_and_output_weights
        self.fp16_lm_cross_entropy = config.fp16_lm_cross_entropy
        self.use_query_layer = config.use_query_layer
        self.seq_length = config.seq_length
        self.pad_token = config.dataset_config.eod_id
        self.compute_dtype = config.compute_dtype

        # init visual_encoder if use visual module
        visual_encoder = None
        if config.use_visual_encoder and self.pre_process:
            raise NotImplementedError("use_visual_encoder is not supported for now.")

        self.language_model = TransformerLanguageModel(config,
                                                       encoder_attn_mask_type=None,
                                                       num_tokentypes=num_tokentypes,
                                                       pre_process=self.pre_process,
                                                       post_process=self.post_process,
                                                       visual_encoder=visual_encoder)
        if self.post_process:
            self.head = ParallelLMLogits(config=config,
                                         bias=False,
                                         compute_dtype=config.compute_dtype)
            if self.parallel_output:
                self.loss = VocabParallelCrossEntropy()
            else:
                self.loss = nn.CrossEntropyLoss(reduction='none')

            if self.use_query_layer:
                self.query_layer = PanguQueryLayer(layer_number=1, config=config)

        if not self.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        """ set input_tensor to model """
        self.language_model.set_input_tensor(input_tensor)

    def construct(self, input_ids, position_ids, attention_mask,
                  retriever_input_ids=None, retriever_position_ids=None, retriever_attn_mask=None,
                  labels=None, tokentype_ids=None, inference_params=None,
                  input_image=None, delimiter_position=None, image_embedding=None,
                  loss_mask=None):
        """ pangu model forward """
        if retriever_input_ids is not None:
            raise NotImplementedError("dec_input_ids is not supported for now.")
        if retriever_position_ids is not None:
            raise NotImplementedError("dec_position_ids is not supported for now.")
        if retriever_attn_mask is not None:
            raise NotImplementedError("dec_attn_mask is not supported for now.")
        if inference_params is not None:
            raise NotImplementedError("inference_params is not supported for now.")

        if input_ids.shape[-1] == self.seq_length + 1 and labels is None:
            labels = input_ids[:, 1:]
            input_ids = input_ids[:, :-1]

        if loss_mask is None:
            if self.pad_token is None:
                raise RuntimeError("If 'pad_token' is not pass into model, the 'loss_mask' must be not None.")
            loss_mask = mint.ne(input_ids, self.pad_token).astype(self.compute_dtype)

        hidden_states = self.language_model(input_ids,
                                            position_ids,
                                            attention_mask,
                                            retriever_input_ids=retriever_input_ids,
                                            retriever_position_ids=retriever_position_ids,
                                            retriever_attn_mask=retriever_attn_mask,
                                            tokentype_ids=tokentype_ids,
                                            inference_params=inference_params,
                                            input_image=input_image,
                                            delimiter_position=delimiter_position,
                                            image_embedding=image_embedding)
        # pylint: disable=R1705
        if self.post_process:
            if self.untie_embeddings_and_output_weights:
                logit_weights = self.language_model.output_layer.weight
            else:
                logit_weights = self.shared_embedding_or_output_weight()
            loss, logits, tokens_nums = self.post_language_model_processing(hidden_states,
                                                                            labels,
                                                                            logit_weights,
                                                                            loss_mask,
                                                                            position_ids,
                                                                            attention_mask)
            return loss, logits, tokens_nums
        else:
            return hidden_states

    def post_language_model_processing(self,
                                       lm_output,
                                       labels,
                                       logit_weights,
                                       loss_mask,
                                       position_ids,
                                       attention_mask):
        """ pangu model post process forward """
        if self.use_query_layer:
            lm_output = self.query_layer(lm_output, position_ids, attention_mask)
        logits = self.head(lm_output, logit_weights, self.parallel_output)
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
        loss = self.loss(logits, labels)
        loss = mint.sum(loss * loss_mask) / tokens_nums
        return loss, logits, tokens_nums


class PanguQueryLayer(ParallelTransformerLayer):
    """
    Query Layer of the PanGUAlpha Model, which at the end of the network

    Args:
        layer_number(int): the index of the layer
        config(MindFormerConfig): the config of network
        drop_path_rate(float, optional): the rate of drop path

    Inputs:
        **hidden_states** (Tensor) - The input tensor, the shape is (batch_size, seq_length, hidden_size).
        **query_vector** (Tensor) - The query vector, the shape is (batch_size, seq_length, hidden_size).
        **attention_mask** (Tensor) - The attention mask, the shape is (batch_size, seq_length, seq_length).
        # **rotary_pos_emb** (Not used in Pangu Alpha) (Tensor, optional) - The rotary position embedding, the shape is (1, 1, seq_length, hidden_size / num_attention_heads).

    Outputs:
        **output** (Tensor) - The output tensor, the shape is (batch_size, seq_length, hidden_size).
    """

    def __init__(self, layer_number, config):
        super(PanguQueryLayer, self).__init__(config=config, layer_number=layer_number, drop_path_rate=0.0)
        attention_config = copy.deepcopy(config)
        reduce_scatter_embeddings = config.parallel_config.sequence_parallel
        if config.lora_config.use_lora:
            attention_config.update_lora_config(cell_name='attention')
        self.attention = ParallelAttention(
            layer_number=1, config=attention_config, attention_type="cross_attn"
        )
        self.query_embedding = VocabParallelEmbedding(
            num_embeddings=config.seq_length,
            embedding_dim=config.hidden_size,
            config=config.parallel_config,
            init_method=config.init_method,
            reduce_scatter_embeddings=reduce_scatter_embeddings,
            param_init_dtype=config.params_dtype
        )

    # pylint: disable=W0221
    def construct(self, hidden_states, input_position, attention_mask):
        """construct method"""
        # hidden_states: [B, S, H]

        # query vector
        query_vector = self.query_embedding(input_position)

        # normalization
        norm_output = self.input_norm(hidden_states)

        # attention
        # NOTICE: rotary_pos_emb is not used in Pangu Alpha
        attention_output, _ = self.attention(
            query_vector, attention_mask, norm_output, rotary_pos_emb=None
        )
        attention_output = self.hidden_states_dropout(attention_output)

        # residual connection
        if self.apply_residual_connection_post_norm:
            residual = norm_output
        else:
            residual = hidden_states
        norm_input = attention_output + residual

        # normalization
        norm_output = self.post_attention_norm(norm_input)

        # feedforward
        mlp_output = self.mlp(norm_output)
        mlp_output = self.hidden_states_dropout(mlp_output)

        # residual connection
        if self.apply_residual_connection_post_norm:
            residual = norm_output
        else:
            residual = norm_input
        output = mlp_output[0] + residual

        return output
