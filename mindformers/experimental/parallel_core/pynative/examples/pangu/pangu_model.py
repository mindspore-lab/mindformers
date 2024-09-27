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
"""pangu alpha model definition"""

import copy

import mindspore.common.dtype as mstype
from mindspore import mint

from mindformers.experimental.parallel_core.pynative.parallel_state import get_pipeline_model_parallel_world_size, \
    get_pipeline_model_parallel_rank, \
    is_pipeline_first_stage, is_pipeline_last_stage
from mindformers.experimental.parallel_core.pynative.tensor_parallel import (
    VocabParallelEmbedding,
    ColumnParallelLinear,
)
from mindformers.experimental.parallel_core.pynative.tensor_parallel.random import get_rng_tracer
from mindformers.experimental.parallel_core.pynative.transformer import (
    BasePublicLayer,
    BaseHeadLayer,
    ParallelAttention,
    ParallelTransformer,
    ParallelTransformerLayer,
)
from mindformers.experimental.parallel_core.pynative.transformer.module import Module
from mindformers.experimental.parallel_core.pynative.utils import add_attr_for_shared_weight


class PublicLayer(BasePublicLayer):
    """
    PublicLayer for LLM.
    1. Slice input_ids and labels from dataset input.
    2. Generate input_mask from input_ids sliced from dataset input.
    """

    def __init__(self, pad_token):
        super(PublicLayer, self).__init__()
        self.pad_token = pad_token
        self.output_dict = {}

    def construct(self, input_ids, attention_mask):
        """construct function"""
        # Get input and labels
        labels = input_ids[:, 1:]
        input_ids = input_ids[:, :-1]

        attention_mask = attention_mask.astype(mstype.float16)
        input_mask = mint.ne(input_ids, self.pad_token).astype(mstype.float32)

        labels = labels.reshape(-1)
        input_mask = input_mask.reshape(-1)

        self.output_dict["input_ids"] = input_ids
        self.output_dict["label"] = labels
        self.output_dict["input_mask"] = input_mask
        self.output_dict["attention_mask"] = attention_mask

        return self.output_dict


class PanguEmbeddingLayer(Module):
    """
    Embedding layer of the PanGUAlpha Model

    Args:
        config (MindFormerConfig): the config of network

    Inputs:
        **input_ids** (Tensor) - The input token ids, the shape is (batch_size, seq_length).
        **position_ids** (Tensor) - The position ids, the shape is (batch_size, seq_length).

    Outputs:
        **embedding** (Tensor) - The embedding tensor, the shape is (batch_size, seq_length, hidden_size).
        **word_embedding_table** (Tensor) - The word embedding table, the shape is (vocab_size / tp_size, hidden_size).
    """

    def __init__(self, config):
        super(PanguEmbeddingLayer, self).__init__()
        self.word_embedding = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            config=config.parallel_config,
            init_method="normal",
            param_init_dtype=mstype.float32,
            reduce_scatter_embeddings=config.parallel_config.sequence_parallel
        )
        if get_pipeline_model_parallel_world_size() > 1:
            add_attr_for_shared_weight(self.word_embedding, weight_name="weight")
        self.position_embedding = VocabParallelEmbedding(
            num_embeddings=config.seq_length,
            embedding_dim=config.hidden_size,
            config=config.parallel_config,
            init_method="normal",
            param_init_dtype=mstype.float32,
            reduce_scatter_embeddings=config.parallel_config.sequence_parallel
        )
        self.dropout = mint.nn.Dropout(p=config.hidden_dropout_rate)

    def construct(self, input_ids, position_ids):
        """construct method"""
        word_embedding = self.word_embedding(input_ids)
        word_embedding_table = self.word_embedding.weight
        position_embedding = self.position_embedding(position_ids)
        embedding = word_embedding + position_embedding
        with get_rng_tracer().rng_fork():
            embedding = self.dropout(embedding)
        return embedding, word_embedding_table

    def get_word_embedding_weight(self):
        """
        Get the weight of the word embedding layer.

        Returns:
            weight (Tensor): Word embedding weight tensor of shape (vocab_size / tp_size, hidden_size).
        """
        return self.word_embedding.weight


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
        if config.lora_config.use_lora:
            attention_config.update_lora_config(cell_name='attention')
        self.attention = ParallelAttention(
            layer_number=1, config=attention_config, attention_type="cross_attn"
        )
        self.query_embedding = VocabParallelEmbedding(
            num_embeddings=config.seq_length,
            embedding_dim=config.hidden_size,
            config=config.parallel_config,
            init_method="normal",
            param_init_dtype=mstype.float32,
            reduce_scatter_embeddings=config.parallel_config.sequence_parallel
        )

    def construct(self, hidden_states, position_ids, attention_mask):
        """construct method"""
        # hidden_states: [B, S, H]

        # query vector
        query_vector = self.query_embedding(position_ids)

        # normalization
        norm_output = self.input_norm(hidden_states)

        # attention
        # NOTICE: rotary_pos_emb is not used in Pangu Alpha
        attention_output, _ = self.attention(
            query_vector, attention_mask, norm_output, rotary_pos_emb=None
        )
        with get_rng_tracer().rng_fork():
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
        with get_rng_tracer().rng_fork():
            mlp_output = self.hidden_states_dropout(mlp_output)

        # residual connection
        if self.apply_residual_connection_post_norm:
            residual = norm_output
        else:
            residual = norm_input
        output = mlp_output[0] + residual

        return output


class PanguHead(BaseHeadLayer):
    """
    Head to get the logits of each token in the vocab
    Args:
        config (PanguConfig): the config of network
    Inputs:
        **hidden_states** (Tensor) - The input tensor, the shape is (batch_size, seq_length, hidden_size).
        **word_embedding_table** (Tensor) - The word embedding table, the shape is (vocab_size / tp_size, hidden_size).
    """

    def __init__(self, config):
        super(PanguHead, self).__init__()
        self.skip_weight_param_allocation = get_pipeline_model_parallel_world_size() == 1 \
                                            and config.head_skip_weight_param_allocation
        self.matmul = ColumnParallelLinear(
            config=config,
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            skip_weight_param_allocation=self.skip_weight_param_allocation,
            bias=False,
            gather_output=True,
            compute_dtype=config.compute_dtype,
            init_method=config.init_method,
            bias_init=config.bias_init
        )
        if get_pipeline_model_parallel_world_size() > 1:
            add_attr_for_shared_weight(self.matmul)

    def construct(self, hidden_states, word_embedding_table=None):
        """construct method"""
        # [B, S, H] * [H, V / tp_size]^T -> [B, S, V]
        if self.skip_weight_param_allocation:
            if word_embedding_table is None:
                raise ValueError("word_embedding_table should not be None when skip_weight_param_allocation is True.")
            logits, _ = self.matmul(hidden_states, word_embedding_table)
        else:
            logits, _ = self.matmul(hidden_states)

        # flattern logits
        logits = logits.reshape(-1, logits.shape[-1])
        return logits


class PanguAlphaBackbone(Module):
    """
    The base backbone of the PanGuAlpha model.
    It consists of the embedding layer, transformer layers, and the query layer.

    Args:
        config (MindFormerConfig): the config of network
    Inputs:
        **input_ids** (Tensor) - The input token ids, the shape is (batch_size, seq_length).
        **position_ids** (Tensor) - The position ids, the shape is (batch_size, seq_length).
        **attention_mask** (Tensor) - The attention mask, the shape is (batch_size, seq_length, seq_length).
    Outputs:
        **hidden_states** (Tensor) - The output tensor, the shape is (batch_size, seq_length, hidden_size).
        **word_embedding_table** (Tensor) - The word embedding table, the shape is (vocab_size / tp_size, hidden_size).
    """

    def __init__(self, config):
        super(PanguAlphaBackbone, self).__init__()
        self.config = config
        self.embedding = PanguEmbeddingLayer(config)

        # one layer is Query Layer
        config.num_layers -= 1
        if config.num_layers < 0:
            raise ValueError("The number of layers should be greater than 0.")
        self.num_layers = config.num_layers

        # NOTICE: will do norm at the end
        layers_config = copy.deepcopy(config)
        query_layer_config = copy.deepcopy(config)
        if config.lora_config.use_lora:
            layers_config.update_lora_config(cell_name='layers')
            query_layer_config.update_lora_config(cell_name='query_layer')
        self.layers = ParallelTransformer(config=layers_config,
                                          post_norm=True,
                                          pre_process=is_pipeline_first_stage(),
                                          post_process=is_pipeline_last_stage())
        self.query_layer = PanguQueryLayer(layer_number=1, config=query_layer_config)

    def construct(self, input_ids, position_ids, attention_mask):
        r"""forward pass of the model"""
        # transformer blocks
        # hideen_states: [B, S, H], word_embedding_table: [V / tp_size, H]
        hidden_states, word_embedding_table = self.embedding(input_ids, position_ids)
        hidden_states = self.layers(hidden_states, attention_mask)

        # query layer
        # [B, S, H] -> [B, S, H]
        hidden_states = self.query_layer(
            hidden_states, position_ids, attention_mask
        )

        return hidden_states, word_embedding_table


class PanguAlphaWithHead(Module):
    """
    The whole PanguAlpha network, which is consisting of two parts: the backbone and the head
    Args:
        config (PanguAlphaConfig): the config of network
    Inputs:
        **input_ids** (Tensor) - The input token ids, the shape is (batch_size, seq_length).
        **position_ids** (Tensor) - The position ids, the shape is (batch_size, seq_length).
        **attention_mask** (Tensor) - The attention mask, the shape is (batch_size, seq_length, seq_length).
    Outputs:
        **logits** (Tensor) - The output tensor, the shape is (batch_size, seq_length, vocab_size).
    """

    def __init__(self, config):
        super(PanguAlphaWithHead, self).__init__()

        # Network head to get logits over vocabulary
        backbone_config = copy.deepcopy(config)
        if config.lora_config.use_lora:
            backbone_config.update_lora_config('backbone')
        self.backbone = PanguAlphaBackbone(backbone_config)
        self.head = PanguHead(config)

    def construct(self, input_ids, position_ids, attention_mask):
        """construct method"""
        # backbone
        # [B, S, H] -> [B, S, H]
        hidden_states, word_embedding_table = self.backbone(
            input_ids, position_ids, attention_mask
        )

        # head
        # [B, S, H] -> [B, S, V]
        logits = self.head(hidden_states, word_embedding_table)
        return logits


class ModelWithLossCell(Module):
    """
    PanguAlpha training wrapper with loss function
    Args:
        pad_token (int): the token id for padding
        network (PanguAlphaWithHead): the network
        loss_fn (callable): the loss function
    Inputs:
        **input_ids** (Tensor) - The input token ids, the shape is (batch_size, seq_length).
        **position_ids** (Tensor) - The position ids, the shape is (batch_size, seq_length).
        **attention_mask** (Tensor) - The attention mask, the shape is (batch_size, seq_length, seq_length).
    Outputs:
        **output** (Tensor) - The loss tensor.
        **logits** (Tensor) - The output tensor, the shape is (batch_size, seq_length, vocab_size).
    """

    def __init__(self, network, loss_fn, pad_token):
        super(ModelWithLossCell, self).__init__(auto_prefix=False)
        self.network = network
        self.pad_token = pad_token
        self.loss_fn = loss_fn
        self.public_layer = PublicLayer(pad_token)

    def construct(self, input_ids, position_ids=None, attention_mask=None):
        """construct method"""
        # Get input and labels
        processed_inputs = self.public_layer(input_ids, attention_mask)
        input_ids = processed_inputs["input_ids"]
        labels = processed_inputs["label"]
        input_mask = processed_inputs["input_mask"]
        attention_mask = processed_inputs["attention_mask"]

        # feed into network
        # [B, S, H]
        logits = self.network(input_ids, position_ids, attention_mask)

        # calculate loss
        # logits, labels, and input_mask has been flatterned
        loss = self.loss_fn(logits, labels, input_mask)
        return loss, logits

    def build_pipeline_model(self):
        """build pipeline model"""
        pp_rank = get_pipeline_model_parallel_rank()
        last_pp_stage_rank = get_pipeline_model_parallel_world_size() - 1
        num_layers = self.network.backbone.num_layers

        # get the layers at current stage
        per_stage_layers_num = num_layers // get_pipeline_model_parallel_world_size()
        remainder = num_layers % get_pipeline_model_parallel_world_size()
        start_idx = per_stage_layers_num * pp_rank
        # add remainder layers to the first few stages except the first stage
        if remainder and 0 < pp_rank <= remainder:
            per_stage_layers_num += 1
            start_idx += max(pp_rank - 1, 0)
        else:
            start_idx += remainder
        end_idx = min(per_stage_layers_num * (pp_rank + 1), num_layers)

        cur_stage_transformer_layers = self.slice_transformer_layers(
            self.network.backbone.layers.layers, start_idx=start_idx, end_idx=end_idx
        )
        if pp_rank == 0:
            self.pp_submodel.add([
                self.network.backbone.embedding,
                *[layer for layer in cur_stage_transformer_layers]
            ])
        elif pp_rank == last_pp_stage_rank:
            self.pp_submodel.add([
                *[layer for layer in cur_stage_transformer_layers],
                self.network.backbone.layers.final_norm,
                self.network.backbone.query_layer,
                self.network.head,
                self.loss_fn
            ])
        else:
            self.pp_submodel.add([
                *[layer for layer in cur_stage_transformer_layers]
            ])
