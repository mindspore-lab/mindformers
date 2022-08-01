# Copyright 2020 Huawei Technologies Co., Ltd
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

"""GPT model"""
import copy
from dataclasses import dataclass

import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.transformer import TransformerOpParallelConfig
from mindspore.nn.transformer.transformer import default_transformer_config
from mindspore.nn.transformer.layers import _LayerNorm
from mindspore.nn.transformer.transformer import AttentionMask, Transformer, VocabEmbedding
from mindspore.nn.transformer.loss import CrossEntropyLoss

from transformer.utils import _convert_dtype_class

@dataclass
class GPTConfig:
    """
    GPT config class which defines the model size
    """
    batch_size: int = 32
    seq_length: int = 1024
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    expand_ratio: int = 4
    post_layernorm_residual: bool = False
    dropout_rate: float = 0.1
    compute_dtype: mstype = mstype.float16
    layernorm_dtype: mstype = mstype.float32
    softmax_dtype: mstype = mstype.float16
    parallel_config: TransformerOpParallelConfig = default_transformer_config

class GPTModel(nn.Cell):
    """
    The backbone of GPT network

    Args:
        config(GPTConfig): the config of network

    Inputs:
        input_ids: the tokenized inputs with datatype int32
        input_mask: the mask indicating whether each position is a valid input
        layer_past: the previous feature map

    Returns:
        output_state: Tensor, the output logit of backbone
        present_layer: Tensor, the current feature map
        embedding_table: Tensor, the embedding table for the vocabulary
    """
    def __init__(self, config):
        super(GPTModel, self).__init__()
        self.get_attention_mask = AttentionMask(seq_length=config.seq_length,
                                                parallel_config=config.parallel_config.dp_mp_config)
        self.word_embedding = VocabEmbedding(vocab_size=config.vocab_size,
                                             embedding_size=config.hidden_size,
                                             param_init=initializer("truncatedNormal",
                                                                    [config.vocab_size, config.hidden_size],
                                                                    dtype=config.compute_dtype),
                                             parallel_config=config.parallel_config.embedding_dp_mp_config)

        new_parallel_config = copy.deepcopy(config.parallel_config.embedding_dp_mp_config)
        new_parallel_config.vocab_emb_dp = True

        self.position_embedding = VocabEmbedding(vocab_size=config.seq_length,
                                                 embedding_size=config.hidden_size,
                                                 param_init=initializer(TruncatedNormal(0.02),
                                                                        [config.seq_length, config.hidden_size],
                                                                        dtype=config.compute_dtype),
                                                 parallel_config=new_parallel_config)
        self.blocks = nn.CellList()
        moe_config = config.parallel_config.moe_config
        self.transformer = Transformer(hidden_size=config.hidden_size,
                                       batch_size=config.batch_size,
                                       ffn_hidden_size=config.hidden_size * 4,
                                       src_seq_length=config.seq_length,
                                       tgt_seq_length=config.seq_length,
                                       encoder_layers=config.num_layers,
                                       attention_dropout_rate=config.dropout_rate,
                                       hidden_dropout_rate=config.dropout_rate,
                                       decoder_layers=0,
                                       param_init_type=config.compute_dtype,
                                       layernorm_compute_type=config.layernorm_dtype,
                                       softmax_compute_type=config.softmax_dtype,
                                       num_heads=config.num_heads,
                                       parallel_config=config.parallel_config,
                                       moe_config=moe_config)
        self.use_moe = (moe_config.expert_num > 1)
        self.layernorm = _LayerNorm((config.hidden_size,)).to_float(config.layernorm_dtype)
        self.layernorm.shard(((config.parallel_config.data_parallel, 1, 1),))
        self.add = P.Add().shard(
            ((config.parallel_config.data_parallel, 1, 1), (config.parallel_config.data_parallel, 1, 1)))

    def construct(self, input_ids, input_mask):
        """GPT model"""
        input_embedding, embedding_table = self.word_embedding(input_ids)

        batch_size, seq_length = F.shape(input_ids)
        input_position = F.tuple_to_array(F.make_range(seq_length))
        input_position = P.Tile()(input_position, (batch_size, 1))

        position_embedding, _ = self.position_embedding(input_position)
        hidden_states = self.add(input_embedding, position_embedding)

        hidden_states = P.Cast()(hidden_states, mstype.float16)
        attention_mask = self.get_attention_mask(input_mask)
        moe_loss = 0
        if self.use_moe:
            hidden_states, present_layer, _, moe_loss = self.transformer(hidden_states, attention_mask)
        else:
            hidden_states, present_layer, _ = self.transformer(hidden_states, attention_mask)
        output_state = self.layernorm(hidden_states)
        if self.use_moe:
            return output_state, present_layer, embedding_table, moe_loss
        return output_state, present_layer, embedding_table

class GPTHead(nn.Cell):
    """
    Head for GPT to get the logits of each token in the vocab

    Args:
        config(GPTConfig): the config of network

    Inputs:
        state: the output of the backbone
        embedding_table: the embedding table of the vocabulary

    Returns:
        logits: Tensor, the logits of the corresponding inputs
    """
    def __init__(self,
                 hidden_size,
                 compute_type=mstype.float16,
                 parallel_config=None):
        super(GPTHead, self).__init__()
        if parallel_config.vocab_emb_dp:
            self.matmul = P.MatMul(transpose_b=True).shard(((parallel_config.data_parallel, 1), (1, 1)))
        else:
            self.matmul = P.MatMul(transpose_b=True).shard(((parallel_config.data_parallel, 1), (
                parallel_config.model_parallel, 1)))
        self.hidden_size = hidden_size
        self.dtype = compute_type
        self.cast = P.Cast()

    def construct(self, state, embedding_table):
        state = P.Reshape()(state, (-1, self.hidden_size))
        logits = self.matmul(self.cast(state, self.dtype), self.cast(embedding_table, self.dtype))
        return logits


class GPT(nn.Cell):
    """
    The GPT network consisting of two parts the backbone and the head

    Args:
        config(GPTConfig): the config of network

    Inputs:
        input_ids: the tokenized inputs
        input_mask: the mask indicating whether each position is a valid input
        past: the previous feature map

    Returns:
        logits: Tensor: the logits of the corresponding inputs with shape (batch_size, seq_length, vocab_size)
    """
    def __init__(self, config):
        super(GPT, self).__init__()
        self.backbone = GPTModel(config)
        self.head = GPTHead(config.hidden_size, parallel_config=config.parallel_config)
        self.use_moe = self.backbone.use_moe

    def construct(self, input_ids, input_mask):
        if self.use_moe:
            output_states, _, embedding_table, moe_loss = self.backbone(input_ids, input_mask)
            logits = self.head(output_states, embedding_table)
            return logits, moe_loss
        output_states, _, embedding_table = self.backbone(input_ids, input_mask)
        logits = self.head(output_states, embedding_table)
        return logits

class GPTWithLoss(nn.Cell):
    """
    GPT training loss

    Args:
        network: backbone network of GPT2/3
        loss: loss function, e.g., crossentropy
        eos_token: the end_of_sentence token

    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map

    Returns:
        output: Tensor, the loss of the network
    """
    def __init__(self, network, loss, parallel_config, eos_token=50256):
        super(GPTWithLoss, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = loss
        self.eos_token = eos_token
        self.shape = P.Shape()
        dp = parallel_config.data_parallel
        self.stridedslice = P.StridedSlice().shard(((dp, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.use_moe = self.network.use_moe
        self.add = P.Add().shard(((1,), ()))
    def construct(self, input_ids):
        """GPT model with loss"""
        moe_loss = 0
        batch_size, seq_length = self.shape(input_ids)
        tokens = self.stridedslice(input_ids, (0, 0), (batch_size, seq_length - 1), (1, 1))
        input_mask = F.cast(self.not_equal(tokens, self.eos_token), mstype.float32)
        if self.use_moe:
            logits, moe_loss = self.network(tokens, input_mask)
        else:
            logits = self.network(tokens, input_mask)
        labels = self.stridedslice(input_ids, (0, 1), (batch_size, seq_length), (1, 1))
        labels = P.Reshape()(labels, (-1,))
        input_mask = P.Reshape()(input_mask, (-1,))
        output = self.loss(logits, labels, input_mask)
        if self.use_moe:
            return self.add(output, moe_loss)
        return output

class EvalNet(nn.Cell):
    """
    GPT evaluation net

    Args:
        backbone: backbone network of GPT2/3
        generate: enable generate mode

    Inputs:
        input_ids: the tokenized inpus

    Returns:
        outputs: Tensor, corresponding output for different tasks
    """
    def __init__(self, backbone, generate=False):
        super(EvalNet, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.argmax = P.Argmax()
        self.generate = generate
        self.cast = P.Cast()

    def construct(self, input_ids, input_mask):
        """evaluation net"""
        input_mask = self.cast(input_mask, mstype.float32)
        logits = self.backbone(input_ids, input_mask)
        outputs = None
        if self.generate:
            outputs = nn.LogSoftmax()(logits)
            outputs = F.tensor_pow(np.e, outputs)
        else:
            outputs = self.argmax(logits)
        return outputs


def get_gpt_network(_, model_config):
    loss = CrossEntropyLoss(model_config.parallel_config.dp_mp_config)
    # maps fp16 to mstype.float16 and fp32 to mstype.float32
    for k, v in model_config.__dict__.items():
        model_config.__dict__[k] = _convert_dtype_class(v)
    net = GPT(model_config)
    net_with_loss = GPTWithLoss(net, loss, model_config.parallel_config)
    return net_with_loss
