# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""Transformer model."""
from dataclasses import dataclass

import math
import copy
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.ops.primitive import constexpr
from mindspore.nn.transformer import VocabEmbedding, TransformerEncoder, TransformerDecoder
from mindspore.nn.transformer.loss import CrossEntropyLoss


@dataclass
class TransformerConfig:
    """Transformer Config"""
    batch_size: int
    seq_length: int = 128
    vocab_size: int = 36560
    hidden_size: int = 1024
    num_hidden_layers: int = 6
    num_heads: int = 16
    intermediate_size: int = 4096
    hidden_act: str = "relu"
    hidden_dropout_prob: float = 0.3
    attention_probs_dropout_prob: float = 0.3
    max_position_embeddings: int = 128
    initializer_range: float = 0.02
    label_smoothing: float = 0.1
    beam_width: int = 4
    max_decode_length: int = 80
    length_penalty_weight: float = 1.0
    dtype: ms.dtype = ms.float32
    compute_dtype: ms.dtype = ms.float32

def position_encoding(length,
                      depth,
                      min_timescale=1,
                      max_timescale=1e4):
    """
    Create Tensor of sinusoids of different frequencies.

    Args:
        length (int): Length of the Tensor to create, i.e. Number of steps.
        depth (int): Hidden size.
        min_timescale (float): Default: 1.
        max_timescale (float): Default: 10000.

    Returns:
        Tensor of shape (length, depth)
    """
    depth = depth // 2
    positions = np.arange(length, dtype=np.float32)
    log_timescale_increment = (np.log(max_timescale / min_timescale) / (depth - 1))
    inv_timescales = min_timescale * np.exp(np.arange(depth, dtype=np.float32) * -log_timescale_increment)
    scaled_time = np.expand_dims(positions, 1) * np.expand_dims(inv_timescales, 0)
    x = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return x


class EmbeddingPostprocessor(nn.Cell):
    """
    Postprocessors apply positional embeddings to word embeddings.

    Args:
        embedding_size (int): The size of each embedding vector.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        max_position_embeddings (int): Maximum length of sequences used in this
                                 model. Default: 128.
        dropout_prob (float): The dropout probability. Default: 0.1.
    """
    def __init__(self,
                 embedding_size,
                 max_position_embeddings=128,
                 dropout_prob=0.1):
        super(EmbeddingPostprocessor, self).__init__()
        self.scores_mul = Tensor([math.sqrt(float(embedding_size))], dtype=ms.float32)
        self.multiply = ops.Mul()
        self.add = ops.Add()
        self.dropout = nn.Dropout(1 - dropout_prob, dtype=ms.float32)
        self.expand_dims = ops.ExpandDims()
        self.position_embedding_table = Tensor(position_encoding(max_position_embeddings, embedding_size),
                                               ms.float32)
        self.shape = ops.Shape()
        self.slice = ops.StridedSlice().shard(((1, 1),))

    def construct(self, word_embeddings):
        """Postprocessors apply positional embeddings to word embeddings."""
        input_shape = self.shape(word_embeddings)
        input_len = input_shape[1]
        hidden_size = input_shape[2]

        output = self.multiply(word_embeddings, self.scores_mul)

        # add position embeddings
        position_embeddings = self.slice(self.position_embedding_table, (0, 0), (input_len, hidden_size), (1, 1))
        position_embeddings = self.expand_dims(position_embeddings, 0)
        output = self.add(output, position_embeddings)

        output = self.dropout(output)
        return output


class CastWrapper(nn.Cell):
    """
    Cast wrapper.
    """
    def __init__(self, dst_type=ms.float32):
        super(CastWrapper, self).__init__()
        self.cast = ops.Cast()
        self.dst_type = dst_type

    def construct(self, x):
        return self.cast(x, self.dst_type)


class CreateAttentionMaskFromInputMask(nn.Cell):
    """
    Create attention mask according to input mask.

    Args:
        config (:class:`TransformerConfig`): Configuration for Transformer.
    """
    def __init__(self, parallel_config):
        super(CreateAttentionMaskFromInputMask, self).__init__()
        self.cast = ops.Cast()
        self.reshape = ops.Reshape()
        self.shape = ops.Shape()
        self.batch_matmul = ops.BatchMatMul().shard(
            ((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))

    def construct(self, input_mask):
        """Create attention mask according to input mask."""
        input_shape = self.shape(input_mask)
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)

        input_mask = self.cast(input_mask, ms.float32)
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = self.batch_matmul(mask_left, mask_right)

        return attention_mask


@constexpr
def convert_np_to_tensor_encoder(seq_length):
    ones = np.ones(shape=(seq_length, seq_length))
    return Tensor(np.tril(ones), dtype=ms.float32)


class T5Head(nn.Cell):
    """
    Head to get the logits of each token in the vocab
    Args:
        config(): the config of network
    Inputs:
        state: the output of the backbone
        embedding_table: the embedding table of the vocabulary
    Returns:
        logits: Tensor, the logits of the corresponding inputs
    """

    def __init__(self,
                 hidden_size,
                 compute_dtype=ms.float16,
                 parallel_config=None):
        super(T5Head, self).__init__()
        if parallel_config.vocab_emb_dp:
            self.matmul = ops.MatMul(transpose_b=True).shard(((parallel_config.data_parallel, 1), (1, 1)))
        else:
            self.matmul = ops.MatMul(transpose_b=True).shard(((parallel_config.data_parallel, 1), (
                parallel_config.model_parallel, 1)))
        self.hidden_size = hidden_size
        self.dtype = compute_dtype
        self.cast = ops.Cast()

    def construct(self, state, embed):
        state = ops.Reshape()(state, (-1, self.hidden_size))
        # output logits over vocabulary [bs*seq_length, vocab_size]
        logits = self.matmul(self.cast(state, self.dtype), self.cast(embed, self.dtype))
        return logits


class TransformerModel(nn.Cell):
    """
    Transformer with encoder and decoder.

    Args:
        config (Class): Configuration for Transformer.
        is_training (bool): True for training mode. False for eval mode.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """
    def __init__(self,
                 config):
        super(TransformerModel, self).__init__()
        config = copy.deepcopy(config)

        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embedding_size = config.hidden_size

        self.last_idx = self.num_hidden_layers - 1
        self.beam_width = config.beam_width
        self.max_decode_length = config.max_decode_length

        self.tfm_embedding_lookup = VocabEmbedding(vocab_size=config.vocab_size,
                                                   embedding_size=self.embedding_size,
                                                   parallel_config=config.parallel_config.embedding_dp_mp_config)
        self.tfm_embedding_postprocessor_for_encoder = EmbeddingPostprocessor(embedding_size=
                                                                              self.embedding_size,
                                                                              max_position_embeddings=
                                                                              config.max_position_embeddings,
                                                                              dropout_prob=
                                                                              config.hidden_dropout_prob)
        self.tfm_embedding_postprocessor_for_decoder = EmbeddingPostprocessor(
            embedding_size=self.embedding_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)
        self.tfm_encoder = TransformerEncoder(
            batch_size=self.batch_size,
            hidden_size=self.hidden_size,
            num_heads=config.num_heads,
            num_layers=self.num_hidden_layers,
            seq_length=config.seq_length,
            ffn_hidden_size=config.intermediate_size,
            attention_dropout_rate=config.attention_probs_dropout_prob,
            hidden_dropout_rate=config.hidden_dropout_prob,
            hidden_act=config.hidden_act,
            moe_config=config.parallel_config.moe_config)

        self.tfm_decoder = TransformerDecoder(
            batch_size=self.batch_size,
            hidden_size=self.hidden_size,
            src_seq_length=config.seq_length,
            tgt_seq_length=config.max_decode_length,
            num_heads=config.num_heads,
            ffn_hidden_size=config.intermediate_size,
            attention_dropout_rate=config.attention_probs_dropout_prob,
            hidden_dropout_rate=config.hidden_dropout_prob,
            num_layers=config.num_hidden_layers,
            hidden_act=config.hidden_act,
            moe_config=config.parallel_config.moe_config)

        self.projection = T5Head(self.hidden_size,
                                 compute_dtype=ms.float16,
                                 parallel_config=config.parallel_config)

        # TODO: Support BeamSearch
        self.cast = ops.Cast()
        self.dtype = config.dtype
        self.cast_compute_type = CastWrapper(dst_type=config.compute_dtype)
        self.expand = ops.ExpandDims()
        self.multiply = ops.Mul()
        self.shape = ops.Shape()

        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask(config.parallel_config)

    def construct(self, source_ids, source_mask, target_ids=None, target_mask=None, memory_mask=None):
        """Transformer with encoder and decoder."""
        seq_length = self.shape(source_ids)[1]

        # process source sentence
        src_word_embeddings, embedding_tables = self.tfm_embedding_lookup(source_ids)
        src_embedding_output = self.tfm_embedding_postprocessor_for_encoder(src_word_embeddings)
        # attention mask [batch_size, seq_length, seq_length]
        if len(ops.shape(source_mask)) == 2:
            enc_attention_mask = self._create_attention_mask_from_input_mask(source_mask)
        else:
            enc_attention_mask = source_mask
        # transformer encoder
        encoder_output, _ = self.tfm_encoder(self.cast_compute_type(src_embedding_output),
                                             self.cast_compute_type(enc_attention_mask))

        # process target sentence
        tgt_word_embeddings, _ = self.tfm_embedding_lookup(target_ids)
        tgt_embedding_output = self.tfm_embedding_postprocessor_for_decoder(tgt_word_embeddings)
        # attention mask [batch_size, seq_length, seq_length]
        if len(ops.shape(target_mask)) == 2:
            future_mask = convert_np_to_tensor_encoder(seq_length)
            tgt_attention_mask = self._create_attention_mask_from_input_mask(target_mask)
            tgt_attention_mask = self.multiply(tgt_attention_mask, self.expand(future_mask, 0))
        else:
            tgt_attention_mask = target_mask

        # transformer decoder
        decoder_output, _ = self.tfm_decoder(self.cast_compute_type(tgt_embedding_output),
                                             self.cast_compute_type(tgt_attention_mask),
                                             encoder_output, memory_mask)
        # calculate logits and log_probs
        log_probs = self.projection(decoder_output, embedding_tables)

        return log_probs


class TransformerNetworkWithLoss(nn.Cell):
    """
    Provide  transformer training loss through network.

    Args:
        config (TransformerConfig): The config of Transformer.
        is_training (bool): Specifies whether to use the training mode.

    Returns:
        Tensor, the loss of the network.
    """
    def __init__(self, network, loss):
        super(TransformerNetworkWithLoss, self).__init__(auto_prefix=False)
        self.transformer = network
        self.loss = loss
        self.cast = ops.Cast()
        self.shape = ops.Shape()

    def construct(self,
                  source_ids,
                  source_mask,
                  target_ids,
                  target_mask,
                  label_ids,
                  label_weights,
                  memory_mask):
        """Transformer network with loss."""
        prediction_scores = self.transformer(source_ids, source_mask, target_ids, target_mask, memory_mask)
        label_ids = ops.Reshape()(label_ids, (-1,))
        label_weights = ops.Reshape()(label_weights, (-1,))
        total_loss = self.loss(prediction_scores, label_ids, self.cast(label_weights, ms.float32))
        return self.cast(total_loss, ms.float32)


def get_t5_network(_, model_config):
    parallel_config = model_config.parallel_config
    network = TransformerModel(config=model_config)
    loss = CrossEntropyLoss(parallel_config=parallel_config.dp_mp_config)
    net_with_loss = TransformerNetworkWithLoss(network=network, loss=loss)
    net_with_loss.set_train(True)
    return net_with_loss
