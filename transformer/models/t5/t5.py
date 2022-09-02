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
import mindspore as mstype
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.ops.primitive import constexpr
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from mindspore.nn.transformer.loss import CrossEntropyLoss
from mindspore.nn.transformer import VocabEmbedding

from transformer.models.t5.T5Transformer import TransformerEncoder, TransformerDecoder, LayerNorm


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
    dtype: mstype.dtype = mstype.float32
    compute_dtype: mstype.dtype = mstype.float32
    has_relative_bias: bool = True
    scale_output: bool = True


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
        self.scores_mul = Tensor([math.sqrt(float(embedding_size))], dtype=mstype.float32)
        self.multiply = ops.Mul()
        self.add = ops.Add()
        self.dropout = nn.Dropout(1 - dropout_prob, dtype=mstype.float32)
        self.expand_dims = ops.ExpandDims()
        self.position_embedding_table = Tensor(position_encoding(max_position_embeddings, embedding_size),
                                               mstype.float32)
        self.shape = ops.Shape()
        self.slice = ops.StridedSlice().shard(((1, 1),))

    def construct(self, word_embeddings):
        """Postprocessors apply positional embeddings to word embeddings."""
        output = self.multiply(word_embeddings, self.scores_mul)

        output = self.dropout(output)
        return output


class CastWrapper(nn.Cell):
    """
    Cast wrapper.
    """
    def __init__(self, dst_type=mstype.float32):
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

        input_mask = self.cast(input_mask, mstype.float32)
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = self.batch_matmul(mask_left, mask_right)

        return attention_mask


@constexpr
def convert_np_to_tensor_encoder(seq_length):
    ones = np.ones(shape=(seq_length, seq_length))
    return Tensor(np.tril(ones), dtype=mstype.float32)


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
                 compute_dtype=mstype.float16,
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
        self.seq_length = config.seq_length

        self.last_idx = self.num_hidden_layers - 1
        self.beam_width = config.beam_width
        self.max_decode_length = config.max_decode_length
        self.scale_output = config.scale_output

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
                                 compute_dtype=mstype.float16,
                                 parallel_config=config.parallel_config)


        self.cast = ops.Cast()
        self.dtype = config.dtype
        self.cast_compute_type = CastWrapper(dst_type=config.compute_dtype)
        self.expand = ops.ExpandDims()
        self.multiply = ops.Mul()
        self.shape = ops.Shape()
        self.encoder_layernorm = LayerNorm(normalized_shape=(self.embedding_size,)).to_float(mstype.float32)
        self.decoder_layernorm = LayerNorm(normalized_shape=(self.embedding_size,)).to_float(mstype.float32)
        self.encoder_layernorm.shard(((config.parallel_config.data_parallel, 1),))
        self.decoder_layernorm.shard(((config.parallel_config.data_parallel, 1),))
        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask(config.parallel_config)

    def construct(self, source_ids=None, source_mask=None, target_ids=None, target_mask=None, memory_mask=None,
                  encoder_cache=None):
        """Transformer with encoder and decoder."""
        if source_ids is not None:
            encoder_output = self.encoder_forward(source_ids, source_mask)
        else:
            encoder_output = encoder_cache

        if target_ids is None:
            return encoder_output

        # process target sentence
        tgt_embedding_output, embedding_table = self.tfm_embedding_lookup(target_ids)
        # attention mask [batch_size, seq_length, seq_length]
        tgt_length = self.shape(target_ids)[1]

        if memory_mask is None:
            memory_mask = self.create_memory_mask(source_mask, target_mask)

        if len(ops.shape(target_mask)) == 2:
            future_mask = convert_np_to_tensor_encoder(tgt_length)
            tgt_attention_mask = self._create_attention_mask_from_input_mask(target_mask)
            tgt_attention_mask = self.multiply(tgt_attention_mask, self.expand(future_mask, 0))
        else:
            tgt_attention_mask = target_mask

        # transformer decoder
        decoder_output, _ = self.tfm_decoder(self.cast_compute_type(tgt_embedding_output),
                                             self.cast_compute_type(tgt_attention_mask),
                                             encoder_output, memory_mask)
        decoder_output = self.decoder_layernorm(decoder_output)

        if self.scale_output:
            decoder_output = decoder_output * (self.hidden_size ** -0.5)
        # calculate logits and log_probs
        log_probs = self.projection(decoder_output, embedding_table)

        return log_probs

    def encoder_forward(self, source_ids, source_mask):
        """Execute the forward process"""
        # process source sentence
        src_embedding_output, _ = self.tfm_embedding_lookup(source_ids)
        # attention mask [batch_size, seq_length, seq_length]
        if len(F.shape(source_mask)) == 2:
            enc_attention_mask = self._create_attention_mask_from_input_mask(source_mask)
        else:
            enc_attention_mask = source_mask
        # transformer encoder
        encoder_output, _ = self.tfm_encoder(self.cast_compute_type(src_embedding_output),
                                             self.cast_compute_type(enc_attention_mask))
        encoder_output = self.encoder_layernorm(encoder_output)

        return encoder_output

    def create_memory_mask(self, source_mask, target_mask):
        memory_mask = P.Ones()((self.batch_size, self.max_decode_length, self.seq_length), mstype.float32)
        memory_mask = memory_mask * F.expand_dims(source_mask, 1)
        memory_mask = memory_mask * F.expand_dims(target_mask, 2)
        return memory_mask


class TransformerNetworkWithLoss(nn.Cell):
    """
    Provide  transformer training loss through network.

    Args:
        network (nn.Cell): The network of the transformer.
        loss (nn.Cell): Loss cell for.

    Returns:
        Tensor, the loss of the network.
    """
    def __init__(self, network, loss):
        super(TransformerNetworkWithLoss, self).__init__(auto_prefix=False)
        self.transformer = network
        self.loss = loss
        self.cast = ops.Cast()
        self.shape = ops.Shape()

        self.start_token = Tensor(np.zeros((4, 1)).astype(np.int32))
        self.concat = P.Concat(axis=1)

    def construct(self,
                  source_ids,
                  source_mask,
                  target_ids,
                  target_mask=None,
                  memory_mask=None):
        """Transformer network with loss."""
        labels = target_ids[:, :-1]
        if target_mask is None:
            target_mask = F.cast(labels != 0, mstype.float32)

        decoder_inputs = self.concat((self.start_token, labels[:, :-1]))

        logits = self.transformer(source_ids, source_mask, decoder_inputs, target_mask, memory_mask)

        label_ids = ops.Reshape()(labels, (-1,))
        label_weights = ops.Reshape()(target_mask, (-1,))
        total_loss = self.loss(logits, label_ids, self.cast(label_weights, mstype.float32))
        return total_loss


class EvalNet(nn.Cell):
    """
    T5 evaluation net

    Args:
        backbone(nn.Cell): backbone network of GPT2/3
        generate(bool): enable generate mode

    Returns:
        outputs: Tensor, corresponding output for different tasks
    """
    def __init__(self, backbone, generate=False):
        super(EvalNet, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.argmax = P.Argmax()
        self.generate = generate
        self.cast = P.Cast()
        self.gather = P.Gather()
        self.pad_token = 0

    def construct(self, input_ids, input_mask, current_index=None,
                  cache_encoder=None, target_id=None, target_mask=None):
        """evaluation net"""
        if cache_encoder is None:
            input_mask = self.cast(input_mask, mstype.float32)
            outputs = self.backbone(input_ids, input_mask)
        else:
            input_mask = self.cast(input_mask, mstype.float32)
            if target_mask is None:
                target_mask = F.cast(target_id != self.pad_token, mstype.float32)
            logits = self.backbone(None, input_mask, target_id, target_mask, None, cache_encoder)
            outputs = None
            if self.generate:
                index = current_index.view(1,)
                logits = self.gather(logits, index, 0)
                outputs = nn.LogSoftmax()(logits)
                outputs = F.tensor_pow(np.e, outputs)
            else:
                outputs = self.argmax(logits)
        return outputs


def get_t5_network(opt, model_config):
    """Get the t5 network"""
    parallel_config = model_config.parallel_config
    network = TransformerModel(config=model_config)
    if opt.eval:
        opt.logger.info("Detect the eval is True, return the eval net.")
        net = EvalNet(network, generate=opt.generate)
        return net
    loss = CrossEntropyLoss(parallel_config=parallel_config.dp_mp_config)
    net_with_loss = TransformerNetworkWithLoss(network=network, loss=loss)
    net_with_loss.set_train(True)

    # disable the bias
    for param in net_with_loss.trainable_params():
        if ('bias' in param.name or 'beta' in param.name) and 'relative' not in param.name:
            param.requires_grad = False
        opt.logger.info(f"Param name {param.name} is disabled gradients.")
    return net_with_loss
