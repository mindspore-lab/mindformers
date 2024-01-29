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
"""iFlytekSpark model APIs."""
import copy
import os
import numpy as np

import mindspore as ms
import mindspore.ops as P
from mindspore import dtype as mstype
from mindspore import nn
from mindspore.communication.management import get_rank
from mindspore.ops import functional as F

from mindformers.models import BaseModel
from mindformers.core.loss import CrossEntropyLoss
from mindformers.modules import KVCachePreprocess
from mindformers.modules.transformer.transformer import AttentionMask
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister

from iflytekspark_config import IFlytekSparkConfig
from iflytekspark_layers import IFlytekSparkEmbedding, IFlytekSparkTransformer
from iflytekspark_text_generator import IFlytekSparkGeneratorMixin



class IFlytekSparkModel(BaseModel):
    r"""
    Transformer backbone of iFlytekSpark model.
    Args:
        config(IFlytekSparkConfig): the config of network

    Returns:
            output: Tensor, the output of llama decoderlayer
    """
    def __init__(self, config: IFlytekSparkConfig = None):
        super(IFlytekSparkModel, self).__init__(config, auto_prefix=True)

        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size
        self.num_layers = config.num_layers
        self.init_method = "xavier_normal"
        self.use_past = config.use_past
        self.is_reward_model = config.is_reward_model
        self.is_dynamic = config.is_dynamic
        self.is_lite_infer = config.is_lite_infer

        self.not_equal = P.NotEqual().shard(((config.parallel_config.data_parallel, 1), ()))
        self.expand_dim = P.ExpandDims().shard(((1, 1), ()))
        self.get_attention_mask_train = IFlytekSparkAttentionMask(
            seq_length=config.seq_length,
            parallel_config=config.parallel_config.dp_mp_config,
            use_past=config.use_past,
            is_dynamic=self.is_dynamic).to_float(mstype.float16)
        self.get_attention_mask_prefill = self.get_attention_mask_train
        infer_len = config.seq_length if config.seq_length <= config.sparse_local_size else config.sparse_local_size
        self.get_attention_mask_decode = IFlytekSparkAttentionMask(seq_length=infer_len,
                                                                   parallel_config=config.parallel_config.dp_mp_config,
                                                                   use_past=config.use_past,
                                                                   is_dynamic=self.is_dynamic).to_float(mstype.float16)
        self.mul = P.Mul().shard(((config.parallel_config.data_parallel, 1, 1), (1, 1, 1)))
        self.sub = P.Sub().shard(((), (config.parallel_config.data_parallel, 1, 1)))
        self.dummy_input = ms.Tensor(0, mstype.int32)

        if self.is_lite_infer:
            self.kvcache_preprocess = IFlytekSparkKVCachePreprocess(max_batch_size=config.batch_size,
                                                                    max_seq_length=infer_len,
                                                                    is_dynamic=self.is_dynamic,
                                                                    use_kvcache_op=True,
                                                                    is_flexible_shape=False)

        # Embeddings
        self.embedding = IFlytekSparkEmbedding(
            self.embed_dim,
            config.vocab_size,
            None,
            config.dropout_rate,
            config.embedding_init_type,
            self.init_method,
            None,
            parallel_config=config.parallel_config
        )

        # Configure the shard configure of the Embedding layer
        self.embedding.pipeline_stage = 0

        # Transformer
        self.rank_size = int(os.getenv("RANK_SIZE", '1'))
        if self.rank_size > 1:
            rank_id = get_rank()
        else:
            rank_id = 0
        self.transformer = IFlytekSparkTransformer(
            config,
            rank_id=rank_id,
        )

    # pylint: disable=W0221
    def construct(self, input_ids, position_ids, attention_mask=None,
                  init_reset=True, batch_valid_length=None, zactivate_len=None):
        """Forward of iflytekspark model"""
        embeddings, word_table = self.embedding(input_ids, position_ids)

        # make sure this number is larger than vocab size
        if attention_mask is None:
            input_mask = self.cast(self.not_equal(input_ids, self.vocab_size + 1), ms.float16)
            attention_mask = self.get_attention_mask_train(input_mask)
        else:
            if self.use_past and self.is_first_iteration:
                attention_mask = self.get_attention_mask_prefill(attention_mask)
            else:
                attention_mask = self.get_attention_mask_decode(attention_mask)
        attention_mask = self.sub(1.0, attention_mask)

        zactivate_len = batch_index_pad = seq_length_tensor_pad = self.dummy_input # To avoid PFA bug
        if self.is_lite_infer:
            zactivate_len, batch_index_pad, seq_length_tensor_pad \
                                                = self.kvcache_preprocess(zactivate_len=zactivate_len)

        # Transformer.
        output = self.transformer(
            hidden_states=embeddings,
            attention_mask=attention_mask,
            init_reset=init_reset,
            batch_valid_length=batch_valid_length,
            zactivate_len=zactivate_len,
            batch_index_pad=batch_index_pad,
            seq_length_tensor_pad=seq_length_tensor_pad
        )

        return output, word_table


class IFlytekSparkKVCachePreprocess(KVCachePreprocess):
    """ iFlytekSpark model KVCache Manager. """

    def __init__(self,
                 max_batch_size=8,
                 max_seq_length=4096,
                 is_dynamic=False,
                 use_kvcache_op=False,
                 is_flexible_shape=False,
                 ):
        super(IFlytekSparkKVCachePreprocess, self).__init__(max_batch_size, max_seq_length, is_dynamic,
                                                            use_kvcache_op, is_flexible_shape)
        self.max_batch_size = max_batch_size

    def construct(self, batch_index=None, zactivate_len=None):
        """precompute kvcache inputs"""
        if batch_index is None:
            batch_index = P.arange(0, self.max_batch_size, 1)
        batch_index_pad = self.concat((batch_index, self.cache_pad_tensor))
        seq_length_tensor_pad = self.get_seq_length_tensor_pad(batch_size=self.max_batch_size)
        return zactivate_len, batch_index_pad, seq_length_tensor_pad


class IFlytekSparkAttentionMask(AttentionMask):
    r"""
        Get the Lower triangular matrix or sparse matrix from the input mask.
        The input mask is a 2D tensor (batch_size, seq_length) with 1 and 0,
        where 1 indicates the current position is a valid token, otherwise not.

        Args:
            seq_length(int): The sequence length of the input tensor.
            parallel_config(OpParallelConfig): The parallel configure. Default `default_dpmp_config`,
                                               an instance of `OpParallelConfig` with default args.

        Inputs:
            - **input_mask** (Tensor) - The mask indicating whether each position is a valid input with
              (batch_size, seq_length).

        Outputs:
            Tensor. The attention mask matrix with shape (batch_size, seq_length, seq_length).
    """

    def __init__(self, seq_length, parallel_config, use_past=False, is_dynamic=False, sparse_local_size=None):
        super(IFlytekSparkAttentionMask, self).__init__(seq_length, parallel_config)
        self.use_past = use_past
        self.is_dynamic = is_dynamic
        slice_sparse_mask = self.lower_triangle_mask.astype(mstype.float16)

        self.slice = P.StridedSlice().shard(((1, 1),))
        if sparse_local_size is not None:
            sparse_mask = ~self.get_slice_sparse_block(seq_length, sparse_local_size)
            ones = np.ones(shape=(seq_length, seq_length))
            lower_triangle_mask = np.tril(ones)
            slice_sparse_mask = sparse_mask * lower_triangle_mask
            slice_sparse_mask = ms.Tensor(slice_sparse_mask, mstype.float16)
        self.slice_sparse_mask = ms.Parameter(slice_sparse_mask, name="attention_mask", requires_grad=False)
        del self.lower_triangle_mask

    def construct(self, input_mask):
        """Forward process of the IFlytekSparkAttentionMask"""
        input_mask = P.Cast()(self.not_equal(input_mask, 0), mstype.float16)
        input_shape = P.Shape()(input_mask)
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)
        # Mask the padded inputs
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = self.mul(mask_left, mask_right)

        # dynamic mode only support ms lite
        if self.use_past and self.is_first_iteration and self.is_dynamic:
            slice_sparse_mask = self.slice(self.slice_sparse_mask,
                                           (0, 0), (input_shape[1], input_shape[1]), (1, 1))
        else:
            slice_sparse_mask = self.slice_sparse_mask

        slice_sparse_mask = self.expand_dim(slice_sparse_mask, 0)
        # the returned shape is [bs, seq_length, seq_length]
        attention_mask = self.multiply(
            attention_mask, slice_sparse_mask)
        return attention_mask

    def get_slice_sparse_block(self, seqlen, local_size):
        srange = np.arange(seqlen)
        mask1 = srange[:, None] < srange[None, :]
        mask2 = srange[:, None] >= srange[None, :] + local_size
        mask = mask1 | mask2
        return mask


class LmLogits(nn.Cell):
    """
    Head to get the logits of each token in the vocab
    Args:
        hidden_size:
        compute_type: compute type
        parallel_config: the config of parallel
    Inputs:
        state: the output of the backbone
        embed: the embedding table of the vocabulary
    Returns:
        logits: Tensor, the logits of the corresponding inputs
    """

    def __init__(self,
                 hidden_size,
                 compute_type=mstype.float16,
                 parallel_config=None):
        super(LmLogits, self).__init__()
        if parallel_config.vocab_emb_dp:
            self.matmul = P.MatMul(transpose_b=True).shard(((parallel_config.data_parallel, 1), (1, 1)))
        else:
            self.matmul = P.MatMul(transpose_b=True).shard(((parallel_config.data_parallel, 1), (
                parallel_config.model_parallel, 1)))
        self.hidden_size = hidden_size
        self.dtype = compute_type
        self.cast = P.Cast()

    # pylint: disable=W0221
    def construct(self, state, embed):
        state = P.Reshape()(state, (-1, self.hidden_size))
        # output logits over vocabulary [bs*seq_length, vocab_size]
        logits = self.matmul(self.cast(state, self.dtype), self.cast(embed, self.dtype))
        return logits

@MindFormerRegister.register(MindFormerModuleType.MODELS)
class IFlytekSparkModelForCasualLM(IFlytekSparkGeneratorMixin, BaseModel):
    """
    The IFlytekSparkModelForCasualLM network consisting of two parts the backbone and the head
    Args:
        config(IFlytekSparkConfig): the config of network
    Inputs:
        input_ids: the tokenized inputs
        input_position: position id of inputs.
        attention_mask: attention mask
    Returns:
        logits: Tensor: the logits of the corresponding inputs with shape (batch_size, seq_length, vocab_size)

    Examples:
    """

    def __init__(self, config: IFlytekSparkConfig = None):
        # pylint: disable=E1003
        super(IFlytekSparkGeneratorMixin, self).__init__(config, auto_prefix=True)
        dp = config.parallel_config.data_parallel
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        config.is_dynamic = config.is_dynamic if not self.training else False
        self.compute_type = config.compute_type

        # backbone
        self.transformer = IFlytekSparkModel(config)
        # head
        copied_parallel_config = copy.deepcopy(config.parallel_config)
        if copied_parallel_config.pipeline_stage > 1:
            copied_parallel_config.vocab_emb_dp = False
        self.lm_head = LmLogits(config.hidden_size,
                                parallel_config=copied_parallel_config,
                                compute_type=self.compute_type)
        self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1
        self.transformer.embedding.word_embeddings.embedding_table.add_pipeline_stage(self.lm_head.pipeline_stage)
        # loss function
        self.loss = CrossEntropyLoss(config.parallel_config.dp_mp_config)

        self.slice = P.StridedSlice().shard(((dp, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.add = P.Add().shard(((config.parallel_config.data_parallel, 1), ()))
        self.fillv2 = P.FillV2().shard(((1, 1), ()))
        self.use_past = config.use_past
        self.p_all_ones = ms.Tensor(np.ones((config.batch_size,), np.float32), mstype.float32)
        self.is_first_iteration = True

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": ms.Tensor(input_ids, mstype.int32)
        }

    # pylint: disable=W0221
    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, zactivate_len=None):
        r"""Forward process of the pangu alpha model"""
        batch_size, seq_length = self.shape(input_ids)

        if self.phase == "train":
            seq_length = seq_length - 1
            tokens = self.slice(input_ids, (0, 0), (batch_size, -1), (1, 1))
        else:
            tokens = input_ids

        if position_ids:
            position_ids = self.slice(position_ids, (0, 0), (batch_size, seq_length), (1, 1))

        if self.phase == "predict":
            if self.use_past and not self.is_first_iteration:
                attention_mask = self.fillv2((batch_size, seq_length), ms.Tensor(1, ms.float32))
            else:
                attention_mask = F.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        # logits
        output_states, word_table = self.transformer(tokens, position_ids,
                                                     attention_mask=attention_mask,
                                                     init_reset=init_reset,
                                                     batch_valid_length=batch_valid_length,
                                                     zactivate_len=zactivate_len)
        logits = self.lm_head(output_states, word_table)

        input_mask = labels
        if input_mask is None:
            input_mask = F.cast(self.not_equal(tokens, self.vocab_size + 1), mstype.float32)
        else:
            input_mask = self.slice(input_mask, (0, 0), (batch_size, -1), (1, 1))

        if self.phase != "train":
            if self.phase == "predict":
                logits = self.reshape(logits, (-1, logits.shape[-1]))
                # makes cast effective to avoid allgather issue in Mindspore1.10
                input_mask = self.add(input_mask, 1)
                if self.is_first_iteration and input_position is not None:
                    index = input_position.view(-1,)
                    logits = P.Gather()(logits, index, 0)
            else:
                logits = self.reshape(logits, (batch_size, seq_length, -1))
                input_mask = self.add(input_mask, 1)
            return logits, tokens, input_mask

        # Get label corresponding to input tokens
        labels = self.slice(input_ids, (0, 1), (batch_size, seq_length + 1), (1, 1))
        labels = P.Reshape()(labels, (-1,))
        input_mask = P.Reshape()(input_mask, (-1,))
        logits = self.cast(logits, mstype.float32)
        loss = self.loss(logits, labels, input_mask)

        return loss
