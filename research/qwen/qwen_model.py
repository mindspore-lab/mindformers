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
# ============================================================================
"""Qwen_7B models' APIs."""

import numpy as np
import mindspore.common.dtype as mstype
from mindspore import log as logger
from mindspore import nn
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
try:
    # pylint: disable=W0611
    from mindspore.nn.layer.flash_attention import FlashAttention

    FLASHATTENTION_VALID = True
except ImportError:
    FLASHATTENTION_VALID = False

from mindformers.models.base_model import BaseModel
from mindformers.modules.layers import Linear, _check_input_dtype, _args_type_validator_check, _valid_value_checks
from mindformers.tools.logger import _LogActionOnce
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.models.utils import cell_reuse
from mindformers.modules.transformer import TransformerOpParallelConfig
from mindformers.models.llama.llama_layer import LlamaEmbedding
from mindformers.models.llama.llama_transformer import LLamaDecodeLayer


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class QwenForCausalLM(BaseModel):
    r"""
        Provide qwen training loss or logits through network.
        Args:
            config (QwenConfig): The config of Qwen model.

        Returns:
            Tensor, the loss or logits of the network.

        Examples:
            >>> from mindformers.models.llama import LlamaConfig
            >>> config = LlamaConfig(batch_size=2)
            >>> network = QwenForCausalLM(config=config)
        """

    @cell_reuse
    def __init__(self, config=None):
        super().__init__(config)

        self.transformer = QwenModel(config=config)
        self.lm_head = Linear(in_channels=config.hidden_size,
                              out_channels=config.vocab_size,
                              has_bias=False,
                              compute_dtype=config.compute_dtype,
                              param_init_type=config.param_init_type,
                              weight_init="normal")
        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.not_equal = P.NotEqual()
        self.cast = P.Cast()
        self.add = P.Add()
        self.reshape = P.Reshape()
        self.ones = P.Ones()

        self.load_checkpoint(config)

    # pylint: disable=W0613
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": Tensor(input_ids, mstype.int32)
        }

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None):
        """construct"""
        bsz, seqlen = input_ids.shape
        if self.use_past:
            if not isinstance(init_reset, Tensor):
                init_reset = Tensor([init_reset], mstype.bool_)
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz, 1), mstype.int32)

        out = self.transformer(input_ids, init_reset=init_reset, batch_valid_length=batch_valid_length)
        logits = self.lm_head(out)

        input_mask = self.cast(self.not_equal(input_ids, self.pad_token_id), mstype.float32)
        logits = self.cast(logits, mstype.float32)
        logits = self.reshape(logits, (bsz, seqlen, -1))

        # makes cast effective to avoid allgather issue in Mindspore1.10
        input_mask = self.add(input_mask, 1)
        return logits, input_ids, input_mask


class QwenModel(BaseModel):
    """transformer"""

    def __init__(self, config):
        super().__init__(config)
        self.dtype = config.compute_dtype
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_dim = config.hidden_size
        self.seq_length = config.seq_length
        self.pad_token_id = config.pad_token_id
        self.num_attention_heads = config.num_attention_heads
        self.use_past = config.use_past

        self.is_first_iteration = True
        self.use_flash_attention = config.use_flash_attention and FLASHATTENTION_VALID
        if self.use_flash_attention:
            logger.info("Enable flash attention.")
        elif config.use_flash_attention:
            logger.info("Current MindSpore do not support flash attention.")

        # 1. wte
        self.wte = LlamaEmbedding(self.vocab_size, self.embed_dim, param_init_type=config.param_init_type)

        # 2. drop
        self.drop = nn.Dropout(p=config.emb_dropout_prob)

        # 4. h hidden layers for transformer
        self.layers = nn.CellList()
        for layer_id in range(config.num_hidden_layers):
            layer = QwenDecodeLayer(config.batch_size,
                                    config.seq_length,
                                    layer_id,
                                    dim=config.hidden_size,
                                    n_heads=config.num_attention_heads,
                                    multiple_of=config.multiple_of,
                                    norm_eps=config.rms_norm_eps,
                                    compute_dtype=config.compute_dtype,
                                    layernorm_compute_dtype=config.layernorm_compute_type,
                                    softmax_compute_dtype=config.softmax_compute_type,
                                    rotary_dtype=config.rotary_dtype,
                                    param_init_type=config.param_init_type,
                                    qkv_has_bias=True,
                                    use_past=config.use_past,
                                    parallel_config=config.parallel_config)

            from mindformers.models.llama.llama import layer_compute_dtype
            layer_compute_dtype(layer, layer_id, config.offset,
                                config.parallel_config, config.num_hidden_layers)

            self.layers.append(layer)

        from mindformers.modules.transformer.transformer import AttentionMask
        self.get_attention_mask = AttentionMask(
            config.seq_length, parallel_config=config.parallel_config.dp_mp_config).to_float(config.compute_dtype)

        from mindformers.models.llama.llama_layer import precompute_freqs_cis
        self.freqs_cos, self.freqs_sin, self.swap_mask = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads, config.seq_length)

        # 5. ln_f
        from mindformers.models.llama.llama_layer import LlamaRMSNorm
        self.ln_f = LlamaRMSNorm(
            self.embed_dim,
            eps=config.rms_norm_eps,
            compute_type=config.layernorm_compute_type
        )

        self.assign = P.Assign()
        self.slice = P.StridedSlice()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.tile = P.Tile()
        self.mul_mask = P.Mul()
        self.sub = P.Sub()
        self.expand_dims = P.ExpandDims()
        self.not_equal = P.NotEqual()
        self.gather = P.Gather()

        self.multiply_data = Tensor([-10000.0], dtype=config.compute_dtype)
        self.one = Tensor([1.0], dtype=config.compute_dtype)
        if self.use_past:
            seq_range = np.arange(config.seq_length).reshape(1, 1, -1)
            self.range = Tensor(np.tile(seq_range, (config.batch_size, 1, 1)), mstype.int32)
            self.gather_past = P.Gather()
            self.expand_dims = P.ExpandDims()
            self.le_past = P.LessEqual()

    def construct(self, input_ids: Tensor, init_reset=True, batch_valid_length=None):
        """construct"""
        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])

        # 1. wte
        hidden_states = self.wte(input_ids)

        # 3. drop
        hidden_states = self.drop(hidden_states)

        # 2. rotary_emb
        bs, seq_len = input_ids.shape
        if self.is_first_iteration:
            freqs_cis = (self.tile(self.reshape(self.freqs_cos, (1, 1, seq_len, -1)), (bs, 1, 1, 1)),
                         self.tile(self.reshape(self.freqs_sin, (1, 1, seq_len, -1)), (bs, 1, 1, 1)),
                         self.swap_mask)
            input_mask = self.cast(self.not_equal(input_ids, self.pad_token_id), self.dtype)
            mask = self.get_attention_mask(input_mask)
        else:
            cur_pos = batch_valid_length - 1
            valid_length = self.reshape(cur_pos, (-1, 1, 1))
            freqs_cis = (self.reshape(self.gather_past(self.freqs_cos, cur_pos, 0), (bs, 1, seq_len, -1)),
                         self.reshape(self.gather_past(self.freqs_sin, cur_pos, 0), (bs, 1, seq_len, -1)),
                         self.swap_mask)
            mask = self.cast(self.le_past(self.range, valid_length), self.dtype)
        mask = self.sub(self.one, self.cast(mask, self.dtype))
        if not self.use_flash_attention:
            mask = self.expand_dims(mask, 1)
            mask = self.mul_mask(mask, self.multiply_data)

        # 4. hidden_states
        for i in range(self.num_hidden_layers):
            hidden_states, _ = self.layers[i](hidden_states, freqs_cis, mask, init_reset=init_reset,
                                              batch_valid_length=batch_valid_length)

        # 5. ln_f
        hidden_states = self.ln_f(hidden_states)

        return hidden_states


class QwenDecodeLayer(LLamaDecodeLayer):
    """Qwen decode layer"""

    def __init__(self,
                 batch_size,
                 seq_length,
                 layer_id,
                 **kwargs):
        kwargs['qkv_has_bias'] = True
        super().__init__(batch_size, seq_length, layer_id, **kwargs)

        rotary_dtype = kwargs.get('rotary_dtype', mstype.float32)
        self.attention.apply_rotary_emb = QwenRotaryEmbedding(self.head_dim, rotary_dtype)

        multiple_of = kwargs.get('multiple_of', 256)
        ffn_dim_multiplier = kwargs.get('ffn_dim_multiplier', None)
        compute_dtype = kwargs.get('compute_dtype', mstype.float16)
        param_init_type = kwargs.get('param_init_type', mstype.float32)
        self.feed_forward = QwenFeedForward(dim=self.hidden_size,
                                            hidden_dim=4 * self.hidden_size,
                                            multiple_of=multiple_of,
                                            ffn_dim_multiplier=ffn_dim_multiplier,
                                            compute_dtype=compute_dtype,
                                            param_init_type=param_init_type)

        parallel_config = kwargs.get('parallel_config', TransformerOpParallelConfig())
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.feed_forward.shard(parallel_config)
            if not self.compute_in_2d:
                self.feed_forward.mul.shard(((dp, 1, mp), (dp, 1, mp)))
        if parallel_config.use_seq_parallel and self.is_first_iteration:
            self.feed_forward.w2.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))


class QwenRotaryEmbedding(nn.Cell):
    r"""
    Rotary Position Embedding.

    Args:
            - **head_dim** (int): The dim of multi head attention.
            - **compute_dtype** (mstype): The compute type, default mstype.float16.
    Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

    Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, head_dim=128, compute_dtype=mstype.float32):
        super().__init__(auto_prefix=False)
        self.head_dim = head_dim
        self.dtype = compute_dtype

        self.add = P.Add()
        self.mul = P.Mul()
        self.split = P.Split(3, 2)
        self.concat = P.Concat(3)

        self.cast = P.Cast()

    def rotate_half(self, x):
        x1, x2 = self.split(x)
        return self.concat((-x2, x1))

    def construct(self, xq: Tensor, xk: Tensor, freqs_cis):
        """Forward of rotary position embedding."""
        original_type = xq.dtype
        xq = self.cast(xq, self.dtype)
        xk = self.cast(xk, self.dtype)
        # xq, xk: [bs, n_head/n_kv_head, seq/1, head_dim]
        freqs_cos, freqs_sin, _ = freqs_cis
        xq_out = self.add(self.mul(xq, freqs_cos),
                          self.mul(self.rotate_half(xq), freqs_sin))
        xk_out = self.add(self.mul(xk, freqs_cos),
                          self.mul(self.rotate_half(xk), freqs_sin))

        xq_out = self.cast(xq_out, original_type)
        xk_out = self.cast(xk_out, original_type)
        return xq_out, xk_out


class QwenFeedForward(nn.Cell):
    r"""
    Qwen FeedForward.

    .. math::
            (xW_1 * xW_3)W_2

        Inputs:
            - **x** (Tensor) - should be `[batch, seq_length, hidden_size] or [batch * seq_length, hidden_size]`.
              Float tensor.

        Outputs:
            Tensor, the output of this layer after mapping. The shape is `[batch, seq_length, hidden_size] or
            [batch * seq_length, hidden_size]`.

        Raises:
            ValueError: `hidden_dim` is not a multiple of the model parallel way.
            ValueError: `dim` is not a multiple of the model parallel way.
    """

    @_LogActionOnce(m_logger=logger, key='FeedForward',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(dim=Validator.check_positive_int,
                                hidden_dim=Validator.check_positive_int,
                                multiple_of=Validator.check_positive_int,
                                compute_dtype=_valid_value_checks([mstype.float32, mstype.float16],
                                                                  "FeedForward"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                    "FeedForward"))
    def __init__(self, dim,
                 hidden_dim,
                 multiple_of,
                 ffn_dim_multiplier=None,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32):
        super().__init__()

        if ffn_dim_multiplier is not None:
            hidden_dim = int((ffn_dim_multiplier + 0.01) * hidden_dim)
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * \
                     ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dtype = compute_dtype
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.mul = P.Mul()
        self.cast = P.Cast()
        self.w1 = Linear(in_channels=dim,
                         out_channels=hidden_dim,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)

        self.w2 = Linear(in_channels=hidden_dim,
                         out_channels=dim,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)

        self.w3 = Linear(in_channels=dim,
                         out_channels=hidden_dim,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)

    def construct(self, x):
        """Forward process of the FeedForward"""
        _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16], self.cls_name)
        x = self.cast(x, self.dtype)
        # [bs, seq, hidden_dim] or [bs * seq, hidden_dim]
        gate = self.w1(x)  # dp,1 -> dp, mp
        hidden = self.w3(x)  # dp,1 -> dp, mp
        hidden = self.mul(gate, F.silu(hidden).astype(self.dtype))  # dp,mp -> dp, mp
        output = self.w2(hidden)  # dp,mp -> dp, 1
        return output

    def shard(self, parallel_config):
        """sharding for feedforward"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        if self.hidden_dim % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'hidden_dim' must be a multiple of the"
                             "num of model parallel, but got the hidden_dim is {} and the num of model "
                             "parallel is {}.".format(self.hidden_dim, mp))
        if self.dim % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'dim' must be a multiple of the num of "
                             "model parallel, but got the dim is {} and the num of model parallel is {}."
                             .format(self.dim, mp))
        self.w1.shard(((dp, 1), (mp, 1)), strategy_activation=((dp, mp),))
        self.w2.shard(((dp, mp), (1, mp)))
        self.w3.shard(((dp, 1), (mp, 1)))
        self.mul.shard(((dp, mp), (dp, mp)))
