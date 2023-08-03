# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from typing import Optional, Tuple, Union
from mindspore import nn, ops
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore.common.initializer import Initializer, initializer, Constant


def masked_fill(inputs, mask, value):
    masked = ops.full_like(inputs, value, dtype=inputs.dtype)
    outputs = ops.select(mask, masked, inputs)
    return outputs


class BucketPositionBias(nn.Cell):
    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        num_segment_bucket: int = 32,
        max_distance: int = 128,
        dtype: mstype.float_ = mstype.half,
        param_init: Union[str, Initializer] = 'normal',
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.num_segment_bucket = num_segment_bucket
        self.max_distance = max_distance

        self.relative_attention_bias = Parameter(
            initializer(param_init, (num_buckets + num_segment_bucket, num_heads), dtype=dtype),
            'relative_attention_bias'
        )
        self.equal = ops.Equal()
        self.sub = ops.Sub()

    def construct(
        self,
        query_pos: Tensor,  # (batch, len_q)
        key_pos: Tensor,  # (batch, len_k)
        rel_buckets: Tensor,  # (batch, len_q, len_k)
    ):

        batch = key_pos.shape[0]
        keylen = key_pos.shape[1]
        querylen = query_pos.shape[1]

        assert key_pos.shape[0] == query_pos.shape[0]
        assert (
            rel_buckets.shape[0] == batch
            and rel_buckets.shape[1] == querylen
            and rel_buckets.shape[2] == keylen
        )

        relative_position_bucket = self.sub(rel_buckets, 1 - self.num_buckets)  # 与相对位置编码区间不重叠

        # b*q*k
        inner_segment_bucket = self._position_bucket(
            key_pos[..., None, :] - query_pos[..., :, None],
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        relative_position_bucket = ops.where(
            self.equal(rel_buckets, 0),
            inner_segment_bucket,
            relative_position_bucket,
        )
        # (batch, len_q, len_k)
        relative_position_bucket = ops.stop_gradient(relative_position_bucket)

        # (batch, len_q, len_k, num_heads)
        embeds = ops.gather(self.relative_attention_bias, relative_position_bucket, 0)
        embeds = ops.cast(embeds, mstype.float16)
        # (batch, num_heads, len_q, len_k)
        embeds = embeds.transpose(0, 3, 1, 2)
        return embeds

    def _position_bucket(self, relative_position, num_buckets=32, max_distance=128):
        relative_buckets = 0
        num_buckets //= 2
        relative_buckets = (relative_position > 0).astype(mstype.int32) * num_buckets
        relative_position = ops.abs(relative_position)

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_postion_if_large = max_exact + (
            ops.log(relative_position.float() / max_exact + 1e-5)
            / ops.log(ops.scalar_to_tensor(max_distance / max_exact) + 1e-5)
            * (num_buckets - max_exact)
        ).astype(mstype.int32)
        relative_postion_if_large = ops.minimum(
            relative_postion_if_large,
            ops.full_like(relative_postion_if_large, num_buckets - 1),
        )
        relative_buckets += ops.where(
            is_small, relative_position.to(mstype.int32), relative_postion_if_large
        )
        return relative_buckets

    def shard(self, dp, mp):
        self.equal.shard(((dp * mp, 1, 1), ()))
        self.sub.shard(((dp * mp, 1, 1), ()))


class RotaryEmbedding(nn.Cell):
    def __init__(
        self,
        dim,
        base=10000,
        distance_scale: Union[int, float] = 1,
        dtype: mstype.float_ = mstype.half,
    ):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
        )
        self.distance_scale = distance_scale
        self.dtype = dtype
        self.inv_freq = Tensor(inv_freq, dtype)
        self.cat = ops.Concat(axis=-1)
        self.cat_3d = ops.Concat(axis=-1)
        self.add = ops.Add()
        self.add_3d = ops.Add()
        self.cos_mul = ops.Mul()
        self.sin_mul = ops.Mul()
        self.cos_mul_3d = ops.Mul()
        self.sin_mul_3d = ops.Mul()

    def construct(self, x: Tensor, x_pos: Tensor):
        """
        Args:
            x (:obj:`Tensor` of shape ``(..., dim)``): Inputs.
            x_pos (:obj:`Tensor` of shape ``(...)``): Positions of inputs.
        """
        x_pos = x_pos * self.distance_scale
        freqs = x_pos[..., None].astype(self.dtype) * self.inv_freq[None, :]  # (..., dim/2)

        # the same implementation as sat
        emb = ops.cat((freqs, freqs), axis=-1)  # (..., dim)
        emb_cos = ops.cos(emb)  # (..., dim)
        emb_sin = ops.sin(emb)  # (..., dim)

        if len(x.shape) > 2:
            rotate_x = self.cat_3d(
                [-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]])  # (..., dim)
            return self.add_3d(self.cos_mul_3d(x, emb_cos), self.sin_mul_3d(rotate_x, emb_sin))
        else:
            rotate_x = self.cat(
                [-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]])  # (..., dim)
            return self.add(self.cos_mul(x, emb_cos), self.sin_mul(rotate_x, emb_sin))

    def shard(self, dp, mp):
        # self.add.shard(((dp * mp, 1), (dp * mp, 1)))
        self.add_3d.shard(((dp, mp, 1), (dp, mp, 1)))


class EmbeddingExt(nn.Cell):
    def __init__(
            self,
            vocab_size: int,
            embedding_size: int,
            distance_scale: int = 16,
            dtype: mstype.float_ = mstype.half,
            param_init: Union[str, Initializer] = 'normal',
    ):

        super().__init__()

        self.dim_model = embedding_size
        self.vocab_size = vocab_size
        self.rotary_emb = RotaryEmbedding(
            dim=embedding_size, distance_scale=distance_scale, dtype=mstype.float32
        )

        self.weight = Parameter(
            initializer(param_init, (vocab_size, embedding_size), dtype=mstype.float32),
            'weight'
        )
        self.weight.parallel_optimizer = True

        self.gather = ops.Gather()
        self.gather_2d = ops.Gather()
        self.matmul = ops.MatMul(transpose_b=True)
        self.matmul_2 = ops.MatMul(transpose_b=True)
        self.cat = ops.Concat(axis=-1)

    def construct(self, ids: Tensor, ids_sub: Tensor):
        """
        Args:
            ids (:obj:`Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens.
            ids (:obj:`Tensor` of shape ``(batch_size)``): Subscript of input sequence tokens.
        Return:
            :obj:`Tensor` of shape ``(batch_size, seq_len, embedding_size)``: The embedding output.
        """  # noqa: E501
        if ids.ndim > 1:
            embeds = self.gather_2d(self.weight, ids, 0) / ops.sqrt(
                ops.scalar_to_tensor(self.dim_model, self.weight.dtype))
        else:
            embeds = self.gather(self.weight, ids, 0) / ops.sqrt(
                ops.scalar_to_tensor(self.dim_model, self.weight.dtype))
        output = self.rotary_emb(embeds, ids_sub)
        output = ops.cast(output, mstype.float16)
        return output

    def projection(self, x: Tensor, ext_table: Optional[Tensor] = None):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size, than projection map embed_size back to vocab_size.
        Args:
            x (:obj:`Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
            ext_table (:obj:`Tensor` of shape ``(ext_table_size, dim_model)``): Ext vocab table.
        Returns:
            :obj:`Tensor` of shape ``(batch, seq_len, vocab_size + ext_table_size)``: The projection output.
        """  # noqa: E501
        x_shape = x.shape
        x = x.reshape((-1, x_shape[-1]))
        x = ops.cast(x, mstype.float16)
        inputs = x / ops.sqrt(ops.scalar_to_tensor(self.dim_model, x.dtype))
        logits = self.matmul(inputs.to(mstype.float16), ops.cast(self.weight, mstype.float16))
        if ext_table is not None:
            logits_ext = self.matmul_2(x, ext_table)
            logits = self.cat([logits, logits_ext])
            logits = ops.reshape(logits, x_shape[:-1] + (self.vocab_size + ext_table.shape[0],))
        else:
            logits = ops.reshape(logits, x_shape[:-1] + (self.vocab_size,))

        return logits

    def shard(self, dp, mp):
        # self.gather.shard(((1, 1), (dp,)))
        self.gather_2d.shard(((1, 1), (dp, 1)))
        self.matmul.shard(((dp, 1), (1, 1)))  # keep same strategy with gather
        self.matmul_2.shard(((dp, 1), (1, 1)))
        self.rotary_emb.shard(dp, mp)


class Linear(nn.Cell):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dtype: mstype.float_ = mstype.half,
        param_init: Union[str, Initializer] = 'normal',
        scale_before: bool = False,
    ):
        super().__init__()
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.scale_before = scale_before

        self.weight = Parameter(initializer(param_init, (dim_out, dim_in), dtype=dtype), 'weight')
        self.matmul = ops.MatMul(transpose_b=True)

    def construct(self, x: Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501
        x_shape = x.shape
        x = ops.cast(x, mstype.float16)
        if self.scale_before:
            x = x / ops.sqrt(ops.scalar_to_tensor(self.dim_in, mstype.float16))
            x = self.matmul(x.reshape(-1, x_shape[-1]), ops.cast(self.weight, mstype.float16))
        else:
            x = self.matmul(x.reshape(-1, x_shape[-1]), ops.cast(self.weight, mstype.float16))
            x = x / ops.sqrt(ops.scalar_to_tensor(self.dim_in, mstype.float16))
        x = x.reshape(x_shape[:-1] + (x.shape[-1],))
        return x

    def shard(self, dp, mp):
        self.matmul.shard(((dp, 1), (mp, 1)))


class LayerNorm(nn.Cell):
    """RMS LayerNorm"""

    def __init__(
            self,
            dim_norm: int,
            dtype: mstype.float_ = mstype.half,
            eps: float = 1e-6,
            init_var: float = 1.0,
    ):
        super().__init__()

        self.eps = eps
        self.dim_norm = dim_norm
        self.weight = Parameter(initializer(Constant(init_var), (dim_norm,), dtype=dtype), 'weight')
        self.cast = ops.Cast()
        self.rsqrt = ops.Rsqrt()
        self.mean = ops.ReduceMean(keep_dims=True)
        self.square = ops.Square()
        self.add = ops.Add()
        self.mul = ops.Mul()
        self.expand = ops.ExpandDims()

    def construct(self, x: Tensor):
        """
        Args:
            x (:obj:`Tensor` of shape ``(batch_size, seq_len, dim_norm)``): Input tensor that need to be normalized.
        Return:
            :obj:`Tensor` of shape ``(batch_size, seq_len, dim_norm)``: The layernorm output.
        """  # noqa: E501
        assert x.shape[-1] == self.dim_norm
        old_dtype = x.dtype
        variance = self.cast(x, mstype.float32)
        variance = self.square(variance)
        variance = self.mean(variance, -1)
        x = (x * self.rsqrt(self.add(variance, self.eps))).astype(mstype.float16)
        return self.mul(x, ops.cast(self.weight, mstype.float16))

    def shard(self, dp, mp):
        self.cast.shard(((dp, mp, 1),))
        self.square.shard(((dp, mp, 1),))
        self.mean.shard(((dp, mp, 1),))
        self.add.shard(((dp, mp, 1), ()))
        self.expand.shard(((dp, mp, 1), ()))
        self.mul.shard(((dp, mp, 1), (1,)))


class DenseGatedACT(nn.Cell):
    def __init__(
        self,
        dim_in: int,
        dim_ff: int,
        dtype=mstype.half,
    ):
        super().__init__()

        self.w_0 = Linear(
            dim_in=dim_in,
            dim_out=dim_ff,
            dtype=dtype,
            scale_before=False,
        )

        self.w_1 = Linear(
            dim_in=dim_in,
            dim_out=dim_ff,
            dtype=dtype,
            scale_before=False,
        )
        self.act = nn.GELU(False)
        # self.act.recompute()

    def construct(self, x: Tensor):
        """Transform an input tensor from one feature space to another via a nonlinear operation

        Args:
            x (:obj:`Tensor` of shape ``(batch, seq_len, dim_in)``): Tensor that will be subject to nonlinear operations.

        Return:
            out (:obj:`Tensor` of shape ``(batch, seq_len, dim_ff)``)

        """  # noqa: E501
        gate_score = self.act(self.w_0(x))
        x = self.w_1(x)

        x = gate_score * x
        return x

    def shard(self, dp, mp):
        self.w_0.shard(dp, mp)
        self.w_1.shard(dp, mp)


class FeedForward(nn.Cell):
    r"""FeedForward module

    Args:
        dim_in (int): input dimension.
        dim_ff (int): middle dimension.
        dim_out (int, optional): output dimension. Defaults to None, which means dim_in = dim_out.
        dtype (optional): Defaults to torch.half.
        init_mean (float, optional): mean of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in feed-forward layer. Defaults to 0.
        init_std (float, optional): std of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in feed-forward layer. Defaults to 0.02.
        bias (bool, optional): whether to use bias term in fully-connected layers used in feed-forward module. Defaults to False.
        activate_fn (str, optional): Defaults to `gated_gelu`.
        dropout_p (int, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        dtype=mstype.half,
        dropout_p: Optional[float] = None,
    ):

        super().__init__()

        self.w_in = DenseGatedACT(
            dim_in=dim_model,
            dim_ff=dim_ff,
            dtype=dtype,
        )

        if dropout_p is not None:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = None

        self.w_out = Linear(
            dim_in=dim_ff,
            dim_out=dim_model,
            dtype=dtype,
            scale_before=False,
        )

    def construct(self, x: Tensor):
        """
        Args:
            x (:obj:`Tensor` of shape ``(batch, seq_len, dim_in)``): The input of feed-forward module.

        Return:
            :obj:`Tensor` of shape ``(batch, seq_len, dim_out)``: The output of feed-forward module.
        """  # noqa: E501
        x = self.w_in(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.w_out(x)

        return x

    def shard(self, dp, mp):
        self.w_in.shard(dp, mp)
        self.w_out.shard(dp, mp)


class Attention(nn.Cell):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_head: int,
        dtype: mstype.float_ = mstype.half,
        dropout_p: Optional[float] = None,
    ) -> None:

        super().__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head

        self.project_q = Linear(self.dim_model, self.num_heads * self.dim_head, dtype=dtype)
        self.project_k = Linear(self.dim_model, self.num_heads * self.dim_head, dtype=dtype)
        self.project_v = Linear(self.dim_model, self.num_heads * self.dim_head, dtype=dtype)

        self.attention_out = Linear(self.num_heads * self.dim_head, self.dim_model, dtype=dtype)

        self.softmax = nn.Softmax(axis=-1)

        if dropout_p is not None:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = None
        self.matmul = ops.BatchMatMul()
        self.matmul_2 = ops.BatchMatMul()

    def construct(
        self,
        hidden_q: Tensor,
        hidden_kv: Tensor,
        attention_mask: Tensor,
        position_bias: Tensor,
        use_cache: bool = False,
        past_kv: Optional[Tuple[Tensor, Tensor]] = None,
    ):
        """
        Args:
            hidden_q (:obj:`Tensor` of shape ``(batch, len_q, dim_model)``): Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.
            hidden_kv (:obj:`Tensor` of shape ``(batch, len_k, dim_model)``): Length of input sequence before padding.
            attention_mask (:obj:`Tensor` of shape ``(batch, len_q, len_k)``): Used to avoid performing attention on padding token indices.
            position_bias(:obj:`Tensor` of shape ``(num_heads, len_q, len_k)`` or ``(1, num_heads, len_k, len_q)``): Provide positional information about tensor `key_value` and `query`.
        Return:
            out (:obj:`Tensor` of shape ``(batch, len_q, dim_model)``): The attention output.
        """  # noqa: E501

        batch_size = hidden_q.shape[0]
        len_q = hidden_q.shape[1]
        len_k = hidden_kv.shape[1]

        h_q = self.project_q(hidden_q)
        h_k = self.project_k(hidden_kv)
        h_v = self.project_v(hidden_kv)

        h_q = h_q.view((batch_size, len_q, self.num_heads, self.dim_head)).transpose(0, 2, 1, 3)
        h_k = h_k.view((batch_size, len_k, self.num_heads, self.dim_head)).transpose(0, 2, 1, 3)
        h_v = h_v.view((batch_size, len_k, self.num_heads, self.dim_head)).transpose(0, 2, 1, 3)

        if past_kv is not None:
            h_k = ops.cat([past_kv[0], h_k], axis=-2)
            h_v = ops.cat([past_kv[1], h_v], axis=-2)
            len_k = h_k.shape[-2]

        # (b, n_h, len_q, d_h) @ (b, n_h, d_h, len_k) -> (b, n_h, len_q, len_k)
        score = self.matmul(h_q, h_k.swapaxes(-1, -2)) / ops.sqrt(ops.scalar_to_tensor(self.dim_head, h_k.dtype))
        score = score + ops.cast(position_bias, mstype.float16)
        score = masked_fill(
            score,
            attention_mask.view((batch_size, 1, len_q, len_k)) == False,
            -10000,
        )

        score = self.softmax(score)
        score = masked_fill(
            score,
            attention_mask.view((batch_size, 1, len_q, len_k)) == False,
            0,
        )

        if self.dropout is not None:
            score = self.dropout(score)

        # (b, n_h, len_q, len_k) @ (b, n_h, len_k, d_h) -> (b, n_h, len_q, d_h)
        score = self.matmul_2(score, h_v)

        score = score.view((batch_size, self.num_heads, len_q, self.dim_head)).transpose(0, 2, 1, 3)
        score = score.view((batch_size, len_q, self.num_heads * self.dim_head))

        score = self.attention_out(score)
        if use_cache:
            return score, (h_k, h_v)
        else:
            return score

    def shard(self, dp, mp):
        self.project_q.shard(dp, mp)
        self.project_k.shard(dp, mp)
        self.project_v.shard(dp, mp)
        self.attention_out.shard(dp, mp)


class SelfAttentionBlock(nn.Cell):
    """The whole cross-attention block. A sequence of operation. Consists of layernorm, self-attention and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.half.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
            self,
            dim_model: int,
            num_heads: int,
            dim_head: int,
            dtype=mstype.half,
            eps: float = 1e-6,
            dropout_p: Optional[float] = None,
    ):

        super().__init__()

        self.layernorm_before_attention = LayerNorm(
            dim_model,
            dtype=dtype,
            eps=eps,
        )

        self.self_attention = Attention(
            dim_model=dim_model,
            num_heads=num_heads,
            dim_head=dim_head,
            dtype=dtype,
            dropout_p=dropout_p,
        )

        if dropout_p:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = None

    def construct(
            self,
            hidden_states: Tensor,
            attention_mask: Tensor,
            position_bias: Optional[Tensor] = None,
            use_cache: bool = False,
            past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
    ):
        """
        Args:
            hidden_states (:obj:`Tensor` of shape ``(batch, seq_self, dim_model)``): Input of self-attention block. It can be the embedding of a batch of sequences.
            attention_mask (:obj:`Tensor` of shape ``(batch, seq_self, seq_self)``): Avoid invalid areas to participate in the calculation.
            position_bias (:obj:`Tensor` of shape ``(num_heads, seq_self, seq_self)``): Provide positional information to self-attention block.

        Return:
            :obj:`Tensor` of shape ``(batch, seq_self, dim_model)``: The output of attention block.

        """  # noqa: E501
        hidden_states = ops.cast(hidden_states, mstype.float16)
        x = self.layernorm_before_attention(hidden_states)
        x = self.self_attention(x, x, attention_mask, position_bias, use_cache, past_key_value)
        if use_cache:
            x, current_key_value = x
        else:
            current_key_value = None

        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = (hidden_states + x) / 1.05

        if use_cache:
            return hidden_states, current_key_value
        else:
            return hidden_states

    def shard(self, dp, mp):
        self.self_attention.shard(dp, mp)
        self.layernorm_before_attention.shard(dp, mp)


class FFNBlock(nn.Cell):
    """The whole feed-forward block. A sequence of operation. Consists of layernorm, feed-forward and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        dtype (optional): Defaults to torch.half.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
            self,
            dim_model: int,
            dim_ff: int,
            dtype=mstype.half,
            eps: float = 1e-6,
            dropout_p: Optional[float] = 0,
    ):
        super().__init__()

        self.layernorm_before_ffn = LayerNorm(
            dim_model,
            dtype=dtype,
            eps=eps,
        )

        self.ffn = FeedForward(
            dim_model,
            dim_ff,
            dtype=dtype,
            dropout_p=dropout_p,
        )

        if dropout_p:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = None

    def construct(
            self,
            hidden_states: Tensor,
    ):
        """
        Args:
            hidden_states (:obj:`Tensor` of shape ``(batch, seq_self, dim_model)``): Hidden states before feed forward layer.

        Return:
            :obj:`Tensor` of shape ``(batch, seq_self, dim_model)``: The output of feed-forward block

        """  # noqa: E501
        hidden_states = ops.cast(hidden_states, mstype.float16)
        x = self.layernorm_before_ffn(hidden_states)
        x = self.ffn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = (hidden_states + x) / 1.05
        return hidden_states

    def shard(self, dp, mp):
        self.ffn.shard(dp, mp)
        self.layernorm_before_ffn.shard(dp, mp)


class TransformerBlock(nn.Cell):
    """The whole transformer block. A sequence of operation. Consists of self-attention block[, cross-attention block] and feed-forward block.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.half.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
            self,
            dim_model: int,
            dim_ff: int,
            num_heads: int,
            dim_head: int,
            dtype=mstype.half,
            eps: float = 1e-6,
            dropout_p: Optional[float] = None,
            mask_att: bool = False,
            mask_ffn: bool = False,
    ):
        super().__init__()
        self.mask_att = mask_att
        self.mask_ffn = mask_ffn

        if not self.mask_att:
            self.self_att = SelfAttentionBlock(
                dim_model=dim_model,
                num_heads=num_heads,
                dim_head=dim_head,
                dtype=dtype,
                eps=eps,
                dropout_p=dropout_p,
            )

        if not self.mask_ffn:
            self.ffn = FFNBlock(
                dim_model=dim_model,
                dim_ff=dim_ff,
                dtype=dtype,
                eps=eps,
                dropout_p=dropout_p,
            )

    def construct(
            self,
            self_hidden_states: Tensor,
            self_attention_mask: Tensor,
            self_position_bias: Optional[Tensor] = None,
            use_cache: bool = False,
            past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
    ):
        """
        Args:
            self_hidden_states (:obj:`Tensor` of shape ``(batch, seq_self, dim_model)``): Input of transformer block(self-attention block). It can be the raw embedding of a batch of sequences.
            self_attention_mask (:obj:`Tensor` of shape ``(batch, seq_self, seq_self)``): Avoid invalid areas to participate in the calculation of self-attention.
            self_position_bias (:obj:`Tensor` of shape ``(num_heads, seq_self, seq_self)``): Provide positional information to self-attention block.

        Return:
            :obj:`Tensor` of shape ``(batch, seq_self, dim_model)``: The output of transformer block.

        """  # noqa: E501
        # (batch, dim_model, seq_self)
        current_key_value = None
        if not self.mask_att:
            hidden_states = self.self_att(
                self_hidden_states,
                attention_mask=self_attention_mask,
                position_bias=self_position_bias,
                use_cache=use_cache,
                past_key_value=past_key_value,
            )
            if use_cache:
                hidden_states, current_key_value = hidden_states
        else:
            hidden_states = self_hidden_states

        # (batch, dim_model, seq_self)
        if not self.mask_ffn:
            hidden_states = self.ffn(hidden_states)

        if use_cache:
            return hidden_states, current_key_value
        else:
            return hidden_states

    def shard(self, dp, mp):
        if not self.mask_att:
            self.self_att.shard(dp, mp)
        if not self.mask_ffn:
            self.ffn.shard(dp, mp)
