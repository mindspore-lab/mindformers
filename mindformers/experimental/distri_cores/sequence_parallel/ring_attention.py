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
"""Ring Attention APIs."""
from einops import rearrange

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, ops
from mindspore.ops.operations._inner_ops import Send, Receive
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from mindspore.communication import get_group_size

from utils import get_sequence_parallel_world_size, get_sequence_parallel_group, get_sequence_parallel_rank


class RingAttention(nn.Cell):
    """Attention implementation with sequence parallelism
    This function contains the ring attention primitives used in RingAttention
    Specifically, it includes an interface for calling ringattention operation.

    B -- Batch size
    S1 -- Sequence length of query. The value ranges from 1 to 32768 and is a multiple of 16.
    S2 -- Sequence length of key and value. The value ranges from 1 to 32768 and is a multiple of 16.
    N1 -- Num heads of query
    N2 -- Num heads of key and value, and N2 must be a factor of N1
    D -- Head size. Support value: 64, 80, 96, 120, 128 and 256.
    H1 -- Hidden size of query, which equals to N1 * D
    H2 -- Hidden size of key and value, which equals to N2 * D
    Args:
        head_num (int): The head num of query.
        keep_prob (float): The keep probability of dropout. Default: 1.0.
        scale_value (float): The scale factor of score. Default: 1.0.
        pre_tokens (int): Parameter for sparse computation, represents how many tokens are counted forward.
        When sparse_mode is set to 1, 2, 3, or 5, this parameter does not take effect. Default: 2147483647.
        next_tokens (int): Parameter for sparse computation, represents how many tokens are counted backward.
        When sparse_mode is set to 1, 2, 3, or 5, this parameter does not take effect. Default: 2147483647.
        input_layout (str): Specifies the layout of input `query`, key and value. Currently only the value
        "SBH" is supported. Default: "SBH". Currently only input_layout = "SBH" is supported.
        sparse_mode (int): Indicates sparse mode. Default 0. Currently only sparse_mode = 0 is supported.

            - 0: Indicates the defaultMask mode. If attn_mask is not passed, the mask operation is not performed,
              and preTokens and nextTokens(internally assigned as INT_MAX) are ignored. If passed in, the full attn_mask
              matrix (S1 * S2) needs to be passed in, indicating that the part between preTokens and nextTokens needs to
              be calculated.
            - 1: Represents allMask, that is, passing in the complete attn_mask matrix.
            - 2: Representing the leftUpCausal mode corresponds to the lower triangle scenario divided by the left
              vertex, and the optimized attn_mask matrix (2048*2048) is required.
            - 3: Representing the rightDownCausal model corresponds to the lower triangle scene divided by the lower
              right vertex, and the optimized attn_mask matrix (2048*2048) is required.
            - 4: Represents the band scenario, that is, the part between counting preTokens and nextTokens, and the
              optimized attn_mask matrix (2048*2048) is required..
            - 5: Represents the prefix scenario, that is, on the basis of rightDownCasual, a matrix with length S1 and
              width N is added to the left side. The value of N is obtained by the new input prefix, and the N value of
              each Batch axis is different. Not implemented yet.
            - 6: Represents the global scenario, not implemented yet.
            - 7: Represents the dilated scenario, not implemented yet.
            - 8: Represents the block_local scenario, not implemented yet.
        use_attention_mask (bool): The value is True if attention_mask is passed. Default: False.
        use_alibi_mask (bool): The value is True if alibi_mask is passed. Default: False.
        Currently only use_alibi_mask = False is supported.
        use_mqa (bool): Specifies whether using MQA. Default: False. Currently only use_mqa = False is supported.
        dp (int): Data parallel num.
        mp (int): Model parallel num. Currently only mp = 1 is supported.
        sp (int): Sequence parallel num.


    Inputs:
        - **query** (Tensor[float16, bfloat16]) - The query tensor.
          Input tensor of shape :math:`(B, S1, H1)` or `(B, N1, S1, D)`.
        - **key** (Tensor[float16, bfloat16]) - The key tensor.
          Input tensor of shape :math:`(B, S2, H2)` or `(B, N2, S2, D)`.
        - **value** (Tensor[float16, bfloat16]) - The value tensor.
          Input tensor of shape :math:`(B, S2, H2)` or `(B, N2, S2, D)`.
        - **attn_mask** (Union[Tensor[uint8], None]) - The attention mask tensor. For each element, 0 indicates
          retention and 1 indicates discard. Input tensor of shape :math:`(B, N1, S1, S2)`, `(B, 1, S1, S2)`, `(S1, S2)`
          or (2048, 2048). Currently only attn_mask = None is supported and it indicates the causal mask is used.
        - **alibi_mask** (Union[Tensor[float16, bfloat16], None]) - The position embedding code. If S is greater than
          1024 and the mask of the lower triangle is used, enter only the inverse 1024 lines of the lower triangle for
          memory optimization. Currently only alibi_mask = None is supported.
          Input tensor of shape :math: `(B, N1, S1, S2)`, `(1, N1, S1, S2)`, `(B, N1, 1024, S2)`, `(1, N1, 1024, S2)`
          or (1024, 1024).
        - **padding_mask** (None) - Reserved parameter. Not implemented yet.
          Currently only padding_mask = None is supported.
        - **prefix** (Union[Tensor[int64], None]) - N value of each Batch in the prefix sparse calculation scenario.
          Not implemented yet. Input tensor of shape :math:`(B,)`. Currently only prefix = None is supported.

    Outputs:
        - **attention_out** (Tensor[float16, bfloat16]) - The output of attention, its shape, and data type
          are the same as the query.

    Supported Platforms:
        ``Ascend910B``

    Examples:
-
        (1, 16, 2048)
    """
    def __init__(self,
                 head_num,
                 keep_prob=1.0,
                 scale_value=1.0,
                 pre_tokens=2147483647,
                 next_tokens=2147483647,
                 input_layout="SBH",
                 sparse_mode=0,
                 use_attention_mask=False,
                 use_alibi_mask=False,
                 use_mqa=False,
                 dp=1,
                 mp=1,
                 sp=1
                 ):
        super(RingAttention, self).__init__()
        self.head_num = head_num
        self.keep_prob = keep_prob
        self.scale_value = scale_value
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens
        self.input_layout = input_layout
        self.sparse_mode = sparse_mode
        self.use_attention_mask = use_attention_mask
        self.use_alibi_mask = use_alibi_mask
        self.use_mqa = use_mqa
        self.dp = dp
        self.mp = mp
        self.sp = sp

        if sparse_mode != 0:
            raise ValueError(f"Only sparse_mode = 0 is supported")

        if input_layout != "SBH":
            raise ValueError(f"Only input_layout = 'SBH' is supported")

        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        if parallel_mode not in (ms.ParallelMode.STAND_ALONE, ms.ParallelMode.DATA_PARALLEL):
            raise ValueError(f"The ring-attention only supports stand_alone and data_parallel,"
                             f"but got the paralle mode of {parallel_mode}")
        if parallel_mode == ms.ParallelMode.STAND_ALONE:
            if dp != 1 or sp != 1:
                raise ValueError(f"Current parallel mode is stand_alone, dp and sp must be 1",
                                 f"but got the dp is f{dp}, sp is {sp}")
        else:
            world_size = get_group_size()
            if dp * sp != world_size:
                raise ValueError(f"The product of dp and sp should be equal to total device number,"
                                 f"but got dp = {dp}, sp = {sp} and total device number = {world_size}")

        init_sp = get_sequence_parallel_world_size()
        if sp != init_sp:
            raise ValueError(f"The sp group is initialized as {init_sp},"
                             f"but got different sp = {sp} in RingAttention parameters")

        if self.use_alibi_mask:
            raise ValueError(f"Only use_alibi_mask = False is supported")

        if self.use_mqa:
            raise ValueError(f"Only use_mqa = False is supported")

        if self.mp != 1:
            raise ValueError(f"Only mp = 1 is supported")

        self.flash_attention = FlashAttentionScore(head_num=self.head_num,
                                                   keep_prob=self.keep_prob,
                                                   scale_value=self.scale_value,
                                                   pre_tokens=self.pre_tokens,
                                                   next_tokens=self.next_tokens,
                                                   input_layout=self.input_layout,
                                                   inner_precise=0,
                                                   sparse_mode=self.sparse_mode)
        self.stream_send = ms.hal.Stream()
        self.stream_recv = ms.hal.Stream()

    def p2p_communicate(self, rank, send_tensor, send_dst,
                        recv_tensor, recv_src,
                        sp_group, stream_send, stream_recv):
        """Point-to-point communications of KV and dKV in Attention with sequence parallelism"""

        send_recv_ops = []

        send_op = Send(0, send_dst, group=sp_group)
        send_op.add_prim_attr("dtype", mstype.float16)
        recv_op = Receive(0, recv_src, shape=recv_tensor.shape, dtype=recv_tensor.dtype, group=sp_group)

        if rank % 2 == 0:
            with ms.hal.StreamCtx(stream_send):
                send_op(send_tensor)
            with ms.hal.StreamCtx(stream_recv):
                recv_tensor = recv_op(Tensor(0.0, dtype=mstype.float16))
            send_recv_ops.append(stream_send)
            send_recv_ops.append(stream_recv)
        else:
            with ms.hal.StreamCtx(stream_recv):
                recv_tensor = recv_op(Tensor(0.0, dtype=mstype.float16))
            with ms.hal.StreamCtx(stream_send):
                send_op(send_tensor)

            send_recv_ops.append(stream_recv)
            send_recv_ops.append(stream_send)
        send_recv_reqs = send_recv_ops
        return send_recv_reqs, recv_tensor


    def forward_update(self, prev_attn_out, prev_softmax_max, prev_softmax_sum,
                       cur_attn_out, cur_softmax_max, cur_softmax_sum):
        '''Updata ring attention output'''
        # update softmax_max
        softmax_max = ops.maximum(prev_softmax_max, cur_softmax_max)
        prev_scale = ops.exp(prev_softmax_max - softmax_max)
        cur_scale = ops.exp(cur_softmax_max - softmax_max)

        # update softmax_sum
        prev_softmax_sum_scaled = prev_softmax_sum * prev_scale
        cur_softmax_sum_scaled = cur_softmax_sum * cur_scale
        softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled

        # out updating scale
        prev_out_scale = prev_softmax_sum_scaled / softmax_sum
        cur_out_scale = cur_softmax_sum_scaled / softmax_sum

        # [b, n, s, 8] -> [s, b, h]
        n = prev_out_scale.shape[1]
        h = prev_attn_out.shape[-1]
        d = h // n
        prev_out_scale = prev_out_scale[..., 0].unsqueeze(3).tile((1, 1, 1, d))
        prev_out_scale = rearrange(prev_out_scale.asnumpy(), 'b n s d -> s b (n d)')
        prev_out_scale = Tensor(prev_out_scale)

        cur_out_scale = cur_out_scale[..., 0].unsqueeze(3).tile((1, 1, 1, d))
        cur_out_scale = rearrange(cur_out_scale.asnumpy(), 'b n s d -> s b (n d)')
        cur_out_scale = Tensor(cur_out_scale)

        attn_out = prev_attn_out * prev_out_scale + cur_attn_out * cur_out_scale
        return attn_out, softmax_max, softmax_sum

    def check_parameter(self, q, k, v, attn_mask, alibi_mask, prefix, padding_mask):
        '''check ring attention intput'''
        if attn_mask is not None:
            raise ValueError(f"Only attn_mask = None is supported")
        if alibi_mask is not None:
            raise ValueError(f"Only alibi_mask = None is supported")
        if prefix is not None:
            raise ValueError(f"Only prefix = None is supported")
        if padding_mask is not None:
            raise ValueError(f"Only padding_mask = None is supported")

        s1 = q.shape[0]
        s2 = k.shape[0]
        s3 = v.shape[0]
        if s2 != s3:
            raise ValueError(f"The sequence length of input k and v should be equal, but got {s2} and {s3}")

        b1 = q.shape[1]
        b2 = k.shape[1]
        b3 = v.shape[1]
        if b2 != b1:
            raise ValueError(f"The batch size of input q is not equal to k, but got batch size {b1} and {b2}")
        if b3 != b1:
            raise ValueError(f"The batch size of input q is not equal to v, but got batch_size {b1} and {b3}")

        if self.pre_tokens < s1 or self.pre_tokens < s2:
            raise ValueError(f"The pre_tokens should be larger or equal to the sequence of q and k,"
                             f"but got pre_tokens is {self.pre_tokens}, and the sequence length of q is {s1}"
                             f"and sequence length of kv is {s2}")

        if self.next_tokens < 0:
            raise ValueError(f"The next_tokens should be larger or equal to 0, but got {self.next_tokens}")

    def construct(self, q, k, v, attn_mask=None, alibi_mask=None, prefix=None, padding_mask=None):
        '''Forward of RingAttention block'''
        self.check_parameter(q, k, v, attn_mask, alibi_mask, prefix, padding_mask)
        sp_group = get_sequence_parallel_group()
        cp_size = get_sequence_parallel_world_size()
        rank = get_sequence_parallel_rank()
        send_dst = (rank + 1) % cp_size
        recv_src = (rank + cp_size - 1) % cp_size
        if attn_mask is None:
            attn_mask = ops.ones((q.shape[0], k.shape[0]), dtype=mstype.uint8)
            attn_mask = ops.triu(attn_mask, diagonal=1)

        # split chunk[i]~chunk[cp_size-i-1] into chunk[i] and chunk[cp_size-i-1],, [2s, b, h] -> [2, s, b, h]
        q, k, v = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, k, v]]

        send_kv = ops.cat((k.unsqueeze(0), v.unsqueeze(0)), axis=0) # [2, 2, s, b, h]
        recv_tensor = None
        send_recv_ops = []
        attn_out, softmax_max, softmax_sum = None, None, None
        for i in range(cp_size):
            # wait until KV is received from recv_src
            if send_recv_ops:
                for send_recv_op in send_recv_ops:
                    send_recv_op.synchronize()
                send_kv = recv_tensor
            if i < cp_size - 1:
                recv_tensor = ms.numpy.empty_like(send_kv)
                send_recv_ops, recv_tensor = self.p2p_communicate(rank, send_kv, send_dst, recv_tensor,
                                                                  recv_src, sp_group, self.stream_send,
                                                                  self.stream_recv)
            if i == 0:
                cur_k, cur_v = k, v
            else:
                cur_k, cur_v = send_kv[0], send_kv[1] # [2, s, b, h]
            # if causal:
            cur_attn_mask = None
            if i == 0:
                # [2, s, b, h] -> [2s, b, h]
                cur_attn_mask = attn_mask
                cur_q, cur_k, cur_v = [x.view(-1, *x.shape[2:]) for x in [q, cur_k, cur_v]]
            elif i <= rank:
                # [2, s, b, h] -> [2s, b, h]
                cur_q = q.view(-1, *q.shape[2:])
                # only k[0] v[0] need to be calculated
                cur_k, cur_v = [x[0] for x in [cur_k, cur_v]]
            else:
                # only q[1] need to be calculated
                cur_q = q[1]
                # [2, s, b, h] -> [2s, b, h]
                cur_k, cur_v = [x.view(-1, *x.shape[2:]) for x in [cur_k, cur_v]]
            drop_mask = None
            all_att_outs = self.flash_attention(cur_q,
                                                cur_k,
                                                cur_v,
                                                alibi_mask,
                                                drop_mask,
                                                padding_mask,
                                                cur_attn_mask,
                                                prefix)

            # if i <= rank: [2s, b, h], [b, n, 2s, 8], [b, n, 2s, 8]
            # else: [s, b, h], [b, n, s, 8], [b, n, s, 8]
            cur_attn_out = all_att_outs[3]
            cur_softmax_max = all_att_outs[0]
            cur_softmax_sum = all_att_outs[1]

            if i == 0:
                attn_out = cur_attn_out
                softmax_max = cur_softmax_max
                softmax_sum = cur_softmax_sum
            elif i <= rank:
                attn_out_updated, softmax_max_updated, softmax_sum_updated = self.forward_update(
                    attn_out, softmax_max, softmax_sum,
                    cur_attn_out, cur_softmax_max, cur_softmax_sum
                )
                attn_out, softmax_max, softmax_sum = attn_out_updated, softmax_max_updated, softmax_sum_updated
            else:
                # [2s, b, h] -> [2, s, b, h]
                attn_out = attn_out.view(2, attn_out.shape[0] // 2, *attn_out.shape[1:])
                # [b, n, 2s, 8] -> [b, n, 2, s, 8]
                softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1],
                                               2, softmax_max.shape[2] // 2, softmax_max.shape[-1])
                softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1],
                                               2, softmax_sum.shape[2] // 2, softmax_sum.shape[-1])
                attn_out_updated, softmax_max_updated, softmax_sum_updated = self.forward_update(
                    attn_out[1], softmax_max[:, :, 1, :, :], softmax_sum[:, :, 1, :, :],
                    cur_attn_out, cur_softmax_max, cur_softmax_sum
                )

                attn_out[1] = attn_out_updated.copy()
                softmax_max[:, :, 1, :, :] = softmax_max_updated.copy()
                softmax_sum[:, :, 1, :, :] = softmax_sum_updated.copy()
                # [2, s, b, h] -> [2s, b, h]
                attn_out = attn_out.view(-1, *attn_out.shape[2:])
                # [b, n, 2, s, 8] -> [b, n, 2s, 8]

                softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1], -1,
                                               softmax_max.shape[-1])
                softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1], -1,
                                               softmax_sum.shape[-1])

        return attn_out
