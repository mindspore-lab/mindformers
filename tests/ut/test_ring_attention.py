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
"""Test Ring Attention."""
import numpy as np
import pytest
from mindformers.experimental.distri_cores.sequence_parallel.utils import init_sp_group, get_sp_chuncks
from mindformers.experimental.distri_cores.sequence_parallel.ring_attention import RingAttention

import mindspore as ms
from mindspore.communication import init
from mindspore.common import dtype as mstype
from mindspore import ops
from mindspore.common.tensor import Tensor
from mindspore.ops.operations.nn_ops import FlashAttentionScore

def generate_inputs(b, n1, n2, s1, s2, d1, input_layout, dtype, return_tensor=True):
    '''generate inputs'''
    min_value = -1
    max_value = 1
    np.random.seed(42)
    if input_layout == "BSH":
        query = np.random.uniform(min_value, max_value, [b, s1, n1 * d1])
        key = np.random.uniform(min_value, max_value, [b, s2, n2 * d1])
        value = np.random.uniform(min_value, max_value, [b, s2, n2 * d1])
    elif input_layout == "BNSD":
        query = np.random.uniform(min_value, max_value, [b, n1, s1, d1])
        key = np.random.uniform(min_value, max_value, [b, n2, s2, d1])
        value = np.random.uniform(min_value, max_value, [b, n2, s2, d1])
    elif input_layout == "SBH":
        query = np.random.uniform(min_value, max_value, [s1, b, n1 * d1])
        key = np.random.uniform(min_value, max_value, [s2, b, n2 * d1])
        value = np.random.uniform(min_value, max_value, [s2, b, n2 * d1])
    elif input_layout == "BSND":
        query = np.random.uniform(min_value, max_value, [b, s1, n1, d1])
        key = np.random.uniform(min_value, max_value, [b, s2, n2, d1])
        value = np.random.uniform(min_value, max_value, [b, s2, n2, d1])
    elif input_layout == "TND":
        query = np.random.uniform(min_value, max_value, [b * s1, n1, d1])
        key = np.random.uniform(min_value, max_value, [b * s2, n2, d1])
        value = np.random.uniform(min_value, max_value, [b * s2, n2, d1])
    else:
        raise ValueError(f"input_layout is invalid.")
    alibi_mask = None
    prefix = None
    drop_mask = None
    attn_mask = None
    padding_mask = None
    if return_tensor:
        return Tensor(query, dtype=dtype), Tensor(key, dtype=dtype), Tensor(value, dtype=dtype), alibi_mask, \
               drop_mask, padding_mask, attn_mask, prefix
    return query, key, value, alibi_mask, drop_mask, padding_mask, attn_mask, prefix


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ring_attention():
    """
    Feature: Test RingAttention.
    Description: Test RingAttention functional.
    Expectation: Success.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
    init()

    # Init parameter
    dp = 2
    sp = 4
    init_sp_group(sp)

    bs = 16
    n = 8
    s = 4096
    hidden_size = 16
    query_output, key_output, value_output, alibi_mask_output, drop_mask_output, \
        padding_mask_output, ring_attn_mask_output, prefix_out = generate_inputs(bs, n, n, s, s,
                                                                                 hidden_size, "SBH", mstype.float16)

    q2 = get_sp_chuncks(query_output, dp, sp)
    k2 = get_sp_chuncks(key_output, dp, sp)
    v2 = get_sp_chuncks(value_output, dp, sp)

    ring_attention = RingAttention(head_num=n,
                                   pre_tokens=65535,
                                   next_tokens=0,
                                   keep_prob=1.,
                                   input_layout="SBH",
                                   dp=dp,
                                   sp=sp)
    ring_attention_output = ring_attention(q2, k2, v2, ring_attn_mask_output, alibi_mask_output, prefix_out,
                                           padding_mask_output)

    flash_attn_mask = ops.ones((query_output.shape[0], key_output.shape[0]), dtype=mstype.uint8)
    flash_attn_mask = ops.triu(flash_attn_mask, diagonal=1)
    flash_attention = FlashAttentionScore(head_num=n,
                                          pre_tokens=65535,
                                          next_tokens=0,
                                          keep_prob=1.,
                                          input_layout="SBH")
    _, _, _, flash_attention_output = flash_attention(query_output,
                                                      key_output,
                                                      value_output,
                                                      alibi_mask_output,
                                                      drop_mask_output,
                                                      padding_mask_output,
                                                      flash_attn_mask)

    flash_attention_output = get_sp_chuncks(flash_attention_output, dp, sp)
    assert np.allclose(flash_attention_output.asnumpy(), ring_attention_output.asnumpy(), 0.004, 0.004)
    print("end test.")
