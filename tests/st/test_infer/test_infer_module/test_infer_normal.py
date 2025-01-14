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
""" test infer attention"""
import math
import os
import pytest

import numpy as np
import mindspore as ms
from mindspore import dtype as mstype
from mindspore import Tensor
from mindformers.modules import KVCacheMgr
from mindformers.modules.infer_attention import InferAttention
from mindformers.modules.layers import FreqsMgr


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_kv_cache_mgr():
    """
    Feature: Test the kv cache manager.
    Description: Test the forward
    Expectation: No exception
    """
    os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    jit_level = "O0"
    infer_boost = "off"
    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": jit_level, "infer_boost": infer_boost})

    bsz, n_kv_head, seq_len, head_dim = 8, 16, 4096, 128
    compute_dtype = mstype.float16

    key = Tensor(np.ones((bsz, n_kv_head, seq_len, head_dim)), mstype.float16)
    value = Tensor(np.ones((bsz, n_kv_head, seq_len, head_dim)), mstype.float16)
    batch_valid_length = Tensor(np.zeros((bsz, 2)), mstype.int32)

    kv_shape = (bsz, n_kv_head, seq_len, head_dim)
    kv_cache_mgr = KVCacheMgr(n_kv_head,
                              head_dim,
                              batch_size=bsz,
                              seq_length=seq_len,
                              compute_dtype=compute_dtype)
    key_cache, value_cache = kv_cache_mgr(key, value, None, batch_valid_length)
    assert key_cache.shape == kv_shape and value_cache.shape == kv_shape


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_infer_attention():
    """
    Feature: Test the infer attention.
    Description: Test the forward
    Expectation: No exception
    """
    os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    jit_level = "O0"
    infer_boost = "off"
    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": jit_level, "infer_boost": infer_boost})

    bsz, head_num, seq_len, head_dim = 8, 80, 256, 128
    n_kv_head = 8
    hidden_size = head_num * head_dim
    query = Tensor(np.ones((bsz, seq_len, hidden_size)), mstype.float16)
    key = Tensor(np.ones((bsz, seq_len, n_kv_head * head_dim)), mstype.float16)
    value = Tensor(np.ones((bsz, seq_len, n_kv_head * head_dim)), mstype.float16)
    batch_valid_length = Tensor(np.zeros((bsz, 2)), mstype.int32)
    attn_mask = Tensor(np.ones((bsz, 1, seq_len, seq_len)), mstype.uint8)

    freqs_mgr = FreqsMgr(head_dim=head_dim, seq_length=seq_len, max_position_embedding=seq_len)
    freqs_cis = freqs_mgr(seq_len)
    infer_attention = InferAttention(head_num,
                                     head_dim,
                                     n_kv_head,
                                     scale_value=1. / math.sqrt(head_dim),
                                     pre_tokens=65536,
                                     next_tokens=0,
                                     batch_size=bsz,
                                     seq_length=seq_len,
                                     compute_dtype=mstype.float16)

    output = infer_attention(query, key, value, batch_valid_length, None, None, freqs_cis, attn_mask)
    assert output.shape == (bsz, seq_len, hidden_size)
