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
import os
import pytest

import numpy as np
from mindspore import dtype as mstype
from mindspore import Tensor
from mindformers.modules import PagedAttentionMgr


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_paged_attention_mgr():
    """
    Feature: Test the paged attention.
    Description: Test the forward
    Expectation: No exception
    """
    os.environ['GRAPH_OP_RUN'] = '1'
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    bsz, head_num, seq_len, head_dim = 1, 16, 4096, 128
    n_kv_head = 16
    block_size = 1024
    num_blocks = 16
    compute_dtype = mstype.float16
    hidden_size = head_num * head_dim
    batch_valid_length = Tensor(np.ones((bsz, 1)), mstype.int32)
    block_tables = Tensor(np.ones((bsz, num_blocks)), mstype.int64)
    query = Tensor(np.ones((bsz, seq_len, hidden_size)), mstype.float16)
    key = Tensor(np.ones((bsz, seq_len, hidden_size)), mstype.float16)
    value = Tensor(np.ones((bsz, seq_len, hidden_size)), mstype.float16)
    slot_mapping = Tensor(np.ones((bsz,)), mstype.int32)
    paged_attention_mgr = PagedAttentionMgr(head_num,
                                            head_dim,
                                            n_kv_heads=n_kv_head,
                                            block_size=block_size,
                                            num_blocks=num_blocks,
                                            compute_dtype=compute_dtype)
    paged_attention_mgr(key, value, slot_mapping)

    context_layer = paged_attention_mgr.paged_attn(query, batch_valid_length, block_tables)
    assert context_layer.shape == (1, 4096, 2048)
