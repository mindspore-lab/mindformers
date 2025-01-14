# Copyright 20244 Huawei Technologies Co., Ltd
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

"""KV Cache Attention Manager for inference."""
import mindspore.common.dtype as mstype
from mindspore import nn, Parameter, ops
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import Zero
from mindspore.ops.auto_generate import KVCacheScatterUpdate, ReshapeAndCache

from mindformers.tools.utils import get_infer_boost


class KVCacheMgr(nn.Cell):
    """KV Cache Manager."""

    def __init__(self,
                 n_kv_head,
                 head_dim,
                 num_blocks=1024,
                 block_size=128,
                 batch_size=32,
                 seq_length=4096,
                 compute_dtype=mstype.float16):
        super().__init__()
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.enable_infer_boost = get_infer_boost()
        print("enable_infer_boost-----",self.enable_infer_boost)
        if self.enable_infer_boost:
            kv_shape = (self.num_blocks, self.block_size, self.n_kv_head, self.head_dim)
            self.reshape_and_cache = ReshapeAndCache()
        else:
            kv_shape = (self.batch_size, self.n_kv_head, self.seq_length, self.head_dim)
            self.kv_cache_scatter_update = KVCacheScatterUpdate()

        self.key_cache = Parameter(Tensor(shape=kv_shape, dtype=compute_dtype, init=Zero()), name="key_cache",
                                   requires_grad=False)
        self.value_cache = Parameter(Tensor(shape=kv_shape, dtype=compute_dtype, init=Zero()), name="value_cache",
                                     requires_grad=False)
        print("KVCacheMgr--finish--------------------")

    def construct(self, key_update, value_update, slot_mapping=None, batch_valid_length=None):
        """The forward compute of KVCache for Attention."""
        key_cache = self.key_cache
        value_cache = self.value_cache
        if self.enable_infer_boost:
            self.reshape_and_cache(key_update, value_update, self.key_cache, self.value_cache, slot_mapping)
        else:
            # update shape: [real_bs, n_head, max_seqlen, head_dim]
            self.kv_cache_scatter_update(self.key_cache, batch_valid_length, key_update, -2, 'update')
            self.kv_cache_scatter_update(self.value_cache, batch_valid_length, value_update, -2, 'update')
        key_cache = ops.depend(key_cache, key_update)
        value_cache = ops.depend(value_cache, value_update)
        return key_cache, value_cache

    def shard(self, parallel_config):
        """The shard strategy."""
        dp = 1 if parallel_config is None else parallel_config.data_parallel
        mp = 1 if parallel_config is None else parallel_config.model_parallel
        if self.enable_infer_boost:
            self.reshape_and_cache.shard(((dp, 1, mp), (dp, 1, mp), (1, 1, mp, 1), (1, 1, mp, 1), (1,)))
        else:
            self.kv_cache_scatter_update.shard(((dp, mp, 1, 1), (1,), (dp, mp, 1, 1)))
