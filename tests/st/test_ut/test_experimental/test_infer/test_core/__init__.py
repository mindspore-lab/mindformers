# Copyright 2025 Huawei Technologies Co., Ltd
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
"""The unit testing of experimental.infer.core module"""
import mindspore.common.dtype as mstype
from mindspore import Parameter
from mindspore.common.initializer import initializer

NUM_BLOCKS = 128
BLOCK_SIZE = 64

def gen_kv_cache(config_):
    """Generate the cache of key and value."""
    kv_cache_shape = (NUM_BLOCKS, BLOCK_SIZE, config_.num_kv_heads,
                      config_.head_dim)
    key_cache = Parameter(initializer('normal', kv_cache_shape,
                                      mstype.float16),
                          name="key_cache",
                          requires_grad=False)
    value_cache = Parameter(initializer('normal', kv_cache_shape,
                                        mstype.float16),
                            name="value_cache",
                            requires_grad=False)
    return key_cache, value_cache

replacement_map = {
    'w_qkv.weight': 'linear_qkv.weight',
    'wo.weight': 'linear_proj.weight'
}

def convert_weight_name(params):
    """Convert weight name."""
    for old_name, param in list(params.items()):
        new_name = replacement_map.get(old_name, old_name)
        param.name = new_name
        if new_name != old_name:
            params.move_to_end(old_name)
            params[new_name] = params.pop(old_name)
    return params

__all__ = [
    'BLOCK_SIZE',
    'NUM_BLOCKS',
    gen_kv_cache.__name__,
    convert_weight_name.__name__,
]
