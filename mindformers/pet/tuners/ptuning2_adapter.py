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
"""
p-tuning-v2 adapter
https://arxiv.org/pdf/2110.07602.pdf
"""

from mindspore.ops import operations as P


class Ptuning2Adapter:
    """
    Ptuning2Adapter is the adapter to modify the pretrained model, which uses p-tuning-v2.
    """

    @staticmethod
    def add_prefix(prefix_key_value, key, value, seq_len_dim=2):
        """
        Add p-tuning v2 prefix for key, vale.
        """

        if prefix_key_value is not None:
            prefix_key = prefix_key_value[0]
            prefix_value = prefix_key_value[1]
            cat = P.Concat(seq_len_dim)
            prefix_key = P.Cast()(prefix_key, key.dtype)
            key = cat([prefix_key, key])
            prefix_value = P.Cast()(prefix_value, value.dtype)
            value = cat([prefix_value, value])

        return key, value
