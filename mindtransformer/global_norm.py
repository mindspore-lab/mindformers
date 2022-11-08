# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Global Norm utilities."""

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops.composite.clip_ops import get_square_sum

apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")
apply_clip_norm = C.MultitypeFuncGraph("apply_clip_norm")

@apply_global_norm.register("Tensor", "Tensor", "Tensor")
def _apply_global_norm(clip_norm, global_norm, grad):
    return grad * clip_norm / global_norm

@apply_clip_norm.register("Tensor", "Tensor")
def _apply_clip_norm(clip_norm, grad):
    return P.Cast()(grad * clip_norm, mstype.float16)

class GlobalNorm(nn.Cell):
    """
    Calculate the global norm value of given tensors
    """

    def __init__(self):
        super(GlobalNorm, self).__init__()
        self.norm = nn.Norm()
        self.hyper_map = C.HyperMap()

    def construct(self, grads):
        """Calculate global norm construct"""
        square_sum = self.hyper_map(get_square_sum, grads)
        global_norms = F.sqrt(F.addn(square_sum))
        return grads, global_norms

class ClipByGlobalNorm(nn.Cell):
    """

    Clip grads by global norm

    """

    def __init__(self, enable_offload, clip_norm=1.0):
        super(ClipByGlobalNorm, self).__init__()
        self.global_norm = GlobalNorm()
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.hyper_map = C.HyperMap()
        self.enable_offload = enable_offload

    def construct(self, grads):
        """Clip grads by global norm construct"""
        grads, global_norm_value = self.global_norm(grads)
        cond = P.GreaterEqual()(global_norm_value, self.clip_norm)
        global_norm = F.select(cond, global_norm_value, self.clip_norm)
        if self.enable_offload:
            grads = self.hyper_map(F.partial(apply_global_norm, self.clip_norm, global_norm), grads)
        else:
            grads = self.hyper_map(F.partial(apply_clip_norm, self.clip_norm), grads)
        return grads, global_norm
