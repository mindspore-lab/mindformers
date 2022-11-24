# Copyright 2020 Huawei Technologies Co., Ltd
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

"""Operations for clipping tensors to min/max values."""
from mindspore._checkparam import Rel
from mindspore._checkparam import Validator as validator
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.primitive import constexpr

# The attribute grad_scale is needed for enabling the context mode
# If this is removed, c.clip_by_global_norm will have precision error in semi/auto context mode.
expand_dims = P.ExpandDims().add_prim_attr("grad_scale", True)
apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")
get_square_sum = C.MultitypeFuncGraph("get_square_sum")


@get_square_sum.register("Tensor")
def _get_square_sum(x):
    norm = P.ReduceSum(False)(F.square(x), ())
    norm = expand_dims(F.cast(norm, mstype.float32), 0)
    return norm


@apply_global_norm.register("Tensor", "Tensor", "Tensor")
def _apply_global_norm(clip_norm, global_norm, x):
    x_dtype = F.dtype(x)
    clip_coef = clip_norm / (global_norm + 1e-6)
    if clip_coef < 1:
        x = x * clip_coef
    x = F.cast(x, x_dtype)
    return x


class _ClipByGlobalNorm(Cell):
    r"""
    Clips tensor values by the ratio of the sum of their norms.

    Args:
        clip_norm (Union(float, int)): The clipping ratio. Default: 1.0
        use_norm (Union(float, None)): The global norm. Default: None

    Inputs:
        - **x** (Union(tuple[Tensor], list[Tensor])) - Input dataset to clip.

    Outputs:
        Tensor, a clipped Tensor.
    """

    def __init__(self, clip_norm=1.0, use_norm=None):
        """Initialize _ClipByGlobalNorm."""
        super(_ClipByGlobalNorm, self).__init__()
        # Add interface. This parameter is not used at present
        if use_norm is not None:
            raise ValueError(f"For '{self.cls_name}', input 'use_norm' only supports None currently, "
                             f"but got 'use_norm': {use_norm}")
        validator.check_number("clip_norm", clip_norm, 0.0, Rel.GT, self.cls_name)
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.hyper_map = C.HyperMap()
        self.greater_equal = P.GreaterEqual()

    def construct(self, x):
        square_sum = self.hyper_map(get_square_sum, x)
        global_norm = F.sqrt(F.addn(square_sum))
        if global_norm > self.clip_norm:
            print("Global Norm is greater than Max Clip Norm:", global_norm)
        clip_x = self.hyper_map(F.partial(apply_global_norm, self.clip_norm, global_norm), x)
        return clip_x


@constexpr
def _check_value(clip_norm):
    validator.check_number("clip_norm", clip_norm, 0.0, Rel.GT, "clip_by_global_norm")
    return clip_norm


def clip_by_global_norm(x, clip_norm=1.0, use_norm=None):
    r"""
    Clips tensor values by the ratio of the sum of their norms.
    Args:
        x (Union(tuple[Tensor], list[Tensor])): Input dataset to clip.
          The shape of each Tensor in tuple is :math:`(N,*)` where :math:`*` means,
          any number of additional dimensions.
        clip_norm (Union(float, int)): The clipping ratio, it should be greater than 0. Default: 1.0
        use_norm (None): The global norm. Default: None. Currently only none is supported.

    Returns:
        tuple[Tensor], a clipped Tensor. It has the same dataset type as `x` and each Tensor in the output tuple is the
        same as the original input shape.
    """

    clip_norm = _check_value(clip_norm)
    clip_val = _ClipByGlobalNorm(clip_norm, use_norm)(x)
    return clip_val
