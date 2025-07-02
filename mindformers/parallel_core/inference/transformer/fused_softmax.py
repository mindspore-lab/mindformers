# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""ScaleMaskSoftmax"""
from mindspore import mint, ops, nn
from mindspore.common import dtype as mstype


class FusedScaleMaskSoftmax(nn.Cell):
    r"""
    fused operation: scaling + mask + softmax

    Args:
        input_in_fp16: flag to indicate if input in fp16 data format.
        input_in_bf16: flag to indicate if input in bf16 data format.
        attn_mask_type: attention mask type (pad or causal)
        scaled_masked_softmax_fusion: flag to indicate user want to use softmax fusion
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.

    Inputs:
        - **x** (Tensor) - The input tensor
        - **mask** (Tensor) - The mask tensor

    Outputs:
        - The output tensor.
    """

    def __init__(self, mask_func, scale=None, softmax_compute_type=mstype.float32):
        super().__init__()
        self.mask_func = mask_func
        self.softmax_compute_type = softmax_compute_type
        self.scale = scale

        if self.scale is not None and self.softmax_compute_type != mstype.float32:
            raise ValueError("softmax should be in fp32 when scaled")

    def construct(self, x, mask):
        """construct method"""
        origin_dtype = x.dtype
        if self.softmax_compute_type != origin_dtype:
            x = ops.cast(x, self.softmax_compute_type)

        if self.scale is not None:
            x = x * self.scale
        masked_input = self.mask_func(x, mask) if mask is not None else x

        probs = mint.nn.functional.softmax(masked_input, dim=-1)

        if self.softmax_compute_type != origin_dtype:
            probs = ops.cast(probs, origin_dtype)

        return probs
