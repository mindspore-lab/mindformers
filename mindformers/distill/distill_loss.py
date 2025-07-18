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
"""Distill loss for llm model."""
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.ops import functional as F
from mindspore.ops import auto_generate as aclnn_ops


class LogitsKLLoss(nn.Cell):
    """
    KL-Divergence loss on output logits.

    Args:
        temperature: A value used to soften the logits_t and logits_s before computing loss on them. Default: 1.0.

    Inputs:
        logits_s (Tensor): Student's logits, treated as prediction.
        logits_t (Tensor): Teacher's logits, treated as label.
        input_mask (Tensor): The logits mask.

    Returns:
        The KL loss between logits_s and logits_t.
    """

    def __init__(self, temperature: float = 1.0):
        super(LogitsKLLoss, self).__init__()
        self._temperature = temperature
        self.softmax = aclnn_ops.Softmax(axis=-1)
        self.log_softmax = aclnn_ops.LogSoftmax(axis=-1)
        self.sum = aclnn_ops.SumExt()
        self.sum2 = aclnn_ops.SumExt().shard(((1,),))
        self.mul2 = aclnn_ops.Mul().shard(((1,), (1,)))
        self.add2 = aclnn_ops.AddExt()
        self.div2 = aclnn_ops.Div()
        self.cast = aclnn_ops.Cast()
        self.reshape = aclnn_ops.Reshape()

    def construct(self, logits_s, logits_t, input_mask):
        """Compute KD loss on student and teacher logits."""
        soft_log_probs = self.log_softmax(logits_s / self._temperature)
        soft_targets = self.softmax(logits_t / self._temperature)

        kd_loss = F.kl_div(soft_log_probs, soft_targets, "none")
        kd_loss = self.sum(kd_loss, -1)
        # Since the magnitudes of the gradients produced by the soft logits scale as 1/(T^2),
        # multiplying them by T^2 ensures that the relative contributions of the logits
        # remain roughly unchanged while experimenting with meta-parameters.
        kd_loss = kd_loss * self._temperature ** 2

        # Using input_mask to mask the loss
        kd_loss = self.reshape(kd_loss, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        numerator = self.sum2(self.mul2(kd_loss, input_mask))
        denominator = self.add2(
            self.sum2(input_mask),
            self.cast(F.tuple_to_array((1e-8,)), mstype.float32))
        return self.div2(numerator, denominator)


class LossBalancer(nn.Cell):
    """
    Static weights-based loss aggregation of KD losses.

    Args:
        kd_loss_scale: The static weight to be applied to balance the knowledge distillation. Default: 1.0.
        skip_original_loss: Whether to ignore the original loss of model. Default: True.
    """

    def __init__(self, kd_loss_scale: float = 1.0, skip_original_loss: bool = True):
        super(LossBalancer, self).__init__()
        self._kd_loss_scale = kd_loss_scale
        self._skip_original_loss = skip_original_loss

    def construct(self, original_loss, logits_loss):
        """Compute aggregate loss"""
        if self._skip_original_loss:
            return logits_loss
        kd_loss = logits_loss * float(original_loss.item() / logits_loss.item())
        return original_loss + kd_loss * self._kd_loss_scale
