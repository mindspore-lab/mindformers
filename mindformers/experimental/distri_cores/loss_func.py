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
"""loss function"""
import mindspore as ms
from mindspore import nn, mint
from mindspore import ops as P
from mindspore.ops import ReduceOp
from mindspore.common.api import _pynative_executor
from mindformers.experimental.distri_cores.create_comm import get_tp_group, get_tp_rank, get_tp_world_size

from mindformers.experimental.distri_cores.register import ModuleType, ModuleRegistry

__all__ = ['VocabParallelCrossEntropy']


class VocabParallelCrossEntropy(nn.Cell):
    """
    Calculate the paralleled cross entropy loss.

    Inputs:
        - **vocab_parallel_logits** (Tensor) - Tensor of shape (N, C). Data type must be float16 or float32.
          The output logits of the backbone.

        - **target** (Tensor) - Tensor of shape (N, ). The ground truth label of the sample.

        - **label_smoothing** (Float) - smoothing factor, must be in range[0.0, 1.0).

    Returns:
        The corresponding cross entropy loss.

    Examples:
        >>> import numpy as np
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor
        >>> from mindformers.experimental.distri_cores import VocabParallelCrossEntropy
        >>> loss = VocabParallelCrossEntropy()
        >>> logits = Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]), mstype.float32)
        >>> labels_np = np.array([1]).astype(np.int32)
        >>> labels = Tensor(labels_np)
        >>> output = loss(logits, labels)
        >>> output.shape
        (1,)
    """
    # pylint: disable=W0613
    def __init__(self, *args, **kwargs):
        super(VocabParallelCrossEntropy, self).__init__()
        self.label_smoothing = None
        self.vocab_size = None
        self.saved_tensors = [[], [], []]
        self.tp_world_size = get_tp_world_size()

        if self.tp_world_size > 1:
            self.all_reduce_max = P.AllReduce(op=ReduceOp.MAX, group=get_tp_group())
            self.all_reduce_sum = P.AllReduce(op=ReduceOp.SUM, group=get_tp_group())

    def construct(self, vocab_parallel_logits, target, label_smoothing=0.0):
        """Forward process"""
        logits_max = mint.max(vocab_parallel_logits, dim=-1)[0]
        if self.tp_world_size > 1:
            logits_max = self.all_reduce_max(logits_max)
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)

        partition_vocab_size = vocab_parallel_logits.shape[-1]
        vocab_start_index = get_tp_rank() * partition_vocab_size
        vocab_end_index = vocab_start_index + partition_vocab_size

        left = target < vocab_start_index
        right = target >= vocab_end_index
        target_mask = left.astype(ms.int32) | right.astype(ms.int32)
        target_mask = target_mask.astype(ms.bool_)
        masked_target = target - vocab_start_index
        masked_target[target_mask] = 0

        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = mint.arange(start=0, end=logits_2d.shape[0], step=1)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        if self.tp_world_size > 1:
            predicted_logits = self.all_reduce_sum(predicted_logits)

        exp_logits = mint.exp(vocab_parallel_logits)
        sum_exp_logits = exp_logits.sum(axis=-1)
        if self.tp_world_size > 1:
            sum_exp_logits = self.all_reduce_sum(sum_exp_logits)

        loss = mint.log(sum_exp_logits) - predicted_logits

        exp_logits = exp_logits.div(sum_exp_logits.unsqueeze(dim=-1))

        vocab_size = exp_logits.shape[-1]
        if label_smoothing > 0.0:
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            log_probs = mint.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        if _pynative_executor.grad_flag():
            self.saved_tensors[0].append(exp_logits)
            self.saved_tensors[1].append(target_mask)
            self.saved_tensors[2].append(masked_target_1d)
        return loss

    # pylint: disable=W0613, C0111
    def bprop(self, vocab_parallel_logits, target, label_smoothing, grad_output):
        softmax, target_mask, masked_target_1d = self.saved_tensors[0].pop(0),\
            self.saved_tensors[1].pop(0), self.saved_tensors[2].pop(0)
        partition_vocab_size = softmax.shape[-1]
        grad_2d = softmax.view(-1, partition_vocab_size)
        arange_1d = mint.arange(start=0, end=grad_2d.shape[0], step=1)
        target_mask = target_mask.astype(ms.float32)
        softmax_update = 1.0 - target_mask.view(-1)
        if self.label_smoothing > 0.0:
            smoothing = self.label_smoothing * self.vocab_size / (self.vocab_size - 1)
            grad_2d[arange_1d, masked_target_1d] -= (1.0 - smoothing) * softmax_update
            average_grad = 1 / self.vocab_size
            grad_2d[arange_1d, :] -= smoothing * average_grad
        else:
            grad_2d[arange_1d, masked_target_1d] -= softmax_update

        grad_2d = grad_2d.reshape(softmax.shape)
        grad_input = grad_2d.mul(grad_output.unsqueeze(dim=-1))

        self.saved_tensors = [[], [], []]

        return grad_input, None, None


ModuleRegistry.register(nn.CrossEntropyLoss, ModuleType.LOSS_FUNC)
ModuleRegistry.register(VocabParallelCrossEntropy, ModuleType.LOSS_FUNC)


class LossWithMask(nn.Cell):
    """
    Calculate the loss with mask and mean reduction.

    Args:
        - **loss_func** (Function) - Loss function.

    Inputs:
        - **logits** (Tensor) - Tensor of shape (N, C). Data type must be float16 or float32. The output logits of
          the backbone.

        - **label** (Tensor) - Tensor of shape (N, ). The ground truth label of the sample.

        - **input_mask** (Tensor) - Tensor of shape (N, ). input_mask indicates whether there are padded inputs and for
          padded inputs it will not be counted into loss.

    Returns:
        The corresponding cross entropy loss.

    Examples:
        >>> import numpy as np
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor, nn
        >>> loss = LossWithMask(nn.CrossEntropyLoss())
        >>> logits = Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]), mstype.float32)
        >>> labels_np = np.array([1]).astype(np.int32)
        >>> input_mask = Tensor(np.ones(1).astype(np.float32))
        >>> labels = Tensor(labels_np)
        >>> output = loss(logits, labels, input_mask)
        >>> output.shape
        (1,)
    """
    # pylint: disable=W0613
    def __init__(self, loss_func, *args, **kwargs):
        super(LossWithMask, self).__init__()
        self.loss_func = loss_func

    def construct(self, logits, label, input_mask):
        loss_reduce = self.loss_func(logits, label)
        input_mask = input_mask.view(-1)
        loss = mint.sum(loss_reduce * input_mask) / input_mask.sum()
        return loss


def get_loss_func(training_config, return_instance: bool = True, **kwargs):
    """
    Get the loss function based on the provided loss function configuration.

    Args:
        training_config (TrainingConfig): The configuration object for training.

    Returns:
        loss_fn (callable): The loss function based on the provided configuration.

    Raises:
        ValueError: If the specified loss function type is not supported.
    """
    loss_func_kwargs = training_config.loss_func_kwargs
    loss_func_kwargs["reduction"] = training_config.loss_reduction
    loss_func_type = loss_func_kwargs['loss_func_type']
    if "CrossEntropy" in loss_func_type:
        loss_func_kwargs["reduction"] = 'none'
    loss_func_cls = ModuleRegistry.get_item(module_type=ModuleType.LOSS_FUNC, item_name=loss_func_type)
    if return_instance:
        loss_func_kwargs.update(kwargs)
        loss_func_kwargs = ModuleRegistry.get_needed_params_for_init(loss_func_cls, loss_func_kwargs)
        return LossWithMask(loss_func=loss_func_cls(**loss_func_kwargs))
    return loss_func_cls
