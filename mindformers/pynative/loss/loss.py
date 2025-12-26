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
"""loss function"""
from mindspore import nn, mint, ops
from mindspore.common import dtype as mstype
from mindspore import log as logger

from mindformers.tools.logger import _LogActionOnce


class _LogSoftmax(nn.Cell):
    """
    Calculate the log softmax results with given logits. The bprop of the cell is rewritten,
    just returns the accepted dout as returns. This cell should be used together with _NLLLoss,
    to optimize the bprop of the cross entropy loss.

    Inputs:
        - **logits** (Tensor) - Tensor of shape (N, C). Data type must be float16 or float32. The output logits of
          the backbone.

    Returns:
        The corresponding log softmax results.
    """
    def __init__(self):
        super().__init__()
        self.sub = mint.sub
        self.max = mint.max
        self.sum = mint.sum
        self.exp = mint.exp
        self.log = mint.log
        self.cast = ops.cast

    def construct(self, logits):
        """Forward process"""
        logits = self.cast(logits, mstype.float32)
        logit_max, _ = self.max(logits, 1, True)
        logit_sub = self.sub(logits, logit_max)
        logit_exp = self.exp(logit_sub)
        exp_sum = self.sum(logit_exp, -1, True)
        log_exp_sum = self.log(exp_sum)
        logit_neg_logsoftmax = self.sub(log_exp_sum, logit_sub)
        return logit_neg_logsoftmax

    def bprop(self, logits, _, dout):
        """just return the loss of the dout. Note this should be used together with _NLLLoss"""
        return self.cast(dout, logits.dtype)


class _NLLLoss(nn.Cell):
    """
    Calculate the NLLLoss results with given log softmax results and the label. The bprop of the cell is rewritten.
    This cell should be used together with _LogSoftmax, to optimize the bprop of the cross entropy loss.

    Inputs:
        - **logit_neg_logsoftmax** (Tensor) - Tensor of shape (N, C). Data type is float32.
        - **label** (Tensor) - Tensor of shape (N, C). The ground truth label.

    Returns:
        The corresponding loss results.
    """
    def __init__(self):
        super().__init__()
        self.gather = mint.gather
        self.unsqueeze = mint.unsqueeze
        self.exp = mint.exp
        self.neg = mint.neg
        self.zeros_like = mint.zeros_like
        self.reshape = mint.reshape
        self.scatter_add = mint.scatter_add

    def construct(self, logit_neg_logsoftmax, label):
        """Forward process"""
        vocab_indices = self.reshape(label, (-1, 1))
        loss = self.reshape(self.gather(logit_neg_logsoftmax, 1, vocab_indices), (-1,))
        return loss

    def bprop(self, logit_neg_logsoftmax, label, _, dout):
        """A simplified function. Note this should be used together with _LogSoftmax"""
        vocab_indices = self.reshape(label, (-1, 1))
        logits_softmax = self.exp(self.neg(logit_neg_logsoftmax))
        neg_ones = self.zeros_like(vocab_indices, dtype=logits_softmax.dtype) - 1
        grad = self.scatter_add(logits_softmax, 1, vocab_indices, neg_ones)
        grad = grad * self.unsqueeze(dout, -1)
        return grad, self.zeros_like(label)


class CrossEntropyLoss(nn.Cell):
    """
    Calculate the cross entropy loss.

    CrossEntropyLoss supports two different types of targets:

    - Class indices (int), where the range of values is :math:`[0, C)` with :math:`C` being the number of classes.
      When reduction is set to 'none', the cross-entropy loss is computed as follows:

      .. math::
          \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
          l_n = - w_{y_n} \\log \\frac{\\exp(x_{n,y_n})}{\\sum_{c=1}^C \\exp(x_{n,c})}
          \\cdot \\mathbb{1}\\{y_n \\not= \\text{ignore_index}\\}

      where :math:`x` denotes the predicted values, :math:`t` denotes the target values, :math:`w` denotes the weights,
      and :math:`N` is the batch size. The index :math:`c` ranges from [0, C-1], representing the class indices,
      where :math:`C` is the number of classes.

      If reduction is not set to 'none' (the default is 'mean'), the loss is computed as:

      .. math::
          \\ell(x, y) = \\begin{cases}
              \\sum_{n=1}^N \\frac{1}{\\sum_{n=1}^N w_{y_n} \\cdot \\mathbb{1}\\{y_n \\not= \\text{ignore_index}\\}}
              l_n, &
              \\text{if reduction} = \\text{'mean',}\\\\
              \\sum_{n=1}^N l_n,  &
              \\text{if reduction} = \\text{'sum'.}
              \\end{cases}

    - Class probabilities (float), used when the target is a probability distribution over multiple class labels.
      When reduction is set to 'none', the cross-entropy loss is computed as follows:

      .. math::
          \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
          l_n = - \\sum_{c=1}^C w_c \\log \\frac{\\exp(x_{n,c})}{\\sum_{i=1}^C \\exp(x_{n,i})} y_{n,c}

      where :math:`x` denotes the predicted values, :math:`t` denotes the target values, :math:`w` denotes the weights,
      and :math:`N` is the batch size. The index :math:`c` ranges from [0, C-1], representing the class indices,
      where :math:`C` is the number of classes.

      If reduction is not set to 'none' (the default is 'mean'), the loss is computed as:

      .. math::
          \\ell(x, y) = \\begin{cases}
              \\frac{\\sum_{n=1}^N l_n}{N}, &
              \\text{if reduction} = \\text{'mean',}\\
              \\sum_{n=1}^N l_n,  &
              \\text{if reduction} = \\text{'sum'.}
              \\end{cases}

    Args:
        calculate_per_token_loss (bool): Whether to calculate the loss of each token. Default: ``False``.
        loss_tag (str, optional): Distinguish different types of loss. Default: 'lm'.

    Inputs:
        - **logits** (Tensor) - Tensor of shape (N, C). Data type must be float16 or float32. The output logits of
          the backbone.

        - **label** (Tensor) - Tensor of shape (N, ). The ground truth label of the sample.

        - **input_mask** (Tensor) - Tensor of shape (N, ). input_mask indicates whether there are padded inputs and for
          padded inputs it will not be counted into loss.

    Returns:
        Tensor, the computed cross entropy loss value.

    Examples:
        >>> import numpy as np
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor
        >>> from mindformers.pynative.loss import CrossEntropyLoss
        >>> loss = CrossEntropyLoss()
        >>> logits = Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]), mstype.float32)
        >>> labels_np = np.array([1]).astype(np.int32)
        >>> input_mask = Tensor(np.ones(1).astype(np.float32))
        >>> labels = Tensor(labels_np)
        >>> output = loss(logits, labels, input_mask)
        >>> output.shape
        (1,)
    """
    @_LogActionOnce(m_logger=logger, key='CrossEntropyLoss')
    def __init__(self, calculate_per_token_loss=False, loss_tag='lm', **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.loss_tag = loss_tag
        self.sum = mint.sum
        self.mul = mint.mul
        self.add = mint.add
        self.div = mint.div
        self.relu = mint.nn.functional.relu
        self.reshape = mint.reshape
        self.cast = ops.cast
        self.tuple_to_array = ops.tuple_to_array

        self._log_softmax = _LogSoftmax()
        self._nllloss = _NLLLoss()
        self.calculate_per_token_loss = calculate_per_token_loss

    def construct(self, logits, label, input_mask):
        """Forward process"""
        log_softmax = self._log_softmax(logits)
        loss_reduce = self._nllloss(log_softmax, label)

        # Using input_mask to mask the loss
        input_mask = self.reshape(input_mask, (-1,))
        input_mask = self.cast(input_mask, mstype.float32)
        numerator = self.sum(self.mul(loss_reduce, input_mask))
        denominator = self.add(
            self.sum(input_mask),
            self.cast(self.tuple_to_array((1e-8,)),mstype.float32))
        if not self.calculate_per_token_loss:
            return self.div(numerator, denominator)
        return numerator, denominator
