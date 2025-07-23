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
"""MindFormer Self-Define Loss."""
import os

from mindspore import nn, Tensor, get_auto_parallel_context
from mindspore import ops as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype

from mindspore.context import ParallelMode
from mindspore.parallel import set_algo_parameters

from mindspore import log as logger
from mindspore.parallel._utils import _get_device_num, _get_pipeline_stages, _get_parallel_mode

from mindformers.version_control import is_dump_supported
from mindformers.tools.logger import _LogActionOnce
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.utils import get_real_rank
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config
from mindformers.parallel_core.training_graph.loss_func import get_device_local_loss

__all__ = ['CrossEntropyLoss']


class _LogSoftmax(nn.Cell):
    """
    Calculate the log softmax results with given logits. The bprop of the cell is rewritten,
    just returns the accepted dout as returns. This cell should be used together with _NLLoss,
    to optimize the bprop of the cross entroy loss.

    Args:
        parallel_config (OpParallelConfig): The parallel configuration. Default `default_dpmp_config`,
            an instance of `OpParallelConfig` with default args.

    Inputs:
        - **logits** (Tensor) - Tensor of shape (N, C). Data type must be float16 or float32. The output logits of
          the backbone.

        - **label** (Tensor) - Tensor of shape (N, 1). The ground truth label of the sample.

    Returns:
        The corresponding log softmax results.
    """
    def __init__(self, parallel_config=default_dpmp_config):
        super(_LogSoftmax, self).__init__()
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        # on/off value for onehot, for smooth labeling, modify the off_value
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)

        self.sum = P.ReduceSum(keep_dims=True)
        self.max = P.ReduceMax(keep_dims=True)
        self.sub = P.Sub()
        self.exp = P.Exp()
        self.log = P.Log()
        self.onehot = P.OneHot()
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL, ParallelMode.SEMI_AUTO_PARALLEL):
            self.sum.shard(((dp, mp),))
            self.max.shard(((dp, mp),))
            self.sub.shard(((dp, mp), (dp, 1)))
            self.exp.shard(((dp, mp),))
            self.log.shard(((dp, 1),))
            self.onehot.shard(((dp, mp), (), ()))

    def construct(self, logits, label):
        """Forward process"""
        logits = F.cast(logits, mstype.float32)
        logit_max = self.max(logits, 1)
        logit_sub = self.sub(logits, logit_max)
        logit_exp = self.exp(logit_sub)
        exp_sum = self.sum(logit_exp, -1)
        log_exp_sum = self.log(exp_sum)
        log_softmax_result = self.sub(logit_sub, log_exp_sum)

        one_hot_label = self.onehot(label, F.shape(logits)[-1], self.on_value, self.off_value)
        return log_softmax_result, one_hot_label

    def bprop(self, logits, label, _, dout):
        """just return the loss of the dout. Note this should be used together with _NLLLoss"""
        d_logits = F.cast(dout[0], F.dtype(logits))
        return d_logits, F.zeros_like(label)


class _NLLLoss(nn.Cell):
    """
    Calculate the NLLLoss results with given log softmax results and the label. The bprop of the cell is rewritten.
    This cell should be used together with _Log_softmax, to optimize the bprop of the cross entropy loss.

    Args:
        parallel_config (OpParallelConfig): The parallel configuration. Default `default_dpmp_config`,
            an instance of `OpParallelConfig` with default args.

    Inputs:
        - **log_softmax_result** (Tensor) - Tensor of shape (N, C). Data type is float32.
        - **one_hot_label** (Tensor) - Tensor of shape (N, C). The ground truth label in one-hot format of the sample.

    Returns:
        The corresponding loss results.
    """
    def __init__(self, parallel_config=default_dpmp_config):
        super(_NLLLoss, self).__init__()
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.repeat_loss = 1
        self.gather_d = P.GatherD()
        self.expand_dims = P.ExpandDims()
        self.sum = P.ReduceSum()
        self.mul = P.Mul()
        self.neg = P.Neg()
        # In auto parallel, there will be a virtual div in the back propagation begins. As we use custom bprop function
        # we need to eliminate this virtual div by adding a factor "mp".
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL, ParallelMode.SEMI_AUTO_PARALLEL):
            self.repeat_loss = mp
            self.sum.shard(((dp, mp),))
            self.mul.shard(((dp, mp), (dp, mp)))
            self.neg.shard(((dp, mp),))

    def construct(self, log_softmax_result, one_hot_label):
        """Forward process"""
        loss = self.mul(log_softmax_result, one_hot_label)
        loss_unsum = self.neg(loss)
        loss_reduce = self.sum(loss_unsum, -1)
        return loss_reduce

    def bprop(self, log_softmax_result, one_hot_label, _, dout):
        """A simplified function. Note this should be used together with _Softmax"""
        softmax_result = P.Exp()(log_softmax_result)
        logits = softmax_result - one_hot_label
        logits = logits * P.ExpandDims()(dout, -1) * self.repeat_loss

        return logits, F.zeros_like(one_hot_label)


@MindFormerRegister.register(MindFormerModuleType.LOSS)
class CrossEntropyLoss(nn.Cell):
    r"""
    Calculate the cross entropy loss.

    CrossEntropyLoss supports two different types of targets:

    - Class indices (int), where the range of values is :math:`[0, C)` with :math:`C` being the number of classes.
      When reduction is set to 'none', the cross-entropy loss is computed as follows:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}
          \cdot \mathbb{1}\{y_n \not= \text{ignore_index}\}

      where :math:`x` denotes the predicted values, :math:`t` denotes the target values, :math:`w` denotes the weights,
      and :math:`N` is the batch size. The index :math:`c` ranges from [0, C-1], representing the class indices,
      where :math:`C` is the number of classes.

      If reduction is not set to 'none' (the default is 'mean'), the loss is computed as:

      .. math::
          \ell(x, y) = \begin{cases}
              \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n} \cdot \mathbb{1}\{y_n \not= \text{ignore_index}\}} l_n, &
              \text{if reduction} = \text{'mean',}\\
              \sum_{n=1}^N l_n,  &
              \text{if reduction} = \text{'sum'.}
              \end{cases}

    - Class probabilities (float), used when the target is a probability distribution over multiple class labels.
      When reduction is set to 'none', the cross-entropy loss is computed as follows:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - \sum_{c=1}^C w_c \log \frac{\exp(x_{n,c})}{\sum_{i=1}^C \exp(x_{n,i})} y_{n,c}

      where :math:`x` denotes the predicted values, :math:`t` denotes the target values, :math:`w` denotes the weights,
      and :math:`N` is the batch size. The index :math:`c` ranges from [0, C-1], representing the class indices,
      where :math:`C` is the number of classes.

      If reduction is not set to 'none' (the default is 'mean'), the loss is computed as:

      .. math::
          \ell(x, y) = \begin{cases}
              \frac{\sum_{n=1}^N l_n}{N}, &
              \text{if reduction} = \text{'mean',}\\
              \sum_{n=1}^N l_n,  &
              \text{if reduction} = \text{'sum'.}
              \end{cases}

    Args:
        parallel_config (mindformers.modules.OpParallelConfig, optional): The parallel
            configuration. Default: ``default_dpmp_config``.
        check_for_nan_in_loss_and_grad (bool, optional): Whether to print local loss. Default: ``False``.
        monitor_device_local_loss (bool, optional): Whether to monitor device local loss. Default: ``False``.
        calculate_per_token_loss (bool, optional): Whether to use Megatron loss. Default: ``False``.
        seq_split_num (int, optional): Sequence split number in sequence pipeline parallel mode. Default: ``1``.

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
        >>> from mindformers.core import CrossEntropyLoss
        >>> loss = CrossEntropyLoss()
        >>> logits = Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]), mstype.float32)
        >>> labels_np = np.array([1]).astype(np.int32)
        >>> input_mask = Tensor(np.ones(1).astype(np.float32))
        >>> labels = Tensor(labels_np)
        >>> output = loss(logits, labels, input_mask)
        >>> output.shape
        (1,)
    """
    @_LogActionOnce(m_logger=logger, key='CrossEntropyLoss',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    def __init__(self, parallel_config=default_dpmp_config,
                 check_for_nan_in_loss_and_grad=False, monitor_device_local_loss=False,
                 calculate_per_token_loss=False, seq_split_num=1, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        loss_tag = kwargs.get("loss_tag", "lm")
        self.seq_pipe = seq_split_num > 1
        self.kwargs = kwargs
        self.enable_force_redistribute = False
        self.sum2 = P.ReduceSum()
        self.mul2 = P.Mul()
        self.relu = P.ReLU()
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL, ParallelMode.SEMI_AUTO_PARALLEL):
            self.enable_force_redistribute = True
            self.add = P.Add().shard(((dp, mp), ())).add_prim_attr("keep_alive", True)
            self.add_label = P.Add().shard(((dp,), ())).add_prim_attr("keep_alive", True)
            self._check_and_modify_sharding_context(dp)
            self.sum2.shard(((1,),))
            self.mul2.shard(((1,), (1,)))
            self.relu.shard(((1,),))
        self.add2 = P.Add()
        self.div2 = P.RealDiv()
        self.monitor_local_loss = check_for_nan_in_loss_and_grad
        self.monitor_device_local_loss = monitor_device_local_loss
        if self.monitor_device_local_loss:
            self.device_local_loss = get_device_local_loss()
        self.dump_local_loss = (
            is_dump_supported() and
            bool(get_auto_parallel_context("dump_local_norm_path")) and
            self.monitor_local_loss
        )
        if self.dump_local_loss:
            self.dump = P.TensorDump()
            self.dump_path = os.path.join(get_auto_parallel_context("dump_local_norm_path"), f"rank_{get_real_rank()}")
            self.local_loss_filename = os.path.join(self.dump_path, f"local_loss__{loss_tag}")

        self._log_softmax = _LogSoftmax(parallel_config)
        self._nllloss = _NLLLoss(parallel_config)
        self.calculate_per_token_loss = calculate_per_token_loss

        self.need_monitor = self.monitor_local_loss or self.monitor_device_local_loss
        if self.need_monitor:
            self.local_sum2 = P.ReduceSum().add_prim_attr("cross_batch", True)

    @staticmethod
    def _check_and_modify_sharding_context(dp):
        device_num = _get_device_num()
        stages = _get_pipeline_stages()
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and dp * stages != device_num:
            set_algo_parameters(fully_use_devices=False)

    def construct(self, logits, label, input_mask):
        """Forward process"""
        # The add is used for forcing the redistribution before stepping in sub graphs, when semi/auto parallel enabled.
        if self.enable_force_redistribute:
            logits = self.add(logits, 0)
            label = self.add_label(label, 0)
        log_softmax, one_hot_label = self._log_softmax(logits, label)
        loss_reduce = self._nllloss(log_softmax, one_hot_label)

        # Using input_mask to mask the loss
        input_mask = P.Reshape()(input_mask, (-1,))
        input_mask = F.depend(input_mask, loss_reduce)
        if self.need_monitor:
            local_numerator = self.local_sum2(self.mul2(loss_reduce, input_mask))
            local_denominator = self.add2(
                self.local_sum2(input_mask),
                P.Cast()(F.tuple_to_array((1e-8,)), mstype.float32))
            local_loss = self.div2(local_numerator, local_denominator)
            if self.monitor_local_loss:
                if self.dump_local_loss:
                    self.dump(self.local_loss_filename, local_loss)
                else:
                    print("local loss: ", local_loss)
            if self.monitor_device_local_loss:
                loss_reduce = F.depend(
                    loss_reduce,
                    F.assign_add(self.device_local_loss, P.Cast()(local_loss, self.device_local_loss.dtype)))

        numerator = self.sum2(self.mul2(loss_reduce, input_mask))
        denominator = self.add2(
            self.sum2(input_mask),
            P.Cast()(F.tuple_to_array((1e-8,)), mstype.float32))
        if not self.calculate_per_token_loss and not self.seq_pipe:
            return self.div2(numerator, denominator)
        return numerator, denominator
