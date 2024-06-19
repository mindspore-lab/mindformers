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

"""
For training
"""
from copy import deepcopy
import mindspore.common.dtype as mstype
from mindspore import nn, Tensor, Parameter
from mindspore.nn.cell import Cell
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean, _get_parallel_mode, _is_pynative_parallel
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.parallel._utils import _get_enable_parallel_optimizer
from mindformers.core.clip_grad import ClipGradNorm
from mindformers.experimental.distri_cores.create_comm import get_dp_group, get_dp_world_size


class TrainOneStepCell(Cell):
    r"""
    Network training package class.

    Wraps the `network` with the `optimizer`. The resulting Cell is trained with input '\*inputs'.
    The backward graph will be created in the construct function to update the parameter. Different
    parallel modes are available for training.

    Args:
        network (Cell): The training network. The network only supports single output.
        optimizer (Union[Cell]): Optimizer for updating the network parameters.
        sens (numbers.Number): The scaling number to be filled as the input of backpropagation. Default value is
            ``None`` , which is ``1.0`` .
        return_grad (bool): Whether to return gradient. If ``True``, it will return the gradient in the form of a dict
            while returning loss. The key of the dict is the parameter name corresponding to the gradient, and value
            is the gradient value. Default value is ``False`` .

    Inputs:
        - **\*inputs** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a tensor means the loss value, the shape of which is usually :math:`()`.

    Raises:
        TypeError: If `sens` is not a numbers.Number.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> #1) Using the WithLossCell provided by MindSpore
        >>> loss_net = nn.WithLossCell(net, loss_fn)
        >>> train_net = nn.TrainOneStepCell(loss_net, optim)
        >>>
        >>> #2) Using user-defined WithLossCell
        >>> class MyWithLossCell(nn.Cell):
        ...    def __init__(self, backbone, loss_fn):
        ...        super(MyWithLossCell, self).__init__(auto_prefix=False)
        ...        self._backbone = backbone
        ...        self._loss_fn = loss_fn
        ...
        ...    def construct(self, x, y, label):
        ...        out = self._backbone(x, y)
        ...        return self._loss_fn(out, label)
        ...
        ...    @property
        ...    def backbone_network(self):
        ...        return self._backbone
        ...
        >>> loss_net = MyWithLossCell(net, loss_fn)
        >>> train_net = nn.TrainOneStepCell(loss_net, optim)
    """

    def __init__(self, network, optimizer, sens=None, return_grad=False):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.zero_opt_lists = ["AdamWeightDecayZeRO2"]
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.grad_no_sens = C.GradOperation(get_by_list=True)
        self.sens = sens
        if self.sens == 0:
            raise ValueError("The input argument of 'sens' can not be 0.")
        self.sense_flag = True
        if self.sens is None:
            self.sense_flag = False
            self.sens = 1.0
        self.return_grad = return_grad
        if return_grad:
            self.weights_name = [i.name for i in self.optimizer.parameters]
        self.reducer_flag = False
        self.grad_reducer = nn.Identity()
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL) or \
                            _is_pynative_parallel()
        self.opt_flag = self.check_opt_in_list(optimizer)
        if not self.opt_flag:
            if self.reducer_flag:
                self.mean = _get_gradients_mean()
                self.degree = _get_device_num()
                from mindspore.communication.management import GlobalComm
                group = GlobalComm.WORLD_COMM_GROUP
                if isinstance(self.optimizer, (nn.AdaSumByGradWrapCell, nn.AdaSumByDeltaWeightWrapCell)):
                    from mindspore.communication.management import get_group_size, create_group, get_rank
                    group_number = get_group_size() // 8
                    self.degree = int(self.degree / group_number)
                    group_list = [list(range(x * self.degree, (x + 1) * self.degree)) for x in range(group_number)]
                    current_index = get_rank() // 8
                    server_group_name = "allreduce_" + str(current_index)
                    create_group(server_group_name, group_list[current_index])
                    group = server_group_name
                self.grad_reducer = DistributedGradReducer(self.weights, self.mean, self.degree, group=group)
        self._get_attr_from_cell(network)

    def construct(self, *inputs):
        """construct for TrainOneStepCell."""
        if not self.sense_flag:
            return self._no_sens_impl(*inputs)
        loss = self.network(*inputs)
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        if self.return_grad:
            grad_with_param_name = {}
            for index, value in enumerate(grads):
                grad_with_param_name[self.weights_name[index]] = value
            return loss, grad_with_param_name
        return loss

    def _no_sens_impl(self, *inputs):
        """construct implementation when the 'sens' parameter is passed in."""
        loss = self.network(*inputs)
        grads = self.grad_no_sens(self.network, self.weights)(*inputs)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        if self.return_grad:
            grad_with_param_name = {}
            for index, value in enumerate(grads):
                grad_with_param_name[self.weights_name[index]] = value
            return loss, grad_with_param_name
        return loss

    def check_opt_in_list(self, obj):
        # 获取obj的类名
        obj_class_name = type(obj).__name__

        # 检查类名是否在列表中
        return obj_class_name in self.zero_opt_lists


class PipelineTrainOneStepCell(nn.TrainOneStepWithLossScaleCell):
    """
    Append a train-one-step cell with loss scale of pipeline parallel for MindFormers.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        config (dict): model yaml loaded dict.
        use_clip_grad (bool): grad clip.
        max_grad_norm (float): The max value of grad clip norm
        scale_sense (Union[float, Cell, Tensor]): Cell to do the loss scale. Default: 1.0.
        micro_batch_num (int): Micro batch number of pipeline parallel. Default: 1.

    Inputs:
        - **forward_func** (Callable) - pipeline func for training.
        - **input_data_tuple** (tuple) - input data for training.
        - **input_data_dict** (tuple) - input data for training.

    Outputs:
        Tuple of 3 Tensor, the loss, overflow flag and current loss scale value.

        - **loss** (Tensor) -  A scalar, the loss value.
        - **overflow** (Tensor) -  A scalar, whether overflow occur or not, the type is bool.
        - **scaling_sens** (Tensor) -  The loss scale value, the shape is :math:`()` or :math:`(1,)`.
        - **learning_rate** (Tensor) -  The model learning rate .

    Raises:
        TypeError: If `scale_sense` is neither Cell nor Tensor.
        ValueError: If shape of `scale_sense` is neither (1,) nor ().
    """
    # pylint: disable=W0613
    def __init__(self, network, optimizer, config, use_clip_grad=True,
                 max_grad_norm=1.0, scale_sense=1.0, micro_batch_num=1, **kwargs):
        if isinstance(scale_sense, (int, float)):
            scale_sense = Tensor(scale_sense)
        super().__init__(network, optimizer, scale_sense)
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.optimizer = optimizer
        self.seq_length = config.model_config.seq_length
        self.hidden_size = config.model_config.hidden_size
        self.batch_size = config.training.batch_size
        self.use_clip_grad = use_clip_grad
        self.micro_batch_num = micro_batch_num
        self.config = config
        self.status = Tensor([0] * 8, mstype.int32)
        self.reshape = P.Reshape()
        self.clip_grad_norm = ClipGradNorm(max_norm=max_grad_norm)
        self.opt_shard = _get_enable_parallel_optimizer()
        self.learning_rate = deepcopy(self.optimizer.learning_rate)
        self.reduction = config.parallel_config.reduction

        self.loss_scaling_manager = None
        if isinstance(scale_sense, nn.Cell):
            self.loss_scaling_manager = scale_sense
            self.scale_sense = Parameter(Tensor(scale_sense.get_loss_scale(), dtype=mstype.float32),
                                         name="scale_sense")
        elif isinstance(scale_sense, Tensor):
            if scale_sense.shape == (1,) or scale_sense.shape == ():
                self.scale_sense = Parameter(scale_sense, name='scale_sense')
        self.allreduce = P.AllReduce(group=get_dp_group())

    @C.add_flags(has_effect=True)
    def construct(self, forward_func, *input_data_tuple, **input_data_dict):
        """The construct processes of pipeline wrapper cell."""
        scaling_sens = self.scale_sense
        loss, grads = forward_func(self.network,
                                   self.optimizer,
                                   scaling_sens,
                                   self.micro_batch_num,
                                   self.batch_size,
                                   self.seq_length,
                                   self.hidden_size,
                                   self.config.parallel_config,
                                   *input_data_tuple,
                                   **input_data_dict)
        if self.use_clip_grad:
            grads, _ = self.clip_grad_norm(grads)

        learning_rate = self.learning_rate
        if self.optimizer.dynamic_lr:
            if self.optimizer.is_group_lr:
                learning_rate = self.learning_rate[-1](self.optimizer.global_step).reshape(())
            else:
                learning_rate = self.learning_rate(self.optimizer.global_step).reshape(())

        # sum overflow flag over devices
        cond = self.get_overflow_status(self.status, grads)
        cond = F.depend(cond, grads)
        overflow = self.process_loss_scale(cond)

        if not overflow:
            loss = F.depend(loss, self.optimizer(grads))

        if get_dp_world_size() > 1:
            loss = self.allreduce(loss)
            if self.reduction == "mean":
                loss /= get_dp_world_size()
        return loss, overflow, scaling_sens.value(), learning_rate.value()
