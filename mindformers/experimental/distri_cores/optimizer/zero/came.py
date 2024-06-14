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
"Came Optimizer"
import numpy as np

import mindspore as ms
from mindspore.common import dtype as mstype
from mindspore import ParameterTuple, Tensor, ops, Parameter
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.common.initializer import initializer
from mindspore.communication.management import GlobalComm
from mindformers.experimental.distri_cores.create_comm import get_dp_rank, \
    get_dp_world_size, get_dp_group
from .came_optim import Came
from .came_optim import trans_to_tensor

__all__ = ["CameZeRO2"]

_came_opt = ops.MultitypeFuncGraph("came_opt")
_split_params = ops.MultitypeFuncGraph("split_params")
_update_params = ops.MultitypeFuncGraph("update_params")


@_came_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _update_by_opt(opt, beta2t, lr, gradients, parameters, exp_avg, exp_avg_sq_row, exp_avg_sq_col, \
                   exp_avg_sq, exp_avg_insta_row, exp_avg_insta_col):
    """
    Apply AdamWeigthDecay operator to update parameters.
    """
    output = opt(beta2t, lr, gradients, parameters, exp_avg, exp_avg_sq_row, exp_avg_sq_col, exp_avg_sq, \
                 exp_avg_insta_row, exp_avg_insta_col)
    return output


@_split_params.register("Number", "Number", "Tensor", "Bool")
def _split_params_to_fp32(shard_id, shard_size, param, need_split):
    """
    Split parameters.
    """
    split = P.Split(0, shard_size)
    cast = P.Cast()
    if need_split:
        splited_param = split(param)[shard_id]
    else:
        splited_param = param
    if splited_param.dtype != mstype.float32:
        splited_param = cast(splited_param, mstype.float32)
    return splited_param


@_update_params.register("Tensor", "Tensor", "Function")
def _update_params_opt_parallel(param, update, all_gather):
    """
    Allgather updated parameters and load.
    """
    cast = P.Cast()
    if all_gather:
        update = all_gather(update)
    if update.dtype != param.dtype:
        update = cast(update, param.dtype)
    param.assign_value(update)

class CameZeRO2(Optimizer):
    r"""
    This class is an implementation of Came optimizer, which support ZeRO2 optimizer parallel.

    Args:
        params (Union[list[Parameter], list[dict]]): When the `params` is a list of `Parameter` which will be updated,
            the element in `params` must be class `Parameter`.
        learning_rate (Union[float, Tensor]): A value or a graph for the learning rate.
            When the learning_rate is a Tensor in a 1D dimension.
            If the type of `learning_rate` is int, it will be converted to float. Default: None.
        eps (Union[list, tuple]): The regularization constans for square gradient, parameter scale and
            instability_matrix respectively. default: (1e-30, 1e-3, 1e-16)
        clip_threshold (float): The threshold of root mean square of final gradient update. default: 1.0
        decay_rate (float): The coefficient used to compute running averages of square gradient.
            Should be in range [0.0, 1.0]. default: 0.8.
        beta1 (float): The coefficient to computing running averages of gradient. Should be in range [0.0, 1.0].
               Default: 0.9.
        beta3 (float): The coefficient to computing running averages of gradient. Should be in range [0.0, 1.0].
               Default: 0.99.
        weight_decay (float): Weight decay (L2 penalty). It must be equal to or greater than 0.
            Should be in range [0.0, 1.0]. default: 0.0.
        scale_parameter (bool): If True, learning rate is scaled by root mean square of parameter. default: True
        relative_step (bool): If True, time-dependent learning rate is computed instead of external learning rate.
            default: True
        warmup_init (bool): The time-dependent learning rate computation depends on whether warm-up
            initialization is being used. default: False
        compression (bool): If True, the data type of the running averages exponent will be compression to float16.
            default: False
        loss_scale (int): An integer point value for the loss scale. Should be greater than 0. In general, use the
            default value. Only when `FixedLossScaleManager` is used for training and the `drop_overflow_update` in
            `FixedLossScaleManager` is set to False, then this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.amp.FixedLossScaleManager` for more details.
            Default: 1.
        use_parallel (bool): Enable optimizer parallel. Default: False.
        opt_parallel_group (str): Name of communication group used by optimizer parallel. Default: None.
        cpu_offload (bool): The process of optimizer will be offload to host. The gradients, parameters and optimizer
            status will be offload to host. Default: Flase.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore.communication.management import init, get_rank, get_group_size, GlobalComm
        >>> from mindspore.nn.wrap.cell_wrapper import WithLossCell
        >>> from mindformers import CameZeRO2
        >>> from mindformers.experimental.distri_cores.create_comm import initialize_model_parallel
        >>> loss = SoftmaxCrossEntropyWithLogits()
        >>> ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
        >>> ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
        >>> init()
        >>> initialize_model_parallel()
        >>> optimizer = CameZeRO2(params=network.get_parameters(), learning_rate=1e-1, use_parallel=True,
        ...                       opt_parallel_group=GlobalComm.WORLD_COMM_GROUP, cpu_offload=False)
    """
    _support_parallel_optimizer = True

    def __init__(self, params, learning_rate=None, eps=(1e-30, 1e-3, 1e-16), clip_threshold=1.0, decay_rate=0.8,
                 beta1=0.9, beta3=0.99, weight_decay=0.0, scale_parameter=False, relative_step=False, \
                 warmup_init=False, compression=False, loss_scale=1, use_parallel=False, opt_parallel_group=None, \
                 cpu_offload=False):
        super(CameZeRO2, self).__init__(learning_rate, params, weight_decay)
        self._is_stand_alone_mode = (ms.get_auto_parallel_context("parallel_mode") == ms.ParallelMode.STAND_ALONE)
        self.use_parallel = use_parallel
        if opt_parallel_group:
            self.opt_parallel_group = opt_parallel_group
        elif self.use_parallel:
            self.opt_parallel_group = GlobalComm.WORLD_COMM_GROUP
        self.cpu_offload = cpu_offload
        # init communication group info
        self._init_optimizer_shard_info()

        self._parameter_splited = [False] * len(self._parameters)
        self.all_gather_ops = self._init_all_gather_ops(self._parameters)
        if self.use_parallel or not self._is_stand_alone_mode:
            self._regist_hook_for_params()

        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta3 = Tensor(np.array([beta3]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.compression = compression
        self.shard_parameters = self._init_parameters(self._parameters)
        self.exp_avg, self.exp_avg_sq_row, self.exp_avg_sq_col, self.exp_avg_insta_row, \
        self.exp_avg_insta_col, self.exp_avg_sq = self.init_came_state(self.beta1,
                                                                       self.shard_parameters)
        self._opt_params_need_offload = {"beta1": self.beta1, "beta3": self.beta3, "eps": self.eps, \
                                         "exp_avg": self.exp_avg, "exp_avg_sq_row": self.exp_avg_sq_row, \
                                         "exp_avg_sq_col": self.exp_avg_sq_col,
                                         "exp_avg_insta_row": self.exp_avg_insta_row, \
                                         "exp_avg_insta_col": self.exp_avg_insta_col,
                                         "exp_avg_sq": self.exp_avg_insta_col}

        self.opt = Came(self.shard_parameters, learning_rate, eps, clip_threshold, decay_rate, beta1, beta3,
                        weight_decay, \
                        scale_parameter, relative_step, warmup_init, compression, loss_scale, cpu_offload)
        self.step = Parameter(initializer(0, [1], mstype.float32), name='afactor_step')
        self.decay_rate = trans_to_tensor(-decay_rate)

    def _init_optimizer_shard_info(self):
        """Init optimizer parallel information."""
        if not self.use_parallel:
            self.shard_id = 0
            self.shard_size = 1
        else:
            self.shard_size = get_dp_world_size()
            self.shard_id = get_dp_rank()

    def _init_all_gather_ops(self, params):
        """Init allgather operations for each parameter."""
        op_list = []
        for i, param in enumerate(params):
            if self.use_parallel and param.shape[0] % self.shard_size == 0:
                op_list.append(P.AllGather(group=get_dp_group()))
                self._parameter_splited[i] = True
            else:
                op_list.append(None)
        return tuple(op_list)

    def _regist_hook_for_params(self):
        """Register hook for model parameters for optimizer parallel."""
        def reduce_scatter_hook(grad):
            allreduce = P.AllReduce(group=get_dp_group())
            split = P.Split(0, self.shard_size)
            return split(allreduce(grad))[self.shard_id].contiguous()
        def reduce_hook(grad):
            allreduce = P.AllReduce(group=get_dp_group())
            return allreduce(grad)
        for i, param in enumerate(self._parameters):
            if self._parameter_splited[i]:
                param.register_hook(reduce_scatter_hook)
            else:
                param.register_hook(reduce_hook)

    def _init_parameters(self, params, init="zeros"):
        """Init momentum or variance for came optimizer."""
        moments_list = []
        for i, param in enumerate(params):
            param_shape = param.shape
            if self._parameter_splited[i]:
                param_shape = list(param_shape)
                param_shape[0] = param_shape[0] // self.shard_size
                param_shape = tuple(param_shape)
            moment = ms.Parameter(initializer(init, shape=param_shape, dtype=mstype.float32),
                                  name=param.name)
            moments_list.append(moment)

        return ParameterTuple(moments_list)

    def init_came_state(self, beta1, params):
        """init came variables"""
        exp_avg = []
        for param in params:
            param_shape = param.shape
            if beta1 > 0:
                exp_avg_shard = Parameter(initializer("zeros", shape=param_shape, dtype=mstype.float32),
                                          name=param.name)
            else:
                exp_avg_shard = Parameter(Tensor(0.0))
            exp_avg.append(exp_avg_shard)
        exp_avg = ParameterTuple(exp_avg)

        exp_avg_sq = []
        exp_avg_sq_col = []
        exp_avg_sq_row = []
        exp_avg_insta_col = []
        exp_avg_insta_row = []
        for param in params:
            param_dtype = param.dtype
            param_shape = param.shape
            param_name = param.name
            if len(param_shape) > 1:
                exp_avg_sq_row.append(Parameter(initializer(0, shape=param_shape[:-1], dtype=param_dtype),
                                                name="exp_avg_sq_row_{}".format(param_name)))
                exp_avg_sq_col.append(Parameter(initializer(0, shape=param_shape[:-2] + param_shape[-1:],
                                                            dtype=param_dtype),
                                                name="exp_avg_sq_col_{}".format(param_name)))
                exp_avg_insta_row.append(Parameter(initializer(0, shape=param_shape[:-1], dtype=param_dtype),
                                                   name="exp_avg_insta_row_{}".format(param_name)))
                exp_avg_insta_col.append(Parameter(initializer(0, shape=param_shape[:-2] + param_shape[-1:],
                                                               dtype=param_dtype),
                                                   name="exp_avg_insta_col_{}".format(param_name)))
                exp_avg_sq.append(Parameter(initializer(0, shape=(1,), dtype=param_dtype),
                                            name="exp_avg_sq_{}".format(param_name)))

            else:
                exp_avg_sq_row.append(Parameter(initializer(0, shape=(1,), dtype=param_dtype),
                                                name="exp_avg_sq_row_{}".format(param_name)))
                exp_avg_sq_col.append(Parameter(initializer(0, shape=(1,), dtype=param_dtype),
                                                name="exp_avg_sq_col_{}".format(param_name)))
                exp_avg_insta_row.append(Parameter(initializer(0, shape=(1,), dtype=param_dtype),
                                                   name="exp_avg_insta_row_{}".format(param_name)))
                exp_avg_insta_col.append(Parameter(initializer(0, shape=(1,), dtype=param_dtype),
                                                   name="exp_avg_insta_col_{}".format(param_name)))

                if self.compression:
                    exp_avg_sq.append(Parameter(initializer(0, shape=param_shape, dtype=mstype.float16),
                                                name="exp_avg_sq_{}".format(param_name)))
                else:
                    exp_avg_sq.append(Parameter(initializer(0, shape=param_shape, dtype=param_dtype),
                                                name="exp_avg_sq_{}".format(param_name)))

        exp_avg_sq_row = ParameterTuple(exp_avg_sq_row)
        exp_avg_sq_col = ParameterTuple(exp_avg_sq_col)
        exp_avg_insta_row = ParameterTuple(exp_avg_insta_row)
        exp_avg_insta_col = ParameterTuple(exp_avg_insta_col)
        exp_avg_sq = ParameterTuple(exp_avg_sq)
        return exp_avg, exp_avg_sq_row, exp_avg_sq_col, exp_avg_insta_row, \
               exp_avg_insta_col, exp_avg_sq

    def _offload_optimizer_params(self):
        """Offload optimizer parameters to host."""
        for _, value in self._opt_params_need_offload.items():
            # pylint: disable=W0212
            if isinstance(value, ParameterTuple):
                for param in value:
                    param._offload()
            else:
                value._offload()

    def construct(self, grads):
        """construct method"""
        params = self.hyper_map(F.partial(_split_params, self.shard_id, self.shard_size),
                                self._parameters, self._parameter_splited)
        grads = self.flatten_gradients(grads)

        lr = self.get_lr()

        # pylint: disable=W0212
        if self.cpu_offload:
            self._offload_optimizer_params()
            for grad in grads:
                grad._offload()
            for param in params:
                param._offload()


        self.assignadd(self.global_step, self.global_step_increase_tensor)
        F.assign_add(self.step, 1)
        step = self.step
        beta2t = 1.0 - P.Pow()(step, self.decay_rate)

        optim_result = self.hyper_map(F.partial(_came_opt, self.opt, beta2t, lr),
                                      grads, params, self.exp_avg, self.exp_avg_sq_row, self.exp_avg_sq_col, \
                                      self.exp_avg_sq, self.exp_avg_insta_row, self.exp_avg_insta_col
                                      )
        self.hyper_map(_update_params, self._parameters, optim_result, self.all_gather_ops)
