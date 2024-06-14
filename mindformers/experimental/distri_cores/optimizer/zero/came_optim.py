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
"""came optimizer"""
from __future__ import absolute_import

from mindspore.common import dtype as mstype
from mindspore.log import logging
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
try:
    from mindspore._checkparam import Validator as validator
    from mindspore._checkparam import Rel
except ImportError:
    import mindspore._checkparam as validator
    import mindspore._checkparam as Rel
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.experimental.distri_cores.create_comm import get_dp_world_size, \
    get_dp_group

__all__ = ['Came']


def _rms(update_tensor):
    """calculate rms"""
    col_sum = P.AllReduce("sum", group=get_dp_group())(F.square(update_tensor))
    mean_value = col_sum.mean() / get_dp_world_size()
    return F.sqrt(mean_value)


def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
    """Approximation of exponential moving average of square of gradient"""
    reduce_mean = P.ReduceMean(keep_dims=True)(exp_avg_sq_row, -1)
    div_val = 1.0 / P.Sqrt()(P.RealDiv()(exp_avg_sq_row, reduce_mean))
    r_factor = (P.ExpandDims()(div_val, -1))

    exp_avg_sq_col = P.ExpandDims()(exp_avg_sq_col, -2)
    c_factor = 1.0 / P.Sqrt()(exp_avg_sq_col)
    return P.Mul()(r_factor, c_factor)


reduce_mean_keep_alive = P.ReduceMean().add_prim_attr("keep_alive", True)


def _run_opt_with_one_number(eps, clip_threshold, beta1, beta2t, beta3, weight_decay, scale_parameter,
                             compression, use_first_moment, weight_decay_flag, learning_rate,
                             grad, param, exp_avg, exp_avg_sq_row, exp_avg_sq_col, exp_avg_sq,
                             exp_avg_insta_row, exp_avg_insta_col):
    """Apply came optimizer to the weight parameter using Tensor."""
    cast = P.Cast()
    grad_dtype = F.dtype(grad)
    grad_shape = F.shape(grad)

    if grad_dtype == mstype.float16:
        grad = cast(grad, mstype.float32)
    p_data_fp32 = param
    if F.dtype(p_data_fp32) == mstype.float16:
        p_data_fp32 = cast(p_data_fp32, mstype.float32)

    factored = len(grad_shape) >= 2
    if scale_parameter:
        rms = _rms(p_data_fp32)
        param_scale = P.Maximum()(eps[1], rms)
        learning_rate_update = learning_rate * param_scale * F.ones_like(rms)
    else:
        learning_rate_update = learning_rate
    update = (grad ** 2) + eps[0]

    if factored:
        exp_avg_sq_row_update = cast(exp_avg_sq_row, grad_dtype)
        exp_avg_sq_row_update = P.Mul()(exp_avg_sq_row_update, beta2t)
        update_mean = reduce_mean_keep_alive(update, -1) * (1.0 - beta2t)
        exp_avg_sq_row_update = P.Add()(exp_avg_sq_row_update, update_mean)
        F.assign(exp_avg_sq_row, cast(exp_avg_sq_row_update, F.dtype(exp_avg_sq_row)))
        exp_avg_sq_row_update = exp_avg_sq_row

        exp_avg_sq_col_update = cast(exp_avg_sq_col, grad_dtype)
        exp_avg_sq_col_update = P.Mul()(exp_avg_sq_col_update, beta2t)
        update_mean = reduce_mean_keep_alive(update, -2) * (1.0 - beta2t)
        exp_avg_sq_col_update = P.Add()(exp_avg_sq_col_update, update_mean)
        F.assign(exp_avg_sq_col, cast(exp_avg_sq_col_update, F.dtype(exp_avg_sq_col)))
        exp_avg_sq_col_update = exp_avg_sq_col
        update = _approx_sq_grad(exp_avg_sq_row_update, exp_avg_sq_col_update)
        update = P.Mul()(update, grad)

    else:
        exp_avg_sq_update = cast(exp_avg_sq, grad_dtype)
        update = update * (1.0 - beta2t)
        exp_avg_sq_update = P.Add()(P.Mul()(exp_avg_sq_update, beta2t), update)
        F.assign(exp_avg_sq, cast(exp_avg_sq_update, F.dtype(exp_avg_sq)))
        exp_avg_sq_update = exp_avg_sq
        exp_avg_sq_update = 1.0 / P.Sqrt()(exp_avg_sq_update)
        update = P.Mul()(exp_avg_sq_update, grad)
    update_rms_thres = _rms(update) / clip_threshold
    update_coff = P.Maximum()(update_rms_thres, P.OnesLike()(update_rms_thres))
    update = P.RealDiv()(update, update_coff)
    if use_first_moment:
        exp_avg_update = exp_avg
        if compression:
            exp_avg_update = cast(exp_avg, grad_dtype)
        exp_avg_update = P.Add()(P.Mul()(exp_avg_update, beta1), update * (1 - beta1))
        F.assign(exp_avg, cast(exp_avg_update, F.dtype(exp_avg)))
    ###
    # CAME  optimizer modification is here
    instability_matrix = (update - exp_avg) ** 2 + eps[2]
    if factored:
        exp_avg_insta_row_update = cast(exp_avg_insta_row, grad_dtype)
        exp_avg_insta_row_update = P.Mul()(exp_avg_insta_row_update, beta3)
        update_mean = reduce_mean_keep_alive(instability_matrix, -1) * (1.0 - beta3)
        exp_avg_insta_row_update = P.Add()(exp_avg_insta_row_update, update_mean)
        F.assign(exp_avg_insta_row, cast(exp_avg_insta_row_update, F.dtype(exp_avg_insta_row)))
        exp_avg_insta_row_update = exp_avg_insta_row

        exp_avg_insta_col_update = cast(exp_avg_insta_col, grad_dtype)
        exp_avg_insta_col_update = P.Mul()(exp_avg_insta_col_update, beta3)
        update_mean = reduce_mean_keep_alive(instability_matrix, -2) * (1.0 - beta3)
        exp_avg_insta_col_update = P.Add()(exp_avg_insta_col_update, update_mean)
        F.assign(exp_avg_insta_col, cast(exp_avg_insta_col_update, F.dtype(exp_avg_insta_col)))
        exp_avg_insta_col_update = exp_avg_insta_col

        s_t = _approx_sq_grad(exp_avg_insta_row_update, exp_avg_insta_col_update)
        update = s_t * exp_avg * learning_rate_update
    else:
        update = exp_avg * learning_rate_update
    # ###
    if weight_decay_flag:
        p_data_fp32_coff = p_data_fp32 * -weight_decay * learning_rate_update
        p_data_fp32 = P.Add()(p_data_fp32, p_data_fp32_coff)
    p_data_fp32 = P.Sub()(p_data_fp32, update)
    return p_data_fp32

def _run_fused_ada_factor(fused_ada_factor, eps, clip_threshold, beta1, beta2t, weight_decay, learning_rate,
                          grad, param, exp_avg, exp_avg_sq_row, exp_avg_sq_col, exp_avg_sq):
    fused_ada_factor(eps, clip_threshold, beta1, beta2t, weight_decay, learning_rate,
                     grad, param, exp_avg, exp_avg_sq_row, exp_avg_sq_col, exp_avg_sq)
    return True


def trans_to_tensor(param, is_tuple=False, fp32=True):
    """
    Transform params to tensor.
    """
    if param is None or isinstance(param, bool):
        return param
    data_type = mstype.float32 if fp32 else mstype.float16
    if is_tuple:
        new_param = [Tensor(ele, data_type) for ele in param]
        return tuple(new_param)
    return Tensor(param, data_type)


@MindFormerRegister.register(MindFormerModuleType.OPTIMIZER)
class Came(Optimizer):
    r"""
    Updates gradients by the Confidence-guided Adaptive Memory Efficient Optimization (Came) algorithm.

    The Came algorithm is proposed in `CAME: Confidence-guided Adaptive Memory Efficient Optimization
    <https://arxiv.org/abs/2307.02047>`.
    """
    _support_parallel_optimizer = True

    @opt_init_args_register
    def __init__(self,
                 params,
                 learning_rate=None,
                 eps=(1e-30, 1e-3, 1e-16),
                 clip_threshold=1.0,
                 decay_rate=0.8,
                 beta1=0.9,
                 beta3=0.99,
                 weight_decay=0.0,
                 scale_parameter=False,
                 relative_step=False,
                 warmup_init=False,
                 compression=False,
                 loss_scale=1,
                 cpu_offload=False):

        if learning_rate is not None and relative_step:
            raise ValueError("Cannot combine manual lr and relative_step options", learning_rate)
        if warmup_init and not relative_step:
            raise ValueError("warmup_init requires relative_step=True")
        if learning_rate is None and not relative_step:
            raise ValueError("Cannot learning_rate is None and relative_step=False")
        if learning_rate is None:
            learning_rate = 0.0
        if beta1 is None:
            beta1 = 0.0

        if not isinstance(learning_rate, (float, int)) and learning_rate is not None:
            if relative_step or scale_parameter:
                logging.warning("When learning_rate is learning scheduler, it not support update learning rate!")

        validator.check_value_type("loss_scale", loss_scale, [int], self.cls_name)
        super(Came, self).__init__(learning_rate, params, weight_decay, loss_scale)
        validator.check_value_type("eps", eps, [list, tuple], self.cls_name)
        if len(eps) != 3:
            raise ValueError("eps must have 3 value: (eps1, eps2, eps3).")
        for i, ele in enumerate(eps):
            validator.check_value_type("eps{}".format(i), ele, [float], self.cls_name)
            validator.check_non_negative_float(ele, "eps{}".format(i), self.cls_name)
        validator.check_value_type("clip_threshold", clip_threshold, [float], self.cls_name)
        validator.check_non_negative_float(clip_threshold, "clip_threshold", self.cls_name)
        validator.check_value_type("decay_rate", decay_rate, [float], self.cls_name)
        validator.check_float_range(decay_rate, 0, 1, Rel.INC_BOTH, "decay_rate", self.cls_name)
        validator.check_value_type("weight_decay", weight_decay, [float], self.cls_name)
        validator.check_float_range(weight_decay, 0, 1, Rel.INC_BOTH, "weight_decay", self.cls_name)
        validator.check_value_type("scale_parameter", scale_parameter, [bool], self.cls_name)
        validator.check_value_type("relative_step", relative_step, [bool], self.cls_name)
        validator.check_value_type("warmup_init", warmup_init, [bool], self.cls_name)
        validator.check_value_type("compression", compression, [bool], self.cls_name)
        validator.check_value_type("beta1", beta1, [float], self.cls_name)
        validator.check_float_range(beta1, 0, 1, Rel.INC_BOTH, "beta1", self.cls_name)
        validator.check_value_type("beta3", beta3, [float], self.cls_name)
        validator.check_float_range(beta3, 0, 1, Rel.INC_BOTH, "beta3", self.cls_name)
        self.eps = trans_to_tensor(eps)
        self.clip_threshold = trans_to_tensor(clip_threshold)
        self.decay_rate = trans_to_tensor(-decay_rate)
        self.beta1 = trans_to_tensor(beta1)
        self.beta3 = trans_to_tensor(beta3)
        self.weight_decay = trans_to_tensor(weight_decay)
        self.weight_decay_flag = bool(weight_decay)

        self.scale_parameter = scale_parameter
        self.relative_step = relative_step
        self.warmup_init = warmup_init
        self.compression = compression
        if beta1 > 0:
            self.use_first_moment = True
        else:
            self.use_first_moment = False
        self.step = Parameter(initializer(0, [1], mstype.float32), name='afactor_step')
        self.fused_ada_factor = P.FusedAdaFactor(enable_scale_parameter=self.scale_parameter,
                                                 enable_first_moment=self.use_first_moment,
                                                 enable_weight_decay=self.weight_decay_flag)
        self.use_fused_ada_factor = cpu_offload
        if self.use_fused_ada_factor:
            self._set_base_target("CPU")
            self.fused_ada_factor.set_device("CPU")
        logging.info("Came init completed %s.", self.learning_rate)

    def construct(self, beta2t, lr, gradients, parameters, exp_avg, exp_avg_sq_row, \
                  exp_avg_sq_col, exp_avg_sq, exp_avg_insta_row, exp_avg_insta_col):
        """construct of came optimizer."""
        if self.use_fused_ada_factor:
            update_parameters = self.fused_ada_factor(self.eps, self.clip_threshold,
                                                      self.beta1, beta2t, self.weight_decay, lr,
                                                      gradients, parameters, exp_avg, exp_avg_sq_row, \
                                                      exp_avg_sq_col, exp_avg_sq)
        else:
            update_parameters = _run_opt_with_one_number(self.eps, self.clip_threshold, self.beta1, \
                                                         beta2t, self.beta3, self.weight_decay, \
                                                         self.scale_parameter, self.compression, \
                                                         self.use_first_moment, self.weight_decay_flag, lr, \
                                                         gradients, parameters, exp_avg, exp_avg_sq_row, \
                                                         exp_avg_sq_col, exp_avg_sq, exp_avg_insta_row, \
                                                         exp_avg_insta_col)
        return update_parameters
