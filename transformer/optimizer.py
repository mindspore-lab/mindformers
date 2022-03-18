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
"""AdamWeightDecay, a customized Adam for offloading."""
import numpy as np

from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.nn.optim.optimizer import Optimizer, opt_init_args_register
from mindspore.nn import AdamWeightDecay, Lamb

from transformer.global_norm import ClipByGlobalNorm
_adam_opt = C.MultitypeFuncGraph("adam_opt")
_scaler_one = Tensor(1, mstype.int32)
_scaler_ten = Tensor(10, mstype.float32)
_cpu_div = P.RealDiv().add_prim_attr("primitive_target", "CPU")
_grad_scale = C.MultitypeFuncGraph("grad_scale")
op_mul = P.Mul()
map_ = C.Map()


@_grad_scale.register("Number", "Tensor")
def tensor_grad_scale(scale, grad):
    """Get grad with scale."""
    if scale == 1.0:
        return grad
    return op_mul(grad, F.cast(scale, F.dtype(grad)))


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale_with_tensor(scale, grad):
    """Get grad with scale."""
    return op_mul(grad, F.cast(scale, F.dtype(grad)))


def scale_grad(gradients, reciprocal_scale):
    gradients = map_(F.partial(_grad_scale, reciprocal_scale), gradients)
    return gradients


@_adam_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Number", "Tensor", "Tensor",
                    "Tensor", "Tensor", "Bool", "Bool")
def _update_run_kernel(opt, clip_value, beta1, beta2, eps, lr, weight_decay,
                       param, m, v, gradient, decay_flags, optim_filter):
    """
    Update parameters by AdamWeightDecay op.
    """
    success = True
    if optim_filter:
        if decay_flags:
            next_param = opt(param, m, v, lr, beta1, beta2, eps, weight_decay,
                             P.Cast()(gradient, mstype.float16), clip_value)
        else:
            next_param = opt(param, m, v, lr, beta1, beta2, eps, 0.0,
                             P.Cast()(gradient, mstype.float16), clip_value)
        return F.depend(success, next_param)
    return success


@_adam_opt.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Number", "Tensor", "Tensor", "Tensor",
                    "Tensor", "Bool", "Bool")
def _update_run_op(beta1_power, beta2_power, beta1, beta2, eps, lr, weight_decay, param, \
                   m, v, gradient, decay_flag, optim_filter):
    """
    Update parameters.

    Args:
        beta1 (Tensor): The exponential decay rate for the 1st moment estimations. Should be in range (0.0, 1.0).
        beta2 (Tensor): The exponential decay rate for the 2nd moment estimations. Should be in range (0.0, 1.0).
        eps (Tensor): Term added to the denominator to improve numerical stability. Should be greater than 0.
        lr (Tensor): Learning rate.
        weight_decay (Number): Weight decay. Should be equal to or greater than 0.
        param (Tensor): Parameters.
        m (Tensor): m value of parameters.
        v (Tensor): v value of parameters.
        gradient (Tensor): Gradient of parameters.
        decay_flag (bool): Applies weight decay or not.
        optim_filter (bool): Applies parameter update or not.

    Returns:
        Tensor, the new value of v after updating.
    """
    if optim_filter:
        # op_mul = P.Mul(), defined output
        op_square = P.Square()
        op_sqrt = P.Sqrt()
        op_cast = P.Cast()
        op_reshape = P.Reshape()
        op_shape = P.Shape()

        param_fp32 = op_cast(param, mstype.float32)
        m_fp32 = op_cast(m, mstype.float32)
        v_fp32 = op_cast(v, mstype.float32)
        gradient_fp32 = op_cast(gradient, mstype.float32)

        next_m = op_mul(beta1, m_fp32) + op_mul(op_cast(F.tuple_to_array((1.0,)), mstype.float32)
                                                - beta1, gradient_fp32)

        next_v = op_mul(beta2, v_fp32) + op_mul(op_cast(F.tuple_to_array((1.0,)), mstype.float32)
                                                - beta2, op_square(gradient_fp32))

        regulate_m = next_m / (_scaler_one - beta1_power)
        regulate_v = next_v / (_scaler_one - beta2_power)

        update = regulate_m / (eps + op_sqrt(regulate_v))
        if decay_flag:
            update = op_mul(weight_decay, param_fp32) + update

        update_with_lr = op_mul(lr, update)
        next_param = param_fp32 - op_reshape(update_with_lr, op_shape(param_fp32))

        next_param = F.depend(next_param, F.assign(param, op_cast(next_param, F.dtype(param))))
        next_param = F.depend(next_param, F.assign(m, op_cast(next_m, F.dtype(m))))
        next_param = F.depend(next_param, F.assign(v, op_cast(next_v, F.dtype(v))))

        return op_cast(next_param, F.dtype(param))
    return gradient


def _check_param_value(beta1, beta2, eps, prim_name):
    """Check the type of inputs."""
    validator.check_value_type("beta1", beta1, [float], prim_name)
    validator.check_value_type("beta2", beta2, [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_float_range(beta1, 0.0, 1.0, Rel.INC_NEITHER, "beta1", prim_name)
    validator.check_float_range(beta2, 0.0, 1.0, Rel.INC_NEITHER, "beta2", prim_name)
    validator.check_positive_float(eps, "eps", prim_name)


class AdamWeightDecayOp(Optimizer):
    """
    Implements the Adam algorithm to fix the weight decay. It is a complete operator, not a combination of other ops.

    Note:
        When separating parameter groups, the weight decay in each group will be applied on the parameters if the
        weight decay is positive. When not separating parameter groups, the `weight_decay` in the API will be applied
        on the parameters without 'beta' or 'gamma' in their names if `weight_decay` is positive.

        To improve parameter groups performance, the customized order of parameters can be supported.

    Args:
        params (Union[list[Parameter], list[dict]]): When the `params` is a list of `Parameter` which will be updated,
            the element in `params` must be class `Parameter`. When the `params` is a list of `dict`, the "params",
            "lr", "weight_decay" and "order_params" are the keys can be parsed.

            - params: Required. The value must be a list of `Parameter`.

            - lr: Optional. If "lr" is in the keys, the value of the corresponding learning rate will be used.
              If not, the `learning_rate` in the API will be used.

            - weight_decay: Optional. If "weight_decay" is in the keys, the value of the corresponding weight decay
              will be used. If not, the `weight_decay` in the API will be used.

            - order_params: Optional. If "order_params" is in the keys, the value must be the order of parameters and
              the order will be followed in the optimizer. There are no other keys in the `dict` and the parameters
              which in the 'order_params' must be in one of group parameters.

        learning_rate (Union[float, Tensor, Iterable, LearningRateSchedule]): A value or a graph for the learning rate.
            When the learning_rate is an Iterable or a Tensor in a 1D dimension, use the dynamic learning rate, then
            the i-th step will take the i-th value as the learning rate. When the learning_rate is LearningRateSchedule,
            use dynamic learning rate, the i-th learning rate will be calculated during the process of training
            according to the formula of LearningRateSchedule. When the learning_rate is a float or a Tensor in a zero
            dimension, use fixed learning rate. Other cases are not supported. The float learning rate must be
            equal to or greater than 0. If the type of `learning_rate` is int, it will be converted to float.
            Default: 1e-3.
        beta1 (float): The exponential decay rate for the 1st moment estimations. Default: 0.9.
            Should be in range (0.0, 1.0).
        beta2 (float): The exponential decay rate for the 2nd moment estimations. Default: 0.999.
            Should be in range (0.0, 1.0).
        eps (float): Term added to the denominator to improve numerical stability. Default: 1e-6.
            Should be greater than 0.
        weight_decay (float): Weight decay (L2 penalty). It must be equal to or greater than 0. Default: 0.0.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        tuple[bool], all elements are True.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = AdamWeightDecayOp(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01},
        ...                 {'params': no_conv_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = AdamWeightDecayOp(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The conv_params's parameters will use default learning rate of 0.1 and weight decay of 0.01.
        >>> # The no_conv_params's parameters will use learning rate of 0.01 and default weight decay of 0.0.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
   """
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0,
                 clip_norm=1.0, param_init_type=mstype.float32):
        super(AdamWeightDecayOp, self).__init__(learning_rate, params, weight_decay)
        _check_param_value(beta1, beta2, eps, self.cls_name)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.enable_init_fp16 = (param_init_type == mstype.float16)
        if self.enable_init_fp16:
            self.moments1 = self.clone_param32(prefix="adam_m", init='zeros')
            self.moments2 = self.clone_param32(prefix="adam_v", init='zeros')
            self.opt = P.FusedCastAdamWeightDecay()
        else:
            self.moments1 = self.parameters.clone(prefix="adam_m", init='zeros')
            self.moments2 = self.parameters.clone(prefix="adam_v", init='zeros')
            self.opt = P.AdamWeightDecay()
        self.hyper_map = C.HyperMap()
        self.opt.add_prim_attr("primitive_target", "CPU")

    def construct(self, gradients, clip_value):
        """AdamWeightDecayOp"""
        lr = self.get_lr()
        cond = P.GreaterEqual()(clip_value, self.clip_norm)
        global_norm = F.select(cond, clip_value, self.clip_norm)
        global_norm = P.Cast()(global_norm, mstype.float16)
        if self.is_group:
            if self.is_group_lr:
                optim_result = self.map_reverse(F.partial(_adam_opt, self.opt, global_norm,
                                                          self.beta1, self.beta2, self.eps),
                                                lr, self.weight_decay, self.parameters, self.moments1, self.moments2,
                                                gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.map_reverse(F.partial(_adam_opt, self.opt, global_norm,
                                                          self.beta1, self.beta2, self.eps, lr),
                                                self.weight_decay, self.parameters, self.moments1, self.moments2,
                                                gradients, self.decay_flags, self.optim_filter)
        else:
            optim_result = self.map_reverse(F.partial(_adam_opt, self.opt, global_norm,
                                                      self.beta1, self.beta2, self.eps, lr,
                                                      self.weight_decay), self.parameters, self.moments1, self.moments2,
                                            gradients, self.decay_flags, self.optim_filter)
        if self.use_parallel:
            self.broadcast_params(optim_result)
        return optim_result

    def clone_param32(self, prefix, init=None):
        """
        Clone the parameters in ParameterTuple element-wisely to generate a new ParameterTuple with float32 data type.
        Inputs:
            prefix (str): The prefix name of the parameters.
            init (Union[Tensor, str, numbers.Number]): Initialize the shape and dtype of the parameters.
                The definition of `init` is the same as in `Parameter` API. If `init` is 'same', the
                parameters in the new parameter tuple are the same as those in the original parameter tuple.
                Default: 'same'.
        Returns:
            Tuple, the new Parameter tuple.
        """
        new = []
        for old_param in self.parameters:
            param_init = init
            if init is None:
                param_init = old_param.init
            new_state = Parameter(initializer(param_init, shape=old_param.shape, dtype=mstype.float32))
            new_state.param_info = old_param.param_info.clone()
            new_state.is_init = False
            new_state.is_param_ps = old_param.is_param_ps
            new_state.init_in_server = old_param.init_in_server
            new_state.cache_enable = old_param.cache_enable
            new_state.requires_aggr = old_param.requires_aggr
            if old_param.cache_shape:
                new_state.cache_shape = old_param.cache_shape
            new_state.name = prefix + '.' + new_state.name
            new.append(new_state)
        return ParameterTuple(new)


class FP32StateAdamWeightDecay(AdamWeightDecay):
    r"""
        This class is almost same with the mindspore's AdamWeightDecay implements, the
        only difference is the optimizer's state will be always initialized with float32,
        where the original AdamWeightDecay will initialize the optimizer's state with float16,
        if the parameters are initialized with fp16.
        This setting will avoid overflow in training PanGu-Alpha model using fp16.
    """

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(FP32StateAdamWeightDecay, self).__init__(params, learning_rate=learning_rate,
                                                       beta1=beta1,
                                                       beta2=beta2,
                                                       eps=eps,
                                                       weight_decay=weight_decay)

        self.moments1 = self.clone_state(self.parameters, prefix='adam_m', init='zeros')
        self.moments2 = self.clone_state(self.parameters, prefix='adam_v', init='zeros')

    def clone_state(self, parameter_tuple, prefix, init):
        r"""
            parameter_tuple: ParameterTuple. The parameters of the network
            prefix: str. The prefix name of the parameters
            init: str. The initialization method
        """
        new = []
        for old_param in parameter_tuple:
            new_state = Parameter(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.param_info = old_param.param_info.clone()
            new_state.is_init = False
            new_state.set_data(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.name = prefix + '.' + new_state.name
            new.append(new_state)
        return ParameterTuple(new)


class AdamWithScale(Optimizer):
    """
    Implements the gradient clipping by norm for a AdamWeightDecay optimizer.
    """
    @opt_init_args_register
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, \
                 weight_decay=0.0, loss_scale=1.0, clip=False):
        super(AdamWithScale, self).__init__(learning_rate, params, weight_decay)
        _check_param_value(beta1, beta2, eps, self.cls_name)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = self.parameters.clone(prefix="adam_m", init='zeros')
        self.moments2 = self.parameters.clone(prefix="adam_v", init='zeros')
        self.hyper_map = C.HyperMap()
        self.beta1_power = Parameter(initializer(1, [1], mstype.float32), name="beta1_power")
        self.beta2_power = Parameter(initializer(1, [1], mstype.float32), name="beta2_power")

        self.reciprocal_scale = Tensor(1.0 / loss_scale, mstype.float32)
        self.clip = clip

    def construct(self, gradients):
        """construct"""
        lr = self.get_lr()
        gradients = scale_grad(gradients, self.reciprocal_scale)
        if self.clip:
            gradients = C.clip_by_global_norm(gradients, 5.0, None)

        beta1_power = self.beta1_power * self.beta1
        self.beta1_power = beta1_power
        beta2_power = self.beta2_power * self.beta2
        self.beta2_power = beta2_power

        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(F.partial(_adam_opt, beta1_power, beta2_power, \
                                              self.beta1, self.beta2, self.eps),
                                              lr, self.weight_decay, self.parameters, self.moments1, self.moments2,
                                              gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.hyper_map(F.partial(_adam_opt, beta1_power, beta2_power, \
                                              self.beta1, self.beta2, self.eps, lr),
                                              self.weight_decay, self.parameters, self.moments1, self.moments2,
                                              gradients, self.decay_flags, self.optim_filter)
        else:
            optim_result = self.hyper_map(F.partial(_adam_opt, beta1_power, beta2_power, self.beta1, self.beta2, \
                                          self.eps, lr, self.weight_decay),
                                          self.parameters, self.moments1, self.moments2,
                                          gradients, self.decay_flags, self.optim_filter)
        if self.use_parallel:
            self.broadcast_params(optim_result)
        return optim_result


def paramter_group(network, weight_decay, no_weight_decay_filter, gc_flag):
    """paramter_group"""
    filter_len = len(no_weight_decay_filter)
    if filter_len > 0:
        decayed_params = []
        no_decayed_params = []
        for param in network.trainable_params():
            if all([key not in param.name for key in no_weight_decay_filter]):
                decayed_params.append(param)
            else:
                no_decayed_params.append(param)

        group_params = [{'params': decayed_params, 'weight_decay': weight_decay, 'grad_centralization': gc_flag},
                        {'params': no_decayed_params},
                        {'order_params': network.trainable_params()}]
    else:
        group_params = [{'params': network.trainable_params(), \
                        'weight_decay': weight_decay, 'grad_centralization': gc_flag},
                        {'order_params': network.trainable_params()}]

    return group_params


def set_weight_decay(params):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest
    """
    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{
        'params': decay_params,
        'weight_decay': 1e-1
    }, {
        'params': other_params,
        'weight_decay': 0.0
    }, {
        'order_params': params
    }]
    return group_params


def get_optimizer(net,
                  lr,
                  optimizer_name,
                  args=None,
                  stage_num=1,
                  param_init_type=mstype.float32,
                  opt_offload=False):
    """ Get the optimizer according to the args_opt and the net"""
    if optimizer_name == "adamw":
        no_weight_decay_filter = [x for x in args.no_weight_decay_filter.split(",") if x]
        group_params = paramter_group(net, args.weight_decay, no_weight_decay_filter, bool(args.gc_flag))
        optimizer = AdamWithScale(group_params, lr, args.beta1, args.beta2, loss_scale=args.loss_scale)
        return optimizer

    params = net.trainable_params() if stage_num <= 1 else net.infer_param_pipeline_stage()

    enable_offload = bool(opt_offload)

    enable_grad_fp16 = enable_offload and param_init_type == mstype.float16

    group_params = set_weight_decay(params)

    optimizer_args = dict(learning_rate=lr, eps=1e-8, beta1=0.9, beta2=0.95)

    if optimizer_name == "lamb":
        optimizer = Lamb
    elif enable_offload:
        optimizer = AdamWeightDecayOp
        optimizer_args['param_init_type'] = param_init_type
    else:
        optimizer = FP32StateAdamWeightDecay

    class OptimizerWithClipNorm(optimizer):
        """An global norm wrapper"""
        def __init__(self, *args, **kwargs):
            super(OptimizerWithClipNorm, self).__init__(*args, **kwargs)
            self.optimizer = super(OptimizerWithClipNorm, self).construct
            self.enable_offload = enable_offload
            self.norm = ClipByGlobalNorm(enable_grad_fp16=enable_grad_fp16,
                                         clip_norm=1.0)

        def construct(self, gradients):
            grads, norm = self.norm(gradients)
            if self.enable_offload:
                return self.optimizer(grads, norm)  # pylint: disable=too-many-function-args
            return self.optimizer(grads)

    optimizer = OptimizerWithClipNorm(group_params, **optimizer_args)
    optimizer.enable_offload = enable_offload

    return optimizer
