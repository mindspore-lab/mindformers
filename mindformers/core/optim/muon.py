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
"""Muon API"""

from __future__ import absolute_import

import hashlib

import numpy as np
from mindspore.common import dtype as mstype
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common.api import jit
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.communication.management import create_group, get_rank
from mindspore.ops.auto_generate import Chunk
from mindspore import get_auto_parallel_context

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.core.context import is_legacy_model
from mindformers.tools.logger import logger

_muon_opt = C.MultitypeFuncGraph("muon_opt")


def _perform_allgather_op(ns_inputs_item, op, tp, tp_dim, op_group, tp_group, param_name):
    """Perform AllGather operations based on op and tp settings."""
    if "mlp.experts.weight" not in param_name:
        # all gather op_shard
        if op > 1:
            ns_inputs_item = P.AllGather(group=op_group)(ns_inputs_item)

        # all gather tp_shard
        if tp > 1:
            if tp_dim == 0:
                ns_inputs_item = P.AllGather(group=tp_group)(ns_inputs_item)
            elif tp_dim == 1:
                ns_inputs_item = P.AllGather(group=tp_group)(ns_inputs_item.T)
                ns_inputs_item = ns_inputs_item.T
    return ns_inputs_item


def zeropower_via_newtonschulz5_2d(x, dim_a, dim_b):
    """Apply Newton-Schulz iteration for 2D tensors."""
    a, b, c = (3.4445, -4.7750, 2.0315)

    if dim_a > dim_b:
        x = x.T
    # Ensure spectral norm is at most 1
    x = x / (x.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(5):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x
    if dim_a > dim_b:
        x = x.T
    return x


def zeropower_via_newtonschulz5_3d(x, dim_a, dim_b):
    """Apply Newton-Schulz iteration for 3D tensors."""
    a, b, c = (3.4445, -4.7750, 2.0315)

    if dim_a > dim_b:
        x = P.Transpose()(x, (0, 2, 1))
    # Ensure spectral norm is at most 1
    x = x / P.ExpandDims()(P.ExpandDims()((x.norm(dim=(1, 2)) + 1e-7), 1), 1)
    # Perform the NS iterations
    for _ in range(5):
        a_mat = P.BatchMatMul(transpose_b=True)(x, x)
        b_mat = b * a_mat + c * P.BatchMatMul()(a_mat, a_mat)
        x = a * x + P.BatchMatMul()(b_mat, x)
    if dim_a > dim_b:
        x = P.Transpose()(x, (0, 2, 1))
    return x


def _slice_tensor_to_shards(x, tp, tp_dim, op, rank_id, op_group, tp_group):
    """Slice tensor to tp_shard and op_shard."""
    # slice X to tp_shard and slice X to op_shard
    if tp > 1:
        if tp_dim >= 0:
            chunk_id = rank_id % tp
            x = Chunk()(x, tp, tp_dim)[chunk_id]

    if op > 1:
        if tp_dim == -1:
            chunk_id = rank_id % op
        else:
            chunk_id = rank_id // tp % op
        x = Chunk()(x, op)[chunk_id]
    return x


def _apply_muon_update(
    gradient, muon_m, momentum, use_nesterov, param, lr, weight_decay,
    matched_adamw_rms, muon_split_fn, muon_merge_fn, param_name,
    op, tp, tp_dim, rank_id, op_group, tp_group):
    """Apply Muon optimizer update."""
    op_sqrt = P.Sqrt()
    op_cast = P.Cast()
    op_reshape = P.Reshape()
    op_shape = P.Shape()

    m_fp32 = op_cast(muon_m, mstype.float32)
    gradient_fp32 = op_cast(gradient, mstype.float32)
    next_m = m_fp32 * momentum + gradient_fp32

    if use_nesterov:
        gradient_fp32 = gradient_fp32 + next_m * momentum
    else:
        gradient_fp32 = next_m

    ns_inputs = op_cast(gradient_fp32, mstype.bfloat16)
    ns_inputs_list = muon_split_fn(param_name, ns_inputs)
    x_list = []

    dim_a, dim_b = None, None
    for ns_inputs_item in ns_inputs_list:
        dim_a, dim_b = op_shape(ns_inputs_item)[-2:]

        if len(op_shape(ns_inputs_item)) == 2:
            ns_inputs_item = _perform_allgather_op(
                ns_inputs_item, op, tp, tp_dim, op_group, tp_group, param_name)
            x = zeropower_via_newtonschulz5_2d(ns_inputs_item, dim_a, dim_b)
            x = _slice_tensor_to_shards(x, tp, tp_dim, op, rank_id, op_group, tp_group)
        else:
            x = zeropower_via_newtonschulz5_3d(ns_inputs_item, dim_a, dim_b)

        x_list.append(x)

    x_ret = muon_merge_fn(param_name, x_list)
    param_fp32 = op_cast(param, mstype.float32)
    param_fp32 = param_fp32 * (1 - lr * weight_decay)

    adjusted_ratio = op_sqrt(op_cast(max(dim_a, dim_b), mstype.float32)) * matched_adamw_rms
    adjusted_lr = lr * adjusted_ratio
    update_with_lr = adjusted_lr * x_ret
    next_param = param_fp32 - op_reshape(update_with_lr, op_shape(param_fp32))
    next_param = F.depend(next_param, F.assign(param, op_cast(next_param, F.dtype(param))))
    next_param = F.depend(next_param, F.assign(muon_m, op_cast(next_m, F.dtype(muon_m))))
    return op_cast(next_param, F.dtype(param))


def _apply_adamw_update(param, exp_avg, exp_avg_sq, gradient, beta1, beta2, step, eps, lr, weight_decay):
    """Apply AdamW optimizer update."""
    op_mul = P.Mul()
    op_pow = P.Pow()
    op_sqrt = P.Sqrt()
    op_cast = P.Cast()
    addcmul = P.Addcmul()

    param_fp32 = op_cast(param, mstype.float32)
    next_param = op_mul(param_fp32, 1 - lr * weight_decay)
    gradient_fp32 = op_cast(gradient, mstype.float32)

    next_param = F.depend(
        next_param,
        F.assign(
            exp_avg,
            op_mul(exp_avg, beta1)
            + op_mul(gradient_fp32, op_cast(F.tuple_to_array((1.0,)), mstype.float32) - beta1),
        ),
    )
    next_param = F.depend(
        next_param,
        F.assign(
            exp_avg_sq,
            addcmul(
                op_mul(exp_avg_sq, beta2),
                gradient_fp32,
                gradient_fp32,
                op_cast(F.tuple_to_array((1.0,)), mstype.float32) - beta2,
            ),
        ),
    )

    bias_correction1 = 1 - op_pow(op_cast(beta1, mstype.float32), step)
    bias_correction2 = 1 - op_pow(op_cast(beta2, mstype.float32), step)
    step_size = lr / bias_correction1
    denom = op_sqrt(exp_avg_sq / bias_correction2) + eps
    return_param = next_param - op_mul(exp_avg / denom, step_size)
    F.assign(param, op_cast(return_param, F.dtype(param)))
    return op_cast(return_param, F.dtype(param))


@_muon_opt.register(
    "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Number",
    "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Number", "Number", "Number", "Bool",
    "Bool", "Bool", "String", "String", "String", "Function", "Function")
def _update_run_op(
    momentum, matched_adamw_rms, beta1, beta2, step, eps, lr, weight_decay, rank_id,
    param, exp_avg, exp_avg_sq, gradient, muon_m, tp, op, tp_dim, use_muon,
    use_nesterov, optim_filter, op_group, tp_group, param_name, muon_split_fn, muon_merge_fn):
    """
    Update parameters.

    Args:
        beta1 (Tensor): The exponential decay rate for the 1st moment estimations. Should be in range (0.0, 1.0).
        beta2 (Tensor): The exponential decay rate for the 2nd moment estimations. Should be in range (0.0, 1.0).
        eps (Tensor): Term added to the denominator to improve numerical stability. Should be greater than 0.
        lr (Tensor): Learning rate.
        weight_decay (numbers.Number): Weight decay. Should be equal to or greater than 0.
        param (Tensor): Parameters.
        m (Tensor): m value of parameters.
        v (Tensor): v value of parameters.
        gradient (Tensor): Gradient of parameters.
        decay_flag (bool): Applies weight decay or not.
        optim_filter (bool): Applies parameter update or not.

    Returns:
        Tensor, the new value of v after updating.
    """
    op_cast = P.Cast()
    if "max_logits_val" in param_name:
        return op_cast(gradient, F.dtype(param))

    if not optim_filter:
        return gradient

    if use_muon:
        return _apply_muon_update(
            gradient, muon_m, momentum, use_nesterov, param, lr, weight_decay,
            matched_adamw_rms, muon_split_fn, muon_merge_fn, param_name,
            op, tp, tp_dim, rank_id, op_group, tp_group)

    return _apply_adamw_update(param, exp_avg, exp_avg_sq, gradient, beta1, beta2, step, eps, lr, weight_decay)


@MindFormerRegister.register(MindFormerModuleType.OPTIMIZER)
class Muon(Optimizer):
    """
    Muon optimizer implementation.

    Args:
        params: model parameters to optimize.
        learning_rate (float): Learning rate. Default: ``2e-2``.
        weight_decay (float): Weight decay factor. Default: ``0.1``.
        matched_adamw_rms (float): RMS matching parameter for AdamW. Default: ``0.2``.
        momentum (float): Momentum factor. Default: ``0.95``.
        nesterov (bool): Whether to use Nesterov momentum. Default: ``True``.
        ns_steps (int): Number of Newton-Schulz steps. Default: ``5``.
        adamw_betas (tuple): Beta parameters for AdamW. Default: ``(0.95, 0.95)``.
        adamw_eps (float): Epsilon for AdamW. Default: ``1e-8``.
        qk_clip_threshold (float): QK clip threshold. Default: ``100``.
        model: The model model. Default: ``None``.
    """

    def __init__(
        self,
        params,
        learning_rate=2e-2,
        weight_decay=0.1,
        matched_adamw_rms=0.2,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        qk_clip_threshold=100,
        model=None,
        **kwargs,
    ):
        super().__init__(learning_rate, params, weight_decay)
        if kwargs.get('swap', False):
            raise ValueError("Muon does not support swap.")

        self._verify_model(model)

        # Initialize basic parameters
        self._initialize_basic_params(adamw_betas, adamw_eps, momentum, matched_adamw_rms, nesterov)

        # Initialize model configuration
        self._initialize_network_config(model)

        # Initialize parameter layers
        self._initialize_param_layers(model)

        # Initialize QK-clip parameters
        self.ones = Tensor([1.0], mstype.float32)
        self.rank_id = get_rank()
        self.rank_ids = tuple(self.rank_id for _ in self._parameters)
        self.logit_threshold = Tensor([qk_clip_threshold], dtype=mstype.float32)

        # Initialize Muon momentum
        self._initialize_muon_moments(model)

        # Initialize tensor parallel dimensions
        self._initialize_tp_dims(model)

        # Initialize AdamW moments
        self._initialize_adamw_moments(model)

        # Initialize parallel configuration
        self._initialize_parallel_config(model)

        # Initialize communication groups
        self._initialize_communication_groups()

        # Initialize optimizer parallel groups
        self._initialize_op_groups(model)

        # Store model for QK-clip
        self.model = model
        self.ns_steps = ns_steps

    def _verify_model(self, model):
        """Verify if the model is compatible with Muon optimizer."""
        if model is None:
            raise ValueError("Model must be provided for Muon optimizer.")

        if is_legacy_model():
            raise ValueError("Muon does not support Legacy Model.")

        config = model.get_gpt_transformer_config()

        if not config.multi_latent_attention:
            raise ValueError("Current Muon implementation only supports models with Multi-Latent Attention enabled.")

    def _initialize_basic_params(self, adamw_betas, adamw_eps, momentum, matched_adamw_rms, nesterov):
        """Initialize basic optimizer parameters."""
        self.beta1 = Tensor(np.array([adamw_betas[0]]).astype(np.float32))
        self.beta2 = Tensor(np.array([adamw_betas[1]]).astype(np.float32))
        self.eps = Tensor(np.array([adamw_eps]).astype(np.float32))
        self.muon_momentum = Tensor(np.array([momentum]).astype(np.float32))
        self.matched_adamw_rms = Tensor(np.array([matched_adamw_rms]).astype(np.float32))
        self.use_nesterov = tuple(nesterov for _ in self._parameters)
        self.param_name_tuple = tuple(p.name for p in self._parameters)

    def _initialize_network_config(self, model):
        """Initialize Model configuration and split/merge functions."""

        self.muon_split_fn, self.muon_merge_fn = model.make_model_muon_fns()
        self.muon_split_fns = tuple(self.muon_split_fn for _ in self._parameters)
        self.muon_merge_fns = tuple(self.muon_merge_fn for _ in self._parameters)

    def _initialize_param_layers(self, model):
        """Initialize parameter layer indices."""
        self.param_layer = model.get_param_layer_indices(self._parameters)

    def _initialize_muon_moments(self, model):
        """Initialize Muon momentum parameters."""
        muon_filter = model.get_muon_filter()

        self.muon_m = []
        self.param_idx_in_opt = {}
        for idx, param in enumerate(self._parameters):
            self.param_idx_in_opt[param.name] = idx

        for param in self._parameters:
            if muon_filter(param):
                x1 = param.clone("zeros")
                x1.name = "muon_m" + "." + x1.name
                self.muon_m.append(x1)
                logger.info(f"Muon apply: {param}")
            else:
                self.muon_m.append(Parameter(Tensor(np.array([0]).astype(np.float32)), name="muon_m." + param.name))
        self.muon_m = ParameterTuple(self.muon_m)
        self.use_muon = tuple(muon_filter(param) for param in self._parameters)

    def _initialize_tp_dims(self, model):
        """Initialize tensor parallel dimensions."""
        self.tp_dims = model.get_tp_dims(self._parameters)

    def _initialize_adamw_moments(self, model):
        """Initialize AdamW momentum parameters."""
        muon_filter = model.get_muon_filter()

        self.moments1 = []
        self.moments2 = []
        for param in self._parameters:
            if not muon_filter(param):
                x1 = param.clone("zeros")
                x1.name = "adam_m" + "." + x1.name
                self.moments1.append(x1)
                x2 = param.clone("zeros")
                x2.name = "adam_v" + "." + x2.name
                self.moments2.append(x2)
                logger.info(f"Adam apply: {param}")
            else:
                self.moments1.append(Parameter(Tensor(np.array([0]).astype(np.float32)), name="adam_m." + param.name))
                self.moments2.append(Parameter(Tensor(np.array([0]).astype(np.float32)), name="adam_v." + param.name))
        self.moments1 = ParameterTuple(self.moments1)
        self.moments2 = ParameterTuple(self.moments2)

    def _initialize_parallel_config(self, model):
        """Initialize parallel configuration."""
        self.tp = model.get_gpt_transformer_config().tensor_model_parallel_size
        self.tps = tuple(self.tp for _ in self._parameters)
        self.dp = model.get_gpt_transformer_config().data_parallel_size
        logger.info(f"Muon tp group size is: {self.tp}")

        if not get_auto_parallel_context('enable_parallel_optimizer'):
            self.op = 1
        else:
            self.op = get_auto_parallel_context('optimizer_weight_shard_size')
            if self.op < 1:
                raise ValueError(
                    "Must set parallel.parallel_optimizer_config.optimizer_weight_shard_size > 1 "
                    "when enable_parallel_optimizer is True.")
            if self.dp < self.op:
                raise ValueError('Must set parallel_config.data_parallel >= '
                                 'parallel.parallel_optimizer_config.optimizer_weight_shard_size when using Muon.')
        logger.info(f"Muon op group size is: {self.op}")

    def _initialize_communication_groups(self):
        """Initialize communication groups for parallel training."""
        self.tp_group = self._get_tp_group_name(self.rank_id, self.tp)
        self.op_group, self.op_in_tp_group = self._get_op_group_name(self.rank_id, self.tp, self.op, self.tp_group)
        self.tp_groups = tuple(self.tp_group for _ in self._parameters)

    def _initialize_op_groups(self, model):
        """Initialize optimizer parallel groups for parameters."""
        self.ops, self.op_groups = model.get_op_groups_info(self._parameters, self.op)

    def _create_communication_group(self, rank_list):
        """
        Create a communication group with a hashed name.
        
        Args:
            rank_list: List of ranks in the communication group
        
        Returns:
            str: The created group name
        """
        rank_list_str = "-".join([str(i) for i in rank_list])
        hashed = hashlib.md5(rank_list_str.encode()).hexdigest()[:48]
        group_name = str(hashed)
        create_group(group_name, rank_list)
        return group_name

    def _get_op_group_name(self, rank_id, tp, op, tp_group):
        """
        Generates a unique group name for optimizer parallel communication group.
        
        Returns:
            tuple: The optimizer group name and optimizer-in-tensor-parallel group name
        """
        dp_range = tp
        op_range = tp * op
        rank_start = rank_id % dp_range + rank_id // op_range * op_range
        rank_end = rank_start + op_range
        rank_list = list(range(rank_start, rank_end, dp_range))
        logger.info(f"Muon op group list is: {rank_list}")
        op_group_name = self._create_communication_group(rank_list)

        if tp == op:
            logger.info(
                f"op_in_tp group will reuse tp group" \
                f", since tensor_parallel_size({tp}) == optimizer_parallel_size({op})."
            )
            op_in_tp_group_name = tp_group
        else:
            logger.info(f"Muon op_in_tp group list is: {rank_list}")
            op_in_tp_group_name = self._get_tp_group_name(rank_id, op)

        return op_group_name, op_in_tp_group_name

    def _get_tp_group_name(self, rank_id, tp):
        """
        Generates a unique group name for tensor parallel communication group.
        
        Returns:
            str: The tensor parallel group name
        """
        rank_start = rank_id // tp * tp
        rank_end = rank_id // tp * tp + tp
        rank_list = list(range(rank_start, rank_end))
        logger.info(f"Muon tp group list is: {rank_list}")
        tp_group_name = self._create_communication_group(rank_list)
        return tp_group_name

    def _hyper_map_func(self, lr, weight_decay, gradients):
        """
        Apply Muon optimizer update using hyper_map across parameter structures.
        """
        hyper_map_args = [
            self.rank_ids,
            self._parameters,
            self.moments1,
            self.moments2,
            gradients,
            self.muon_m,
            self.tps,
            self.ops,
            self.tp_dims,
            self.use_muon,
            self.use_nesterov,
            self.optim_filter,
            self.op_groups,
            self.tp_groups,
            self.param_name_tuple,
            self.muon_split_fns,
            self.muon_merge_fns
        ]

        if self.is_group:
            # If parameters are divided into groups (group-wise hyperparams)
            if self.is_group_lr:
                # Case 1: Both learning rate and weight decay are grouped
                partial_func = F.partial(
                    _muon_opt, self.muon_momentum, self.matched_adamw_rms,
                    self.beta1, self.beta2, self.global_step, self.eps
                )
                hyper_map_args = [lr, weight_decay] + hyper_map_args
            else:
                # Case 2: Only weight decay is grouped, lr is global
                partial_func = F.partial(
                    _muon_opt, self.muon_momentum, self.matched_adamw_rms,
                    self.beta1, self.beta2, self.global_step, self.eps, lr
                )
                hyper_map_args = [weight_decay] + hyper_map_args
        else:
            # No parameter groups: lr and weight decay are global hyperparameters
            partial_func = F.partial(
                _muon_opt, self.muon_momentum, self.matched_adamw_rms,
                self.beta1, self.beta2, self.global_step, self.eps, lr, weight_decay
            )

        return self.hyper_map(partial_func, *hyper_map_args)

    @jit(backend="ms_backend")
    def construct(self, gradients):
        """Construct method for optimizer.

        Args:
            gradients: Gradients for optimization.

            Returns:
            Updated gradients after optimization.
        """
        gradients = self.flatten_gradients(gradients)
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)
        optim_result = self._hyper_map_func(
            lr,
            weight_decay,
            gradients,
        )

        updates = self.model.apply_qk_clip_scaling(
            self._parameters,
            self.param_name_tuple,
            self.param_layer,
            self.logit_threshold,
            self.muon_split_fn,
            self.muon_merge_fn,
        )

        # Apply the weight updates
        for param_idx, weights in updates:
            optim_result = F.depend(optim_result, F.assign(self._parameters[param_idx], weights))

        return optim_result
