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
"""Self-Define Wrapper."""
import hashlib
import os
import shutil
from copy import deepcopy

from mindspore._checkparam import args_type_check
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.parallel._utils import _get_enable_parallel_optimizer
import mindspore.common.dtype as mstype
from mindspore import nn, Parameter, ParallelMode, get_auto_parallel_context
from mindspore.common import RowTensor
from mindspore.common.tensor import Tensor
from mindspore.communication.management import (create_group, get_rank, get_group_size)
from mindspore.nn.wrap.cell_wrapper import _MicroBatch
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.auto_generate import SumExt
from mindspore.ops._grad_experimental.grad_comm_ops import get_squared_device_local_norm_param

from mindformers.core.clip_grad import ClipGradNorm
from mindformers.core.context.build_context import is_legacy_model
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.utils import get_real_rank
from mindformers.utils.parameter_register import parameter_register
from mindformers.version_control import get_identity

__all__ = ['MFTrainOneStepCell', 'MFPipelineWithLossScaleCell', 'PipelineCellWithTwoOutput',
           'GradAccumulationCellWithTwoOutput']

_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


# pylint: disable=W0212
def get_real_models(network):
    """
    Extracts the core model from a potentially nested network wrapper.
    This function traverses through nested 'network' attributes and, if present,
    the '_backbone' attribute to retrieve the underlying real model object.
    Args:
        network (object): The network or model object, possibly wrapped.
    Returns:
        object: The innermost, real model object after unwrapping.
    """
    while hasattr(network, 'network'):
        network = network.network
    if hasattr(network, '_backbone'):
        network = network._backbone
    while hasattr(network, 'network'):
        network = network.network
    return network


def get_aux_loss_scale(moe_aux_loss_coeff, moe_layer_freq, num_layers):
    """
    Calculate the total auxiliary loss scale for Mixture of Experts (MoE) layers.

    Args:
        moe_aux_loss_coeff (float): Coefficient for the auxiliary loss.
        moe_layer_freq (int, list, or None): Frequency or pattern of MoE layers.
        num_layers (int): Total number of layers in the transformer.

    Returns:
        float: The total auxiliary loss scale for all MoE layers.

    Raises:
        ValueError: If `moe_layer_freq` is not an int, list, or None.
    """
    # Get number of MoE layers
    if moe_layer_freq is None:
        num_moe_layers = num_layers
    elif isinstance(moe_layer_freq, int):
        moe_layer_pattern = [1 if (i % moe_layer_freq == 0) else 0 for i in range(num_layers)]
        num_moe_layers = sum(moe_layer_pattern)
    elif isinstance(moe_layer_freq, list):
        num_moe_layers = sum(moe_layer_freq)
    else:
        raise ValueError(f"Invalid moe_layer_freq: {moe_layer_freq}")

    return num_moe_layers * moe_aux_loss_coeff


def _reset_accu_gbs_fi(network):
    """
    Reset the accumulated FI tensor of each MoE router back to zeros.

    This method traverses down through nested `network` objects until it reaches
    the backbone. For each layer that contains a `router` module, its
    `fi_accu` parameter is reassigned to a zero tensor.

    Args:
        network (nn.Cell): Training network backbone containing decoder layers.
    """
    model = network.get_gpt_model()
    if hasattr(model, "reset_accu_gbs_fi"):
        model.reset_accu_gbs_fi()
    else:
        raise NotImplementedError(f"network: {network} does not Implemented function `reset_accu_gbs_fi`")


def _check_network_with_micro_size(cls_name, network, micro_size):
    """
    Check the validity of network and micro_size parameters for cls_name
    """
    if not isinstance(network, nn.Cell):
        raise TypeError(f"For `{cls_name}`, the argument 'network' must cell type, "
                        f"but got the type : {type(network)}.")
    if not isinstance(micro_size, int):
        raise TypeError(f"For `{cls_name}`, the argument 'micro_size' must be integer, "
                        f"but got the type : {type(micro_size)}.")
    if micro_size <= 0:
        raise ValueError(f"For `{cls_name}`, the argument 'micro_size' must be large than 0, "
                         f"but got {micro_size}.")


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return F.cast(grad, mstype.float32) * reciprocal(scale)


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices,
                     grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
                     grad.dense_shape)


get_square_sum = C.MultitypeFuncGraph("get_square_sum")
get_size = C.MultitypeFuncGraph("get_size")


@get_square_sum.register("Tensor", "Number")
def _get_square_sum(grad, value):
    norm = SumExt()(F.square(F.cast(grad, mstype.float32)), ()) * value
    norm = F.expand_dims(norm, 0)
    return norm


# pylint: disable=E0102
@get_square_sum.register("Tensor")
def _get_square_sum(grad):
    norm = SumExt()(F.square(F.cast(grad, mstype.float32)), ())
    norm = F.expand_dims(norm, 0)
    return norm


@get_size.register("Tensor")
def _get_size(grad):
    size = P.Size()(grad)
    return size


class LocalNorm(nn.Cell):
    def __init__(self):
        super().__init__()
        self.hyper_map = C.HyperMap()

    def construct(self, grads):
        square_sum = self.hyper_map(get_square_sum, grads)
        size = self.hyper_map(get_size, grads)
        return square_sum, size


# pylint: disable=W1401
@MindFormerRegister.register(MindFormerModuleType.WRAPPER)
class MFTrainOneStepCell(nn.TrainOneStepWithLossScaleCell):
    """TrainOneStep For MindFormer.
    Network training with loss scaling, grad clip, gradient accumulation, exponential moving average and so on.

    This is a training step with loss scaling. It takes a network, an optimizer and a scale update Cell(or a Tensor) as
    args. The loss scale value can be updated in both host side or device side. If you want to update it on
    host side, using a value of Tensor type as `scale_sense`, otherwise, using a Cell instance for updating loss
    scale as `scale_sense`.

    Args:
        network (Cell): The training network. The network only supports single output.
        optimizer (Cell): Optimizer for updating the network parameters.
        use_clip_grad (bool, optional): Whether to use the gradient clipping function. Default: ``False``.
        max_grad_norm (float, optional): Maximum gradient value. Default: ``1.0``.
        scale_sense (Union[int, float, Tensor, Cell], optional): The scaling number to be filled as the input of
            backpropagation. If this value is a Cell, it will be called by `MFTrainOneStepCell` to update loss scale.
            If this value is a Tensor, the loss scale can be modified by `set_sense_scale`,
            the shape should be :math:`()` or :math:`(1,)`. Default: ``1.0``.
        local_norm (bool, optional): Whether to calculate the local norm. Default: ``False``.
        calculate_per_token_loss (bool, optional): Whether to calculate the loss of each token.
            Default: `False`.
        global_norm_spike_threshold (float, optional): The threshold of global norm when
            the skip data function is enabled. Default: ``1.0``.
        use_skip_data_by_global_norm (bool, optional): The switch of the skip data function. Default: ``False``.
        print_separate_loss (bool, optional): Whether to print loss separately. Default: ``False``.
        **kwargs (Any): Additional parameters.

    Inputs:
        - **\*inputs** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Tuple of 5 or 7 Tensor, the loss, overflow flag, current loss scale value, learning rate,
        global grads norm, local grads norm and size of local norm grads.

        - **loss** (Tensor) -  A scalar, the loss value.
        - **overflow** (Tensor) -  A scalar, whether overflow occur or not, the type is bool.
        - **loss scale** (Tensor) -  The loss scale value, the shape is :math:`()` or :math:`(1,)`.
        - **learning rate** (Tensor) -  A scalar, the learning rate of the optimizer.
        - **global norm** (Tensor) -  A scalar, the global norm of all grads, only be calculated
          when `use_clip_grad=True`, otherwise None.
        - **local_norm** (Tensor) -  The local norm of the grads by group, only be returned when `local_norm=True`.
        - **size** (Tensor) -  The sizes of each grads group, only be returned when `local_norm=True`.

    Raises:
        TypeError: If `scale_sense` is neither Cell nor Tensor.
        ValueError: If shape of `scale_sense` is neither (1,) nor ().

    Examples:
        >>> from mindformers.models.llama import LlamaConfig, LlamaForCausalLM
        >>> from mindformers.wrapper import MFTrainOneStepCell
        >>> import mindspore as ms
        >>> from mindformers.core.optim import AdamW
        >>> import numpy as np
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>>
        >>> config = LlamaConfig(num_layers=2)
        >>> net = LlamaForCausalLM(config=config)
        >>> net.set_train(True)
        >>> optimizer = AdamW(net.trainable_params())
        >>>
        >>> mft = MFTrainOneStepCell(net, optimizer)
        >>> inputs = ms.Tensor(np.ones([1, 2049]), ms.int32)
        >>> out = mft(inputs)
        >>>
        >>> loss, overflow, loss_scale, lr, global_norm = out
        >>> print(loss.shape, overflow, loss_scale, lr, global_norm)
        (1,) False 1.0 0.001, None
    """

    @args_type_check(global_norm_spike_threshold=float, use_skip_data_by_global_norm=bool)
    def __init__(self,
                 network,
                 optimizer,
                 use_clip_grad=False,
                 max_grad_norm=1.0,
                 scale_sense=1.0,
                 local_norm=False,
                 calculate_per_token_loss=False,
                 global_norm_spike_threshold=1.0,
                 use_skip_data_by_global_norm=False,
                 print_separate_loss=False,
                 **kwargs):
        if isinstance(scale_sense, (int, float)):
            scale_sense = Tensor(scale_sense)
        super().__init__(network, optimizer, scale_sense)
        self.use_clip_grad = use_clip_grad
        self.clip_grad_norm = ClipGradNorm(max_norm=max_grad_norm)
        self.parallel_config = kwargs.pop("parallel_config", None)
        self.learning_rate = deepcopy(self.optimizer.learning_rate)
        self.localnorm = LocalNorm()
        self.concat = P.Concat()
        self.cast = P.Cast()
        self.local_norm = local_norm
        self.calculate_per_token_loss = calculate_per_token_loss
        self.zero_t = Tensor([0], dtype=mstype.float32)
        self.grad_scale_factor = Tensor([1], dtype=mstype.float32)
        self.dump_device_local_norm = get_auto_parallel_context("dump_device_local_norm")
        self.if_dump = bool(get_auto_parallel_context("dump_local_norm_path"))

        if self.if_dump:
            self.dump = P.TensorDump()
            self.dump_path = os.path.join(get_auto_parallel_context("dump_local_norm_path"), f"rank_{get_real_rank()}")
            if os.path.exists(self.dump_path):
                shutil.rmtree(self.dump_path)
            os.makedirs(self.dump_path, exist_ok=True)
            self.device_local_norm_filename = os.path.join(self.dump_path, "device_local_norm")
            self.finish_step_filename = os.path.join(self.dump_path, "finish_step")
        if self.dump_device_local_norm:
            self.squared_device_local_norm = get_squared_device_local_norm_param()
        self.global_norm_spike_threshold = global_norm_spike_threshold
        self.use_skip_data_by_global_norm = use_skip_data_by_global_norm

        # loss
        self.use_legacy = is_legacy_model()
        self.print_separate_loss = bool(print_separate_loss and not self.use_legacy)

        real_network = get_real_models(self.network)
        transformer_config = real_network.get_gpt_transformer_config() if not self.use_legacy else None

        if self.print_separate_loss:
            self.aux_loss_parameter = parameter_register.register("aux_loss", Tensor([0], mstype.float32))
            self.mtp_loss_parameter = parameter_register.register("mtp_loss", Tensor([0], mstype.float32))
            self.lm_loss_parameter = parameter_register.register("lm_loss", Tensor([0], mstype.float32))
            self.aux_loss_scale = get_aux_loss_scale(
                transformer_config.moe_aux_loss_coeff,
                transformer_config.moe_layer_freq,
                transformer_config.num_layers
            )

        # Get grouped LR schedulers from base_trainer
        grouped_lr_scheduler = kwargs.get('grouped_lr_scheduler')
        self.use_grouped_lr = grouped_lr_scheduler is not None
        if self.use_grouped_lr:
            # Register default_lr
            self.default_lr = kwargs.get('lr_scheduler')
            self.default_lr_parameter = parameter_register.register(
                "current_default_lr", Tensor(0., mstype.float32))
            # Register grouped_lr
            self.grouped_lr = nn.CellList(
                [deepcopy(current_lr.get('lr_scheduler')) for current_lr in grouped_lr_scheduler])
            self.grouped_lr_parameter = parameter_register.register(
                "current_grouped_lr", Tensor([0.] * len(self.grouped_lr), mstype.float32))

    def construct(self, *inputs):
        """forward and backward."""
        scaling_sens = self.scale_sense
        if self.dump_device_local_norm:
            _ = F.assign(self.squared_device_local_norm, Tensor(0.0, mstype.float32))
            scaling_sens = F.depend(self.scale_sense, _)

        if self.use_legacy:
            loss, grads, grad_scale_factor = self.grads_for_legacy(scaling_sens, *inputs)
        else:
            loss, grads, grad_scale_factor = self.grads_for_mcore(scaling_sens, *inputs)

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens * grad_scale_factor), grads)

        if self.local_norm:
            local_norm, size = self.localnorm(grads)
            local_norm = self.concat(local_norm)

        # apply grad reducer on grads
        grads = self.grad_reducer(grads)

        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)

        global_norm = None
        if self.use_clip_grad:
            grads, global_norm = self.clip_grad_norm(grads)

        learning_rate = self.learning_rate
        if self.optimizer.dynamic_lr:
            if self.use_grouped_lr:
                # Get default_lr, grouped_lr and update parameters registered
                learning_rate = self.default_lr(self.optimizer.global_step).reshape(())
                self.default_lr_parameter = learning_rate
                for group_id, current_lr in enumerate(self.grouped_lr):
                    self.grouped_lr_parameter[group_id] = current_lr(self.optimizer.global_step).reshape(())
            elif self.optimizer.is_group_lr:
                learning_rate = self.learning_rate[-1](self.optimizer.global_step).reshape(())
            else:
                learning_rate = self.learning_rate(self.optimizer.global_step).reshape(())

        if self.use_graceful_exit:
            grads = self.graceful_exit.exit_by_request(grads, self.init_param, self.exit_param)

        # if there is no overflow, do optimize
        if not overflow:
            if self.use_skip_data_by_global_norm:
                if global_norm < self.global_norm_spike_threshold:
                    loss = F.depend(loss, self.optimizer(grads))
            else:
                loss = F.depend(loss, self.optimizer(grads))

        if self.if_dump:
            zero = Tensor(0.0, mstype.float32)
            if self.dump_device_local_norm:
                zero = F.depend(zero, self.dump(self.device_local_norm_filename,
                                                F.sqrt(self.squared_device_local_norm)))
            filename = self.finish_step_filename
            self.dump(filename, zero)
        if self.local_norm:
            return loss, overflow, scaling_sens, learning_rate, global_norm, local_norm, size
        return loss, overflow, scaling_sens, learning_rate, global_norm

    def grads_for_legacy(self, scaling_sens, *inputs):
        """calculate grad for legacy model"""
        if self.calculate_per_token_loss:
            numerator, denominator = self.network(*inputs)
            loss = numerator / denominator

            scaling_sens_filled = C.ones_like(numerator) * F.cast(scaling_sens, F.dtype(numerator))
            scaling_sens_filled2 = self.zero_t * F.cast(scaling_sens, F.dtype(denominator))
            grads = self.grad(self.network, self.weights)(*inputs,
                                                          (self.cast(scaling_sens_filled, mstype.float32),
                                                           self.cast(scaling_sens_filled2, mstype.float32)))
            grad_scale_factor = denominator
        else:
            loss = self.network(*inputs)
            scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
            grads = self.grad(self.network, self.weights)(*inputs, scaling_sens_filled)
            grad_scale_factor = self.grad_scale_factor
        return loss, grads, grad_scale_factor

    def grads_for_mcore(self, scaling_sens, *inputs):
        """calculate grad for mcore model"""
        if self.calculate_per_token_loss:
            numerator0, denominator0, numerator1, _, aux_loss = self.network(*inputs)
            lm_loss = numerator0 / denominator0
            mtp_loss = numerator1 / denominator0
            aux_loss = aux_loss / denominator0
            loss = lm_loss + aux_loss
            loss = loss + mtp_loss

            scaling_sens_filled = C.ones_like(numerator0) * F.cast(scaling_sens, F.dtype(numerator0))
            scaling_sens_filled_zero = self.zero_t * F.cast(scaling_sens, F.dtype(denominator0))
            grads = self.grad(self.network, self.weights)(*inputs,
                                                          (self.cast(scaling_sens_filled, mstype.float32),
                                                           self.cast(scaling_sens_filled_zero, mstype.float32),
                                                           self.cast(scaling_sens_filled, mstype.float32),
                                                           self.cast(scaling_sens_filled_zero, mstype.float32),
                                                           self.cast(scaling_sens_filled, mstype.float32),
                                                           ))
            grad_scale_factor = denominator0
        else:
            lm_loss, mtp_loss, aux_loss = self.network(*inputs)
            loss = lm_loss + aux_loss
            loss = loss + mtp_loss

            scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
            grads = self.grad(self.network, self.weights)(*inputs, (scaling_sens_filled,
                                                                    scaling_sens_filled,
                                                                    scaling_sens_filled))
            grad_scale_factor = self.grad_scale_factor

        if self.print_separate_loss:
            F.assign_add(self.aux_loss_parameter, aux_loss / self.aux_loss_scale if self.aux_loss_scale > 0 else 0)
            F.assign_add(self.mtp_loss_parameter, mtp_loss)
            F.assign_add(self.lm_loss_parameter, lm_loss)
        return loss, grads, grad_scale_factor


class DataOrderWrapperCell(nn.Cell):
    """For passing parameters in lexicographical order."""

    def __init__(self, construct_args_key, network):
        super().__init__(auto_prefix=False)
        self.construct_args_key = construct_args_key
        self.network = network

    def construct(self, *inputs):
        """The construct processes of inputs in lexicographical order."""
        key_inputs = {}
        for key, val in zip(self.construct_args_key, inputs):
            key_inputs[key] = val
        return self.network(**key_inputs)


grad_scale = C.MultitypeFuncGraph("grad_scale")
shard_grad_scale = C.MultitypeFuncGraph("shard_grad_scale")


@grad_scale.register("Tensor", "Tensor", "Tensor", "Tensor")
def tensor_grad_scale_pipeline(scale, grad_scale_factor, grad, accu_grad):
    accu_grad = F.depend(accu_grad, grad)
    new_grad = accu_grad * reciprocal(scale)
    accu_grad = F.depend(accu_grad, new_grad)
    grad_val = F.cast(F.equal(accu_grad, accu_grad), F.dtype(accu_grad))
    zeros = F.mul(grad_val, 0)
    new_grad = F.depend(new_grad, F.assign(accu_grad, zeros))
    new_grad = new_grad * F.cast(reciprocal(grad_scale_factor), F.dtype(new_grad))
    return new_grad


@shard_grad_scale.register("Tensor", "Tensor", "Tensor", "Tensor")
def tensor_shard_grad_scale_pipeline(scale, grad_scale_factor, grad, accu_grad):
    new_grad = grad * F.cast(reciprocal(scale), F.dtype(grad))
    accu_grad = F.depend(accu_grad, new_grad)
    grad_val = F.cast(F.equal(accu_grad, accu_grad), F.dtype(accu_grad))
    zeros = F.mul(grad_val, 0)
    new_grad = F.depend(new_grad, F.assign(accu_grad, zeros))
    new_grad = new_grad * F.cast(reciprocal(grad_scale_factor), F.dtype(new_grad))
    return new_grad


class GradAccumulationCellWithTwoOutput(nn.Cell):
    """
        Wrap the network with Micro Batch to enable the grad accumulation in semi_auto_parallel/auto_parallel mode.

        Note:
            micro_size must be greater or equal to pipeline stages.

        Args:
            network (Cell): The target network to wrap.
            micro_size (int): MicroBatch size.

        Supported Platforms:
            ``Ascend`` ``GPU``
    """

    def __init__(self, network, micro_size):
        super().__init__(auto_prefix=False)
        self.network = network
        self.micro_inputs = nn.CellList()
        self.micro_size = micro_size
        self.add_list = []
        _check_network_with_micro_size('GradAccumulationCellWithTwoOutput', network, micro_size)
        for i in range(micro_size):
            micro_input = _MicroBatch(micro_size)
            micro_input.strided_slice.add_prim_attr("grad_accu_num", micro_size)
            self.micro_inputs.append(micro_input)
            self.add = P.Add().add_prim_attr("forward_end", i)
            self.add_list.append(self.add)
        self._get_attr_from_cell(network)

    def construct(self, *inputs):
        """Construct function for pipeline with multiple outputs."""
        ret = None
        ret2 = None
        for i in range(self.micro_size):
            micro_input = self.micro_inputs[i](i, *inputs)
            output1, output2 = self.network(*micro_input)

            if ret is not None:
                ret = self.add_list[i](ret, output1)
            else:
                ret = output1

            if ret2 is not None:
                ret2 = self.add_list[i](ret2, output2)
            else:
                ret2 = output2

        loss = ret, ret2
        return loss


class GradAccumulationCellWithMultiOutputs(nn.Cell):
    """
        Wrap the network with Micro Batch to enable the grad accumulation in semi_auto_parallel/auto_parallel mode.

        Note:
            micro_size must be greater or equal to pipeline stages.

        Args:
            network (Cell): The target network to wrap.
            micro_size (int): MicroBatch size.

        Supported Platforms:
            ``Ascend`` ``GPU``
    """

    def __init__(self, network, micro_size):
        super().__init__(auto_prefix=False)
        self.network = network
        self.micro_inputs = nn.CellList()
        self.micro_size = micro_size
        self.add_list = []
        _check_network_with_micro_size('GradAccumulationCellWithMultiOutputs', network, micro_size)
        for i in range(micro_size):
            micro_input = _MicroBatch(micro_size)
            micro_input.strided_slice.add_prim_attr("grad_accu_num", micro_size)
            self.micro_inputs.append(micro_input)
            self.add = P.Add().add_prim_attr("forward_end", i)
            self.add_list.append(self.add)
        self._get_attr_from_cell(network)

        # reset for gbs_aux_loss
        real_network = get_real_models(self.network)
        self.use_legacy = is_legacy_model()
        transformer_config = real_network.get_gpt_transformer_config() if not self.use_legacy else None
        self.is_reset_accu_gbs_fi = bool(
            not self.use_legacy and
            transformer_config.moe_router_load_balancing_type == "gbs_aux_loss"
        )

    def construct(self, *inputs):
        """Construct function for pipeline with multiple outputs."""
        ret = None
        ret2 = None
        ret3 = None
        ret4 = None
        ret5 = None
        output = None
        for i in range(self.micro_size):
            micro_input = self.micro_inputs[i](i, *inputs)
            output = self.network(*micro_input)

            if not isinstance(output, tuple):
                if ret is not None:
                    ret = self.add_list[i](ret, output)
                else:
                    ret = output
                continue

            if ret is not None:
                ret = self.add_list[i](ret, output[0])
            else:
                ret = output[0]

            if len(output) >= 2:
                if ret2 is not None:
                    ret2 = self.add_list[i](ret2, output[1])
                else:
                    ret2 = output[1]

            if len(output) >= 3:
                if ret3 is not None:
                    ret3 = self.add_list[i](ret3, output[2])
                else:
                    ret3 = output[2]

            if len(output) >= 4:
                if ret4 is not None:
                    ret4 = self.add_list[i](ret4, output[3])
                else:
                    ret4 = output[3]

            if len(output) >= 5:
                if ret5 is not None:
                    ret5 = self.add_list[i](ret5, output[4])
                else:
                    ret5 = output[4]

        if self.is_reset_accu_gbs_fi:
            _reset_accu_gbs_fi(get_real_models(self.network))

        if not isinstance(output, tuple):
            return ret
        if len(output) == 2:
            return ret, ret2
        if len(output) == 3:
            return ret, ret2, ret3
        if len(output) == 4:
            return ret, ret2, ret3, ret4
        if len(output) == 5:
            return ret, ret2, ret3, ret4, ret5
        return ret


def _get_pipeline_group():
    """
    Calculate the communication group between all pipeline stages
    """
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    local_stage_rank_id = rank % per_stage_device_nums
    group = range(0, stage_nums)
    rank_list = [local_stage_rank_id + x * per_stage_device_nums for x in group]
    rank_str_list = [str(local_stage_rank_id + x * per_stage_device_nums) for x in group]
    rank_list_str = "-".join(rank_str_list)
    return rank_list, rank_list_str


class PipelineCellWithTwoOutput(nn.Cell):
    """
        Slice MiniBatch into finer-grained MicroBatch for use in pipeline-parallel training.

        Note:
            micro_size must be greater or equal to pipeline stages.

        Args:
            network (Cell): The target network to wrap.
            micro_size (int): MicroBatch size.

        Supported Platforms:
            ``Ascend`` ``GPU``
    """
    def __init__(self, network, micro_size):
        super().__init__(auto_prefix=False)
        self.network = network
        self.micro_inputs = nn.CellList()
        self.micro_size = micro_size
        self.add_list = []
        _check_network_with_micro_size('PipelineCellWithTwoOutput', network, micro_size)
        for i in range(micro_size):
            micro_input = _MicroBatch(micro_size)
            self.micro_inputs.append(micro_input)
            self.add = P.Add().add_prim_attr("pipeline_end", i)
            self.add_list.append(self.add)
        self._get_attr_from_cell(network)

    def construct(self, *inputs):
        """
        Construct function for pipeline with multiple input
        Args:
            *inputs:

        Returns:

        """
        ret = None
        ret2 = None
        for i in range(self.micro_size):
            micro_input = self.micro_inputs[i](i, *inputs)
            output1, output2 = self.network(*micro_input)

            if ret is not None:
                ret = self.add_list[i](ret, output1)
            else:
                ret = output1

            if ret2 is not None:
                ret2 = self.add_list[i](ret2, output2)
            else:
                ret2 = output2

        loss = ret, ret2
        return loss


class PipelineCellWithMultiOutputs(nn.Cell):
    """
        Slice MiniBatch into finer-grained MicroBatch for use in pipeline-parallel training.

        Note:
            micro_size must be greater or equal to pipeline stages.

        Args:
            network (Cell): The target network to wrap.
            micro_size (int): MicroBatch size.

        Supported Platforms:
            ``Ascend`` ``GPU``
    """

    def __init__(self, network, micro_size):
        super().__init__(auto_prefix=False)
        self.network = network
        self.micro_inputs = nn.CellList()
        self.micro_size = micro_size
        self.add_list = []
        _check_network_with_micro_size('PipelineCellWithMultiOutputs', network, micro_size)
        for i in range(micro_size):
            micro_input = _MicroBatch(micro_size)
            self.micro_inputs.append(micro_input)
            self.add = P.Add().add_prim_attr("pipeline_end", i)
            self.add_list.append(self.add)
        self._get_attr_from_cell(network)

        # reset for gbs_aux_loss
        self.use_legacy = is_legacy_model()
        real_network = get_real_models(self.network)
        transformer_config = real_network.get_gpt_transformer_config() if not self.use_legacy else None
        self.is_reset_accu_gbs_fi = bool(
            not self.use_legacy and
            transformer_config.moe_router_load_balancing_type == "gbs_aux_loss"
        )

    def construct(self, *inputs):
        """
        Construct function for pipeline with multiple input
        Args:
            *inputs:

        Returns:

        """
        ret = None
        ret2 = None
        ret3 = None
        ret4 = None
        ret5 = None
        output = None
        for i in range(self.micro_size):
            micro_input = self.micro_inputs[i](i, *inputs)
            output = self.network(*micro_input)

            if not isinstance(output, tuple):
                if ret is not None:
                    ret = self.add_list[i](ret, output)
                else:
                    ret = output
                continue

            if ret is not None:
                ret = self.add_list[i](ret, output[0])
            else:
                ret = output[0]

            if len(output) >= 2:
                if ret2 is not None:
                    ret2 = self.add_list[i](ret2, output[1])
                else:
                    ret2 = output[1]

            if len(output) >= 3:
                if ret3 is not None:
                    ret3 = self.add_list[i](ret3, output[2])
                else:
                    ret3 = output[2]

            if len(output) >= 4:
                if ret4 is not None:
                    ret4 = self.add_list[i](ret4, output[3])
                else:
                    ret4 = output[3]

            if len(output) >= 5:
                if ret5 is not None:
                    ret5 = self.add_list[i](ret5, output[4])
                else:
                    ret5 = output[4]

        if self.is_reset_accu_gbs_fi:
            _reset_accu_gbs_fi(get_real_models(self.network))

        if not isinstance(output, tuple):
            return ret
        if len(output) == 2:
            return ret, ret2
        if len(output) == 3:
            return ret, ret2, ret3
        if len(output) == 4:
            return ret, ret2, ret3, ret4
        if len(output) == 5:
            return ret, ret2, ret3, ret4, ret5
        return ret


# pylint: disable=W1401
@MindFormerRegister.register(MindFormerModuleType.WRAPPER)
class MFPipelineWithLossScaleCell(nn.TrainOneStepWithLossScaleCell):
    """
    Append a train-one-step cell with loss scale of pipeline parallel for MindFormers.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        use_clip_grad (bool, optional): Whether to use gradient clipping. Default: ``True``.
        max_grad_norm (float, optional): Maximum gradient constraint value. Default: ``1.0``.
        scale_sense (Union[Tensor, Cell], optional): Cell to do the loss scale. Default: ``1.0``.
        micro_batch_num (int, optional): Micro batch number of pipeline parallel. Default: ``1``.
        local_norm (bool, optional): Whether to calculate the local norm. Default: ``False``.
        calculate_per_token_loss (bool, optional): Whether to calculate the loss of each token. Default: ``False``.
        global_norm_spike_threshold (float, optional): The threshold of global norm when
            the skip data function is enabled. Default: ``1.0``.
        use_skip_data_by_global_norm (bool, optional): The switch of the skip data function. Default: ``False``.
        print_separate_loss (bool, optional): Whether to print loss separately. Default: ``False``.
        **kwargs (Any): Additional parameters.

    Inputs:
        - **\\*inputs** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Tuple of 5 or 7 Tensor, the loss, overflow flag, current loss scale value, learning rate,
        global grads norm, local grads norm and size of local norm grads.

        - **loss** (Tensor) -  A scalar, the loss value.
        - **overflow** (Tensor) -  A scalar, whether overflow occur or not, the type is bool.
        - **loss scale** (Tensor) -  The loss scale value, the shape is :math:`()` or :math:`(1,)`.
        - **learning rate** (Tensor) -  A scalar, the learning rate of the optimizer.
        - **global norm** (Tensor) -  A scalar, the global norm of all grads, only be calculated
          when `use_clip_grad=True`, otherwise None.
        - **local_norm** (Tensor) -  The local norm of the grads by group, only be returned when `local_norm=True`.
        - **size** (Tensor) -  The sizes of each grads group, only be returned when `local_norm=True`.

    Raises:
        TypeError: If `scale_sense` is neither Cell nor Tensor.
        ValueError: If shape of `scale_sense` is neither (1,) nor ().
        ValueError: If the parallel mode is not one of [ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL].
    """

    @args_type_check(global_norm_spike_threshold=float, use_skip_data_by_global_norm=bool)
    def __init__(self, network, optimizer, use_clip_grad=True, max_grad_norm=1.0,
                 scale_sense=1.0, micro_batch_num=1, local_norm=False,
                 calculate_per_token_loss=False, global_norm_spike_threshold=1.0,
                 use_skip_data_by_global_norm=False, print_separate_loss=False, **kwargs):
        if isinstance(scale_sense, (int, float)):
            scale_sense = Tensor(scale_sense)
        super().__init__(network, optimizer, scale_sense)
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.accu_grads = self.weights.clone(prefix="accu_grads", init="zeros")
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.grad_reducer = get_identity()
        self.degree = 1
        self.cast = P.Cast()
        self.status = Tensor([0] * 8, mstype.int32)
        self.reduce_sum = SumExt()
        if self.parallel_mode not in [ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL]:
            raise ValueError(f"ParallelMode must be one of "
                             f"[ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL], but found "
                             f"{self.parallel_mode}.")
        self.allreduce = P.AllReduce()
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.reshape = P.Reshape()
        self.loss_scaling_manager = None
        if isinstance(scale_sense, nn.Cell):
            self.loss_scaling_manager = scale_sense
            self.scale_sense = Parameter(Tensor(scale_sense.get_loss_scale(), dtype=mstype.float32),
                                         name="scale_sense")
        elif isinstance(scale_sense, Tensor):
            if scale_sense.shape in ((1, ), ()):
                self.scale_sense = Parameter(scale_sense, name='scale_sense')
            else:
                raise ValueError(f"The shape of 'scale_sense' must be (1,) or (), but got {scale_sense.shape}")
        else:
            raise TypeError(f"The 'scale_sense' must be Cell or Tensor, but got {type(scale_sense)}")
        self.opt_shard = _get_enable_parallel_optimizer()
        self.use_clip_grad = use_clip_grad
        self.clip_grad_norm = ClipGradNorm(max_norm=max_grad_norm)
        self.micro_size = micro_batch_num
        self.parallel_config = kwargs.pop("parallel_config", None)
        self.learning_rate = deepcopy(self.optimizer.learning_rate)
        self.localnorm = LocalNorm()
        self.concat = P.Concat()
        self.local_norm = local_norm
        # create allreduce for synchronize denominator
        pipeline_group_list, pipeline_group_name = _get_pipeline_group()
        hashed = hashlib.sha256(pipeline_group_name.encode()).hexdigest()[:48]
        pipeline_group_name = str(hashed)
        create_group(pipeline_group_name, pipeline_group_list)
        self.allreduce2 = P.AllReduce(group=pipeline_group_name)
        self.calculate_per_token_loss = calculate_per_token_loss
        self.grad_scale_factor = Tensor([1], dtype=mstype.float32)
        self.zero_t = Tensor([0], dtype=mstype.float32)
        self.dump_device_local_norm = get_auto_parallel_context("dump_device_local_norm")
        self.if_dump = bool(get_auto_parallel_context("dump_local_norm_path"))

        if self.if_dump:
            self.dump = P.TensorDump()
            self.dump_path = os.path.join(get_auto_parallel_context("dump_local_norm_path"), f"rank_{get_real_rank()}")
            if os.path.exists(self.dump_path):
                shutil.rmtree(self.dump_path)
            self.device_local_norm_filename = os.path.join(self.dump_path, "device_local_norm")
            self.finish_step_filename = os.path.join(self.dump_path, "finish_step")
        if self.dump_device_local_norm:
            self.squared_device_local_norm = get_squared_device_local_norm_param()
        self.global_norm_spike_threshold = global_norm_spike_threshold
        self.use_skip_data_by_global_norm = use_skip_data_by_global_norm

        self.use_legacy = is_legacy_model()
        real_network = get_real_models(self.network)
        transformer_config = real_network.get_gpt_transformer_config() if not self.use_legacy else None

        # loss
        self.print_separate_loss = bool(print_separate_loss and not self.use_legacy)
        if self.print_separate_loss:
            self.aux_loss_parameter = parameter_register.register("aux_loss", Tensor([0], mstype.float32))
            self.mtp_loss_parameter = parameter_register.register("mtp_loss", Tensor([0], mstype.float32))
            self.lm_loss_parameter = parameter_register.register("lm_loss", Tensor([0], mstype.float32))
            self.aux_loss_scale = get_aux_loss_scale(
                transformer_config.moe_aux_loss_coeff,
                transformer_config.moe_layer_freq,
                transformer_config.num_layers
            )

        # Get grouped LR schedulers from base_trainer
        grouped_lr_scheduler = kwargs.get('grouped_lr_scheduler')
        self.use_grouped_lr = grouped_lr_scheduler is not None
        if self.use_grouped_lr:
            # Register default_lr
            self.default_lr = kwargs.get('lr_scheduler')
            self.default_lr_parameter = parameter_register.register(
                "current_default_lr", Tensor(0., mstype.float32))
            # Register grouped_lr
            self.grouped_lr = nn.CellList(
                [deepcopy(current_lr.get('lr_scheduler')) for current_lr in grouped_lr_scheduler])
            self.grouped_lr_parameter = parameter_register.register(
                "current_grouped_lr", Tensor([0.] * len(self.grouped_lr), mstype.float32))

    @C.add_flags(has_effect=True)
    def construct(self, *inputs):
        """The construct processes of pipeline wrapper cell."""
        scaling_sens = self.scale_sense

        if self.dump_device_local_norm:
            _ = F.assign(self.squared_device_local_norm, Tensor(0.0, mstype.float32))
            scaling_sens = F.depend(self.scale_sense, _)

        if self.use_legacy:
            loss, grads, grad_scale_factor = self.grads_for_legacy(scaling_sens, *inputs)
        else:
            loss, grads, grad_scale_factor = self.grads_for_mcore(scaling_sens, *inputs)

        if self.local_norm:
            local_norm, size = self.localnorm(grads)
            local_norm = self.concat(local_norm)

        if self.opt_shard:
            grads = self.grad_reducer(grads)
            grads = self.hyper_map(F.partial(shard_grad_scale, scaling_sens * self.degree, grad_scale_factor),
                                   grads, self.accu_grads)
        else:
            accu_grads = self.grad_reducer(self.accu_grads)
            grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree, grad_scale_factor), grads,
                                   accu_grads)

        global_norm = None
        if self.use_clip_grad:
            grads, global_norm = self.clip_grad_norm(grads)

        learning_rate = self.learning_rate
        if self.optimizer.dynamic_lr:
            if self.use_grouped_lr:
                # Get default_lr, grouped_lr and update parameters registered
                learning_rate = self.default_lr(self.optimizer.global_step).reshape(())
                self.default_lr_parameter = learning_rate
                for group_id, current_lr in enumerate(self.grouped_lr):
                    self.grouped_lr_parameter[group_id] = current_lr(self.optimizer.global_step).reshape(())
            elif self.optimizer.is_group_lr:
                learning_rate = self.learning_rate[-1](self.optimizer.global_step).reshape(())
            else:
                learning_rate = self.learning_rate(self.optimizer.global_step).reshape(())

        # sum overflow flag over devices
        cond = self.get_overflow_status(self.status, grads)
        cond = F.depend(cond, grads)
        overflow = self.process_loss_scale(cond)

        if self.use_graceful_exit:
            grads = self.graceful_exit.exit_by_request(grads, self.init_param, self.exit_param)

        if not overflow:
            if self.use_skip_data_by_global_norm:
                if global_norm < self.global_norm_spike_threshold:
                    loss = F.depend(loss, self.optimizer(grads))
            else:
                loss = F.depend(loss, self.optimizer(grads))

        if self.if_dump:
            zero = Tensor(0.0, mstype.float32)
            if self.dump_device_local_norm:
                zero = F.depend(zero, self.dump(self.device_local_norm_filename,
                                                F.sqrt(self.squared_device_local_norm)))
            filename = self.finish_step_filename
            self.dump(filename, zero)

        if self.local_norm:
            return loss, overflow, scaling_sens.value(), learning_rate, global_norm, local_norm, size
        return loss, overflow, scaling_sens.value(), learning_rate, global_norm

    def grads_for_legacy(self, scaling_sens, *inputs):
        """calculate grad for legacy model"""
        if self.calculate_per_token_loss:
            numerator, denominator = self.network(*inputs)
            denominator = self.allreduce2(denominator)
            loss = numerator / denominator
            scaling_sens_filled = C.ones_like(numerator) * F.cast(scaling_sens, F.dtype(numerator))
            scaling_sens_filled2 = self.zero_t * F.cast(scaling_sens, F.dtype(denominator))

            grads = self.grad(self.network, self.weights)(*inputs,
                                                          (self.cast(scaling_sens_filled, mstype.float32),
                                                           self.cast(scaling_sens_filled2, mstype.float32)))
            grad_scale_factor = denominator
        else:
            loss = self.network(*inputs)
            scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
            grads = self.grad(self.network, self.weights)(*inputs,
                                                          self.cast(scaling_sens_filled / self.micro_size,
                                                                    mstype.float32))
            grad_scale_factor = self.grad_scale_factor
        return loss, grads, grad_scale_factor

    def grads_for_mcore(self, scaling_sens, *inputs):
        """calculate grad for mcore model"""
        if self.calculate_per_token_loss:
            numerator0, denominator0, numerator1, _, aux_loss = self.network(*inputs)
            denominator0 = self.allreduce2(denominator0)
            lm_loss = numerator0 / denominator0
            mtp_loss = numerator1 / denominator0
            aux_loss = aux_loss / denominator0
            loss = lm_loss + aux_loss
            loss = loss + mtp_loss

            scaling_sens_filled = C.ones_like(numerator0) * F.cast(scaling_sens, F.dtype(numerator0))
            scaling_sens_filled_zero = self.zero_t * F.cast(scaling_sens, F.dtype(denominator0))
            grads = self.grad(self.network, self.weights)(*inputs,
                                                          (self.cast(scaling_sens_filled, mstype.float32),
                                                           self.cast(scaling_sens_filled_zero, mstype.float32),
                                                           self.cast(scaling_sens_filled, mstype.float32),
                                                           self.cast(scaling_sens_filled_zero, mstype.float32),
                                                           self.cast(scaling_sens_filled / self.micro_size,
                                                                     mstype.float32),
                                                           ))
            grad_scale_factor = denominator0
        else:
            lm_loss, mtp_loss, aux_loss = self.network(*inputs)
            loss = lm_loss + aux_loss
            loss = loss + mtp_loss
            scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
            scaling_sens_filled = self.cast(scaling_sens_filled / self.micro_size, mstype.float32)
            grads = self.grad(self.network, self.weights)(*inputs, (scaling_sens_filled,
                                                                    scaling_sens_filled,
                                                                    scaling_sens_filled))
            grad_scale_factor = self.grad_scale_factor

        if self.print_separate_loss:
            F.assign_add(self.aux_loss_parameter, aux_loss / self.aux_loss_scale)
            F.assign_add(self.mtp_loss_parameter, mtp_loss)
            F.assign_add(self.lm_loss_parameter, lm_loss)
        return loss, grads, grad_scale_factor
