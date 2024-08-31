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

import time
import contextlib
from copy import deepcopy

import mindspore.common.dtype as mstype
from mindspore import nn, Tensor, Parameter, mint, value_and_grad
from mindspore.amp import DynamicLossScaler, StaticLossScaler, all_finite
from mindspore.communication.comm_func import all_reduce
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_enable_parallel_optimizer
from mindformers.core.clip_grad import ClipGradNorm

from mindformers.tools.logger import logger
from mindformers.experimental.distri_cores.grad_handler import (
    inplace_apply_to_tensor_list,
    get_grad_process_func,
    GradAccumulator,
)
from mindformers.experimental.distri_cores.create_comm import (
    get_dp_world_size,
    get_dp_group,
    get_tp_world_size,
    get_tp_group,
    get_pp_world_size,
    get_pp_group,
    is_pipeline_last_stage,
    set_vpp_rank,
    is_pipeline_first_stage,
    get_data_modulo_expert_parallel_group,
    get_ep_world_size
)
from mindformers.experimental.distri_cores.pipeline_parallel import PipelineCell
from mindformers.experimental.distri_cores.pipeline_parallel.schedules import (
    forward_backward_pipelining_without_interleaving,
    forward_backward_pipelining_with_interleaving,
    rename_hidden_states_parameter
)
from mindformers.experimental.distri_cores.config import GeneralConfig
from mindformers.experimental.distri_cores.checkpointing import save_checkpoint


def get_sp_params(training_config):
    """get reduce parameters for sequence parallel"""
    use_lora = training_config.lora_config.use_lora
    if use_lora:
        sp_params = [
            'qkv_proj.lora_a',
            'out_proj.lora_b',
            'mapping.lora_a',
            'projection.lora_b',
            'q_proj.lora_a',
            'kv_proj.lora_a',
            'gating.lora_a'
        ]
    else:
        sp_params = ["norm", "mlp.projection.bias", "attention.out_proj.bias"]
    return sp_params


class ParallelTrainingReducer:
    """The reducer for parallel training"""

    def __init__(self, params, training_config):
        super(ParallelTrainingReducer, self).__init__()

        self.enable_grad_reduce = {
            "dp": False,
            "pp": False,
            "tp": False,  # only valid in the case of sequence parallel
            "ep-dp": False
        }

        self.enable_loss_reduce = {
            "dp": False,
            "pp": False,
            "tp": False,
        }

        self.enable_grad_flag_reduce = {
            "dp": False,
            "pp": False,
            "tp": False,
        }

        self.sp_reduce_params = get_sp_params(training_config)
        self.expert_params = ["mlp.experts.local_experts"]

        self.batch_reduction = training_config.loss_reduction

        # dp
        if get_dp_world_size() > 1:
            self.enable_loss_reduce["dp"] = True
            if training_config.parallel_config.zero_level is None:
                self.enable_grad_reduce["dp"] = True
            else:
                self.enable_grad_flag_reduce["dp"] = True

        # tp / sp
        if get_tp_world_size() > 1:
            self.enable_grad_flag_reduce["tp"] = True
            if training_config.parallel_config.use_sequence_parallel:
                self.enable_grad_reduce["tp"] = True
                self.sp_reduce_filter = [
                    any([sp_param in param.name for sp_param in self.sp_reduce_params]) for param in params
                ]

        # pp
        if get_pp_world_size() > 1:
            self.enable_grad_flag_reduce["pp"] = True

        # ep
        if get_ep_world_size() > 1:
            self.enable_grad_reduce["ep-dp"] = True
            self.expert_filter = [
                any([ep_param in param.name for ep_param in self.expert_params]) for param in params
            ]

    def get_reduce_group(self, idx):
        if self.enable_grad_reduce["ep-dp"] and self.expert_filter[idx]:
            group = get_data_modulo_expert_parallel_group()
        else:
            group = get_dp_group()
        return group

    def inplace_reduce_dp_grad(self, grads):
        """Reduce the gradients in data parallel mode."""
        if not self.enable_grad_reduce["dp"]:
            return
        if self.batch_reduction == "mean":
            for idx, grad in enumerate(grads):
                group = self.get_reduce_group(idx)
                grads[idx] = mint.div(all_reduce(grad, "sum", group), get_dp_world_size())
        elif self.batch_reduction == "sum":
            for idx, grad in enumerate(grads):
                group = self.get_reduce_group(idx)
                grads[idx] = all_reduce(grad, "sum", group)

    def inplace_reduce_sp_grad(self, grads):
        """Reduce the gradients in sequence parallel mode over tp group."""
        if not self.enable_grad_reduce["tp"]:
            return
        for idx, reduce_flag in enumerate(self.sp_reduce_filter):
            if reduce_flag:
                grads[idx] = all_reduce(grads[idx], "sum", get_tp_group())

    def inplace_reduce_grad(self, grads):
        """Reduce the gradients in all parallel modes."""
        self.inplace_reduce_dp_grad(grads)
        self.inplace_reduce_sp_grad(grads)

    def reduce_dp_loss(self, loss):
        """Reduce the loss in data parallel mode."""
        if not self.enable_loss_reduce["dp"]:
            return loss
        if self.batch_reduction == "mean":
            return mint.div(all_reduce(loss, "sum", get_dp_group()), get_dp_world_size())
        return all_reduce(loss, "sum", get_dp_group())

    def reduce_overflow(self, overflow):
        """Reduce the overflow status in all parallel modes."""
        # logical or
        overflow = Tensor(overflow, dtype=mstype.int8)
        if self.enable_grad_flag_reduce["pp"]:
            overflow = all_reduce(overflow, "max", get_pp_group())
        if self.enable_grad_flag_reduce["dp"]:
            overflow = all_reduce(overflow, "max", get_dp_group())
        if self.enable_grad_flag_reduce["tp"]:
            overflow = all_reduce(overflow, "max", get_tp_group())
        return overflow.astype(mstype.bool_)

    def reduce_is_finite(self, is_finite):
        """Reduce the is_finite status in all parallel modes."""
        # logical and
        is_finite = Tensor(is_finite, dtype=mstype.int8)
        if self.enable_grad_flag_reduce["pp"]:
            is_finite = all_reduce(is_finite, "prod", get_pp_group())
        if self.enable_grad_flag_reduce["dp"]:
            is_finite = all_reduce(is_finite, "prod", get_dp_group())
        if self.enable_grad_flag_reduce["tp"]:
            is_finite = all_reduce(is_finite, "prod", get_tp_group())
        return is_finite.astype(mstype.bool_)


def get_model(model_provider_func, parallel_config):
    """ get model """
    model = nn.CellList(auto_prefix=False)
    if get_pp_world_size() > 1:
        if parallel_config.virtual_pipeline_model_parallel_size is not None and \
           parallel_config.virtual_pipeline_model_parallel_size > 1:
            for i in range(parallel_config.virtual_pipeline_model_parallel_size):
                set_vpp_rank(i)
                pre_process = is_pipeline_first_stage()
                post_process = is_pipeline_last_stage()
                this_model = model_provider_func(pre_process=pre_process,
                                                 post_process=post_process)
                rename_hidden_states_parameter(this_model, i)
                model.append(PipelineCell(this_model, model_customize_staged=True))
        else:
            pre_process = is_pipeline_first_stage()
            post_process = is_pipeline_last_stage()
            this_model = model_provider_func(pre_process=pre_process,
                                             post_process=post_process)
            # wrap with PP cell if pipeline parallelism is used
            this_model = PipelineCell(this_model, model_customize_staged=True)
            model.append(this_model)
    else:
        model.append(model_provider_func(pre_process=True, post_process=True))

    if len(model) == 1:
        model = model[0]
    return model


def get_forward_backward_func(network_with_loss, params, training_config, model_config):
    """
    Returns a forward-backward function for training a network with or without pipeline parallelism.

    Args:
        network_with_loss (callable): A function that takes inputs and returns the network output and loss.
        params (list): List of parameters to compute gradients for.
        training_config (TrainingConfig): Training configuration.
        model_config (TransformerConfig): Model configuration.

    Returns:
        callable: A forward-backward function that can be used for training the network.

    Raises:
        NotImplementedError: If pipeline parallelism is not implemented yet.
    """
    forward_backward_func = None
    seq_length = model_config.seq_length
    micro_batch_num = training_config.dataset_config.micro_batch_num
    micro_batch_size = training_config.dataset_config.batch_size

    # no pipeline parallel
    if get_pp_world_size() == 1:

        def forward_with_loss_scale(*inputs_tuple, loss_scale=None, **inputs_dict):
            logits = None
            output = network_with_loss(*inputs_tuple, **inputs_dict)
            if isinstance(output, tuple):
                loss, logits = output[0], output[1]
            else:
                loss = output
            if loss_scale is not None:
                loss = mint.mul(loss, loss_scale.astype(loss.dtype))
            return loss, logits

        grad_position = None
        # If parallel_config.zero_level == z3, gradient with respect to inputs and weights
        if model_config.parallel_config.zero_level == "z3":
            grad_position = 0
        forward_backward_once_func = value_and_grad(
            forward_with_loss_scale, grad_position=grad_position, weights=params, has_aux=True
        )

        if micro_batch_num > 1:
            grad_accumulator = GradAccumulator(micro_batch_num, op="sum")

        def forward_backward_func_with_grad_acc(
                *inputs_tuple, loss_scale=None, forward_only=False, **inputs_dict
        ):
            loss = None
            logits = None
            grads = None

            # fuse loss scale and grad accumulation if do grad acc
            if training_config.loss_reduction == "mean" and micro_batch_num > 1:
                if loss_scale is None:
                    loss_scale = Tensor(1, mstype.float32)
                actual_loss_scale = mint.div(loss_scale, micro_batch_num)
            else:
                actual_loss_scale = loss_scale

            no_sync_func = contextlib.nullcontext

            def forward_backward_on_microbatch(idx):
                nonlocal loss
                nonlocal logits
                nonlocal grads

                # slice inputs over batch size dimension
                inputs_tuple_micro = [
                    input_data[idx * micro_batch_size : (idx + 1) * micro_batch_size] for input_data in inputs_tuple
                ]
                inputs_dict_micro = {
                    key: value[idx * micro_batch_size : (idx + 1) * micro_batch_size]
                    for key, value in inputs_dict.items()
                }

                # step on micro batch
                if forward_only:
                    loss_micro, logits_micro = forward_with_loss_scale(
                        *inputs_tuple_micro, loss_scale=actual_loss_scale, **inputs_dict_micro
                    )
                else:
                    (loss_micro, logits_micro), grads_micro = forward_backward_once_func(
                        *inputs_tuple_micro, loss_scale=actual_loss_scale, **inputs_dict_micro
                    )
                    if grad_position == 0:
                        grads_micro = grads_micro[1]
                    # accumulate grads
                    if micro_batch_num > 1:
                        grads = grad_accumulator(grads_micro)
                    else:
                        grads = grads_micro

                # process output, loss will be averaged in loss unscaling
                loss = loss_micro if loss is None else loss + loss_micro

                if logits is None:
                    logits = logits_micro
                else:
                    logits = mint.cat((logits, logits_micro), dim=0)

            # trigger dp reduce only on last step
            with no_sync_func():
                for idx in range(micro_batch_num - 1):
                    forward_backward_on_microbatch(idx)
            forward_backward_on_microbatch(micro_batch_num - 1)

            # unscale loss
            if loss_scale is not None:
                loss = mint.div(loss, loss_scale)

            if forward_only:
                return loss, logits

            return (loss, logits), grads

        forward_backward_func = forward_backward_func_with_grad_acc

    else:
        all_config = GeneralConfig(
            model_config=model_config,
            training_config=training_config,
        )

        def forward_backward_with_pipelining(
                *inputs_tuple, loss_scale=None, forward_only=False, **inputs_dict
        ):
            if loss_scale is None:
                loss_scale = Tensor(1, mstype.float32)
            if model_config.parallel_config.virtual_pipeline_model_parallel_size is not None and \
               model_config.parallel_config.virtual_pipeline_model_parallel_size > 1:
                loss, logits, grads = forward_backward_pipelining_with_interleaving(
                    network_with_loss,
                    micro_batch_num,
                    seq_length,
                    micro_batch_size,
                    *inputs_tuple,
                    decoder_seq_length=None,
                    forward_only=forward_only,
                    collect_non_loss_data=False,
                    first_val_step=None,
                    config=all_config,
                    total_tokens_nums=None,
                    scale_sense=loss_scale,
                    **inputs_dict
                )
            else:
                loss, logits, grads = forward_backward_pipelining_without_interleaving(
                    network_with_loss,
                    micro_batch_num,
                    seq_length,
                    micro_batch_size,
                    *inputs_tuple,
                    decoder_seq_length=None,
                    forward_only=forward_only,
                    collect_non_loss_data=False,
                    first_val_step=None,
                    config=all_config,
                    total_tokens_nums=None,
                    scale_sense=loss_scale,
                    **inputs_dict
                )
            if forward_only:
                return loss, logits
            return (loss, logits), grads

        forward_backward_func = forward_backward_with_pipelining

    return forward_backward_func


class TrainOneStepCell(nn.Cell):
    r"""TrainOneStepCell with loss scaling, grad clipping, and grad accumulation.

    Args:
        network_with_loss (nn.Cell): The network with loss, output of the network should be loss,
            which is a scalar Tensor.
        optimizer (Optimizer): The optimizer used for training.
        training_config (TrainingConfig): Training Configuration.
        model_config (TransformerConfig): Transformer Configuration.
        **kwargs: Additional keyword arguments.

    Raises:
        NotImplementedError: If gradient accumulation is not supported in pipeline parallel.

    Inputs:
        - **inputs_tuple** (Tuple[Tensor]) - Tuple of input tensors.
        - **inputs_dict** (Dict[str, Tensor]) - Dict of input tensors.

    Outputs:
        Tuple of 4 Tensor, the loss, overflow flag, current loss scale value, and learning rate.

        - **loss** (Tensor) -  A scalar, the loss value.
        - **is_finite** (Tensor) -  A bool, whether grads is finite.
        - **loss scale** (Union[Tensor, None]) -  The loss scale value, None if not using loss scaling.
        - **learning rate** (Union[Tensor, List[Tensor])) -  The model learning rate.
    """

    # pylint: disable=W0613
    def __init__(self, network_with_loss, optimizer, training_config, model_config, **kwargs):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network_with_loss = network_with_loss
        self.optimizer = optimizer
        self.model_config = model_config

        # init loss scaler
        if training_config.loss_scale is not None:
            self.loss_scaler = StaticLossScaler(scale_value=training_config.loss_scale)
        else:
            # dynamic loss scaler is used only if the model is computed in float16
            if model_config.compute_dtype == mstype.float16:
                self.loss_scaler = DynamicLossScaler(
                    scale_value=training_config.loss_scale_value,
                    scale_factor=training_config.loss_scale_factor,
                    scale_window=training_config.loss_scale_window)
            else:
                logger.warning(
                    "Dynamic loss scale is only supported for float16 computation. Not using loss scaling.")
                self.loss_scaler = None

        # init grad clip func
        self.use_grad_clip = training_config.grad_clip_kwargs is not None
        if self.use_grad_clip:
            self.grad_clip_func = get_grad_process_func(
                training_config, params=network_with_loss.trainable_params()
            )
        # init grad scale func
        self.grad_scale_func = inplace_apply_to_tensor_list(mint.mul)

        # init parallel reducer
        self.parallel_reducer = ParallelTrainingReducer(network_with_loss.trainable_params(), training_config)

        self.micro_batch_num = training_config.dataset_config.micro_batch_num
        # init forward_backward_func
        self.forward_backward_func = get_forward_backward_func(
            network_with_loss, network_with_loss.trainable_params(), training_config, model_config
        )

    def unscale_and_clip_grads(self, grads, loss_scale=None):
        """Handle grads with scaling and clipping.

        Args:
            grads (tuple): The gradients.
            loss_scale (Tensor, optional): The scaling factor of loss. Defaults: None.
        """
        if loss_scale is not None:
            inv_scale = mint.reciprocal(loss_scale).astype(grads[0].dtype)
            self.grad_scale_func(grads, inv_scale)
        if self.use_grad_clip:
            self.grad_clip_func(grads)

    def construct(self, *inputs_tuple, **inputs_dict):
        """Forward, backward, grad process, and optimizer step."""
        # forward and backward
        if self.loss_scaler is not None:
            current_step_loss_scale = self.loss_scaler.scale_value
        else:
            current_step_loss_scale = None
        # loss is scale and unscale in forward_backward_func
        (loss, _), grads = self.forward_backward_func(*inputs_tuple, loss_scale=current_step_loss_scale, **inputs_dict)

        # apply grad reducer
        grads = list(grads)
        self.parallel_reducer.inplace_reduce_grad(grads)
        grads_ = tuple(grads)

        # check overflow
        is_finite = all_finite(grads_)
        #    sync over tp and pp group
        is_finite = self.parallel_reducer.reduce_is_finite(is_finite)

        if self.loss_scaler is not None:
            self.loss_scaler.adjust(is_finite)

        if is_finite:
            # scale grads and clip grads if enabled
            grads = list(grads)
            self.unscale_and_clip_grads(grads, current_step_loss_scale)
            grads_ = tuple(grads)

            # optimizer step
            self.optimizer(grads_)

        learning_rate = self.optimizer.get_lr()
        if isinstance(learning_rate, Parameter):
            learning_rate = learning_rate.value()
        if isinstance(learning_rate, tuple):
            learning_rate = tuple([
                individual_learning_rate.value()
                if isinstance(individual_learning_rate, Parameter)
                else individual_learning_rate
                for individual_learning_rate in learning_rate
            ])

        # reduce loss if dp
        loss = self.parallel_reducer.reduce_dp_loss(loss)

        return loss, is_finite, current_step_loss_scale, learning_rate


def train(
        train_one_step_cell,
        train_dataset_iterator,
        training_config,
        val_dataset_iterator=None,
        metrics=None,
        evaluation_func=None,
        **kwargs,
):
    """
    Train the model using the provided training configuration.

    Args:
        train_one_step_cell (TrainOneStepCell): The training cell object.
        train_dataset_iterator (Dataset): The iterator for the training dataset.
        training_config (TrainingConfig): The configuration object for training.
        val_dataset_iterator (iterable, optional): The iterator for the validation dataset. Defaults: None
        metrics (dict[str, Metric], optional): A dictionary of metrics to track during training.
            Defaults: None
        evaluation_func (callable, optional): The evaluation function to use for validation. Defaults: None
        **kwargs: Additional keyword arguments.

    Returns:
        None
    """
    train_one_step_cell.set_train()
    model_config = train_one_step_cell.model_config
    # set input to use dynamic shape
    model_input_data = next(train_dataset_iterator.create_tuple_iterator())
    set_input_data = [
        Tensor(shape=(None,) * len(input_data.shape), dtype=input_data.dtype) for input_data in model_input_data
    ]
    train_one_step_cell.set_inputs(*set_input_data)

    global_step = 0
    current_epoch = 0
    evaluation_flag = (
        val_dataset_iterator is not None
        and evaluation_func is not None
        and metrics is not None
        and training_config.eval_interval is not None
        and training_config.best_metric_comparison is not None
        and training_config.eval_metric is not None)
    save_ckpt_flag = training_config.save_interval is not None
    correct_metric_flag = is_pipeline_last_stage() # not use pp or pp last_stage

    if evaluation_flag:
        if training_config.best_metric_comparison == "less_equal":
            best_metric_compare_func = mint.less_equal
            best_metric = Tensor(float("inf"))
        elif training_config.best_metric_comparison == "greater_equal":
            best_metric_compare_func = mint.greater_equal
            best_metric = Tensor(float("-inf"))
        elif training_config.best_metric_comparison == "less":
            best_metric_compare_func = mint.less
            best_metric = Tensor(float("inf"))
        elif training_config.best_metric_comparison == "greater":
            best_metric_compare_func = mint.greater
            best_metric = Tensor(float("-inf"))

    # check if the training should be stopped
    while not (
            training_config.epochs is not None
            and current_epoch >= training_config.epochs
            or global_step >= training_config.training_iters):
        for data in train_dataset_iterator.create_dict_iterator():
            # check if the training should be stopped
            if global_step >= training_config.training_iters:
                break
            start_time = time.time()
            loss, is_finite, loss_scale, learning_rate = train_one_step_cell(**data)
            end_time = time.time()
            if training_config.log_interval is not None and (global_step + 1) % training_config.log_interval == 0:
                if not correct_metric_flag:
                    logger.warning("Metrics is only calculated on the last stage.")
                logger.warning(
                    f"Epoch: {current_epoch}, Step: {global_step}, Loss: {loss}, "
                    + f"Finite_grads: {is_finite}, "
                    + f"Loss_scale: {loss_scale.value() if loss_scale is not None else None}, "
                    + f"Learning_rate: {learning_rate}, Time: {(end_time - start_time) * 1000:.2f} ms")

            if evaluation_flag and (global_step + 1) % training_config.eval_interval == 0:
                is_best = Tensor(False, dtype=mstype.int8)
                results = evaluation_func(train_one_step_cell, val_dataset_iterator, metrics, **kwargs)

                # update best_metrics only on last stage
                if correct_metric_flag and best_metric_compare_func(results[training_config.eval_metric], best_metric):
                    best_metric = results[training_config.eval_metric]
                    is_best = Tensor(True, dtype=mstype.int8)

                if get_pp_world_size() > 1:
                    is_best = all_reduce(is_best, "max", get_pp_group())

                # save ckpt
                if is_best and save_ckpt_flag:
                    logger.warning("saving best checkpoint")
                    if save_ckpt_flag:
                        ckpt_path = training_config.output_dir + "/best"
                        save_checkpoint(model_config,
                                        train_one_step_cell.network_with_loss,
                                        train_one_step_cell.optimizer,
                                        ckpt_path)

            if save_ckpt_flag and (global_step + 1) % training_config.save_interval == 0:
                ckpt_path = training_config.output_dir + f"/step_{global_step}"
                save_checkpoint(model_config,
                                train_one_step_cell.network_with_loss,
                                train_one_step_cell.optimizer,
                                ckpt_path)

            global_step += 1

        current_epoch += 1

    if save_ckpt_flag:
        ckpt_path = training_config.output_dir + f"/step_{global_step}"
        save_checkpoint(model_config, train_one_step_cell.network_with_loss, train_one_step_cell.optimizer, ckpt_path)


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
