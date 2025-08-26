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
"""MindFormer Self-Define Callback."""
import json
import os
import re
import glob
import sys
import time
import tempfile
import hashlib
import subprocess
import shlex

from collections import OrderedDict, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Callable, Optional, Union, Dict, Tuple, List
import numpy as np

import mindspore as ms
import mindspore.ops.operations as P
import mindspore.ops.functional as F
from mindspore._checkparam import args_type_check
from mindspore import (
    Callback,
    ModelCheckpoint,
    CheckpointConfig,
    context,
    Tensor,
    get_auto_parallel_context,
    set_auto_parallel_context
)
from mindspore.train.callback import SummaryCollector
from mindspore.train.callback._checkpoint import CheckpointManager
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.train.serialization import _get_merged_param_data
from mindspore.nn.cell import Cell
from mindspore.ops.operations.comm_ops import Broadcast
from mindspore.common import jit
from mindspore.train._utils import get_parameter_redundancy, remove_param_redundancy
from mindspore.common.api import flops_collection
from mindspore.communication.management import create_group, get_group_size, get_rank, GlobalComm
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.comm_func import all_gather_into_tensor, barrier
from mindspore.profiler import ProfilerLevel, schedule

from mindformers.core.context.build_context import is_legacy_model
from mindformers.tools import get_output_root_path
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.utils import (
    get_output_subpath,
    get_real_rank,
    get_real_group_size,
    get_real_local_rank,
    get_pipeline_rank_ids,
    is_last_pipeline_stage,
    barrier_world,
    get_ascend_log_path,
    set_safe_mode_for_file_or_dir
)
from mindformers.utils.parameter_register import parameter_register
from mindformers.utils.tensorboard import get_tensorboard_writer, get_tensorboard_args
from mindformers.version_control import check_stress_detect_valid, is_version_ge, check_arf_status
from mindformers.parallel_core.training_graph.loss_func import (
    get_device_local_loss,
    reset_device_local_loss,
    check_device_local_loss
)
from mindformers.checkpoint.checkpoint import AsyncSaveManager, CommonInfo, save_checkpoint

__all__ = ['MFLossMonitor', 'CheckpointMonitor', 'SummaryMonitor', 'ProfileMonitor', 'EvalCallBack']

_cur_dir = os.getcwd()
SAVE_DIR = _cur_dir

VOLTAGE_ERROR_CODE = 574007


class AllReduceNet(Cell):
    """
    Used to accumulate flops in pipeline parallel.
    """

    def __init__(self, group_name):
        super(AllReduceNet, self).__init__()
        self.allreduce_sum = P.AllReduce(op=P.ReduceOp.SUM, group=group_name)
        self.add_flags(skip_auto_parallel_compile=True)

    def construct(self, x):
        return self.allreduce_sum(x)


def _check_mspti_is_on():
    """Check whether mspti is enabled."""
    ld_preload = os.getenv("LD_PRELOAD")
    return isinstance(ld_preload, str) and ld_preload.find("libmspti.so") != -1


def _get_separate_loss():
    """callback drop rate."""
    aux_loss = parameter_register.get("aux_loss").asnumpy()
    mtp_loss = parameter_register.get("mtp_loss").asnumpy()
    lm_loss = parameter_register.get("lm_loss").asnumpy()
    parameter_register.clear("aux_loss")
    parameter_register.clear("mtp_loss")
    parameter_register.clear("lm_loss")
    return lm_loss, aux_loss, mtp_loss


def _get_loss_output(output):
    """Get output of task for MFLossMonitor."""
    overflow = False
    scaling_sens = False
    loss = output
    learning_rate = None
    global_norm = None
    local_norm = None

    if isinstance(output, (tuple, list)):
        if len(output) in [3, 4, 5, 7]:
            loss, overflow, scaling_sens, *res = output
            if len(res) == 4:
                learning_rate, global_norm, local_norm, norm_size = res[0], res[1], res[2], res[3]
                logger.info(f" norm_size: {norm_size}\nlocal_norm:\n{local_norm}")
            if len(res) == 2:
                learning_rate, global_norm = res[0], res[1]
            if len(res) == 1:
                learning_rate = res[0]
            if isinstance(scaling_sens, ms.Tensor):
                scaling_sens = scaling_sens.asnumpy()
        else:
            if isinstance(output[0], ms.Tensor) and isinstance(output[0].asnumpy(), np.ndarray):
                loss = output[0]

    if isinstance(global_norm, ms.Tensor):
        global_norm = global_norm.asnumpy()

    if isinstance(loss, ms.Tensor) and isinstance(loss.asnumpy(), np.ndarray):
        loss = np.mean(loss.asnumpy())

    if isinstance(overflow, ms.Tensor):
        overflow = overflow.asnumpy()

    if isinstance(learning_rate, ms.Tensor):
        learning_rate = learning_rate.asnumpy()

    return loss, overflow, scaling_sens, learning_rate, global_norm


def _get_weight_norm(network):
    """Get the L2 norm of network trainable parameters. Return 0 if there's no trainable parameter"""
    norms = []
    for param in network.trainable_params():
        norms.append(param.to(ms.float32).norm())
    if not norms:
        return 0.0
    norm = float(F.stack(norms).norm().item())
    return norm


def _get_optimizer_state(optim_params, filter_fn: Callable = None):
    """Get the respective L2 norms of specified optimizer parameters. Return a dict"""
    norms = {}
    for param in optim_params:
        if filter_fn is None or filter_fn(param.name):
            norms[param.name] = float(param.to(ms.float32).norm().item())
    return norms


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class MFLossMonitor(Callback):
    """
    Monitor loss and other parameters in training process.

    Args:
        learning_rate (Union[float, LearningRateSchedule], optional): The learning rate schedule. Default: ``None``.
        per_print_times (int, optional): Every how many steps to print the log information. Default: ``1``.
        micro_batch_num (int, optional): MicroBatch size for Pipeline Parallel. Default: ``1``.
        micro_batch_interleave_num (int, optional): split num of batch size. Default: ``1``.
        origin_epochs (int, optional): Training epoches. Default: ``None``.
        dataset_size (int, optional): Training dataset size. Default: ``None``.
        initial_epoch (int, optional): The beginning epoch. Default: ``0``.
        initial_step (int, optional): The beginning step. Default: ``0``.
        global_batch_size (int, optional): The total batch size. Default: ``0``.
        gradient_accumulation_steps (int, optional): The gradient accumulation steps. Default: ``1``.
        check_for_nan_in_loss_and_grad (bool, optional): Whether to check loss and norm of grad is Nan.
            Default: ``False``.
        calculate_per_token_loss (bool, optional): Whether to calculate the loss of each token. Default: ``False``.
        print_separate_loss (bool, optional): Whether to print loss separately. Default: ``False``.

    Examples:
        >>> from mindformers.core import MFLossMonitor
        >>> lr = [0.01, 0.008, 0.006, 0.005, 0.002]
        >>> monitor = MFLossMonitor(learning_rate=lr, per_print_times=10)
    """

    def __init__(self,
                 learning_rate: Optional[Union[float, LearningRateSchedule]] = None,
                 per_print_times: int = 1,
                 micro_batch_num: int = 1,
                 micro_batch_interleave_num: int = 1,
                 origin_epochs: int = None,
                 dataset_size: int = None,
                 initial_epoch: int = 0,
                 initial_step: int = 0,
                 global_batch_size: int = 0,
                 gradient_accumulation_steps: int = 1,
                 check_for_nan_in_loss_and_grad: bool = False,
                 calculate_per_token_loss: bool = False,
                 print_separate_loss: bool = False,
                 **kwargs):
        super(MFLossMonitor, self).__init__()
        self.per_print_times = per_print_times
        self.learning_rate = deepcopy(learning_rate)
        self.last_print_time = 0
        self.mirco_size = micro_batch_num
        self.print_warning_flag = True
        self.loss_list = []
        self.step_time = time.time()
        self.epoch_time = time.time()
        self.run_context = None
        self.steps_per_epoch = dataset_size
        self.micro_batch_interleave_num = micro_batch_interleave_num
        self.origin_epochs = origin_epochs
        self.initial_epoch = initial_epoch
        self.initial_step = initial_step
        self.global_batch_size = global_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device_num = get_real_group_size()
        self.mf_support = None
        self.mf_calculated = False
        self.current_phase = None
        self.full_model_flops = 0.0
        self.tensor_writer = get_tensorboard_writer()
        self.tensorboard = get_tensorboard_args()
        self.check_for_nan_in_loss_and_grad = check_for_nan_in_loss_and_grad
        self.calculate_per_token_loss = calculate_per_token_loss
        self.mstx_range_id = None
        self.mstx_enabled = is_version_ge(ms.__version__, '2.5.0') and _check_mspti_is_on()
        self.print_separate_loss = print_separate_loss
        self.is_moe_model = kwargs.get("is_moe_model", False)
        self.is_mtp_model = kwargs.get("is_mtp_model", False)
        if self.print_separate_loss and is_legacy_model():
            logger.warning("print_separate_loss = True, is not supported when use_legacy = True.")
            self.print_separate_loss = False
        if self.print_separate_loss and not self.is_moe_model and not self.is_mtp_model:
            self.print_separate_loss = False

    def on_train_epoch_begin(self, run_context):
        """
        Record time at the beginning of epoch.

        Args:
            run_context (RunContext): Context of the process running.
        """
        self.loss_list = []
        self.epoch_time = time.time()
        self.run_context = run_context

    def on_train_step_begin(self, run_context):
        """
        Record time at the beginning of step.

        Args:
            run_context (RunContext): Context of the process running.
        """
        self.step_time = time.time()
        self.run_context = run_context
        if self.mstx_enabled:
            cb_params = run_context.original_args()
            step_num = cb_params.cur_step_num
            self.mstx_range_id = ms.profiler.mstx.range_start(f'step {step_num}', ms.runtime.current_stream())

    def on_train_step_end(self, run_context):
        """
        Print training info at the end of step.

        Args:
            run_context (RunContext): Context of the process running.
        """
        parallel_mode = get_auto_parallel_context("parallel_mode")
        full_batch = get_auto_parallel_context("full_batch")
        auto_parallel = parallel_mode in ['semi_auto_parallel', 'auto_parallel']
        if auto_parallel:
            set_auto_parallel_context(parallel_mode='data_parallel', full_batch=False)
        cb_params = run_context.original_args()
        step_seconds = (time.time() - self.step_time) * 1000
        if self.mstx_enabled:
            ms.profiler.mstx.range_end(self.mstx_range_id)
        net_outputs = cb_params.net_outputs
        loss, overflow, scaling_sens, learning_rate, global_norm = _get_loss_output(net_outputs)
        if learning_rate is not None:
            self.learning_rate = learning_rate
        loss = self._fix_loss_for_parallel(loss)
        self.loss_list.append(loss)

        lm_loss, aux_loss, mtp_loss = None, None, None
        if self.print_separate_loss:
            lm_loss, aux_loss, mtp_loss = _get_separate_loss()
            lm_loss = self._fix_loss_for_parallel(lm_loss, print_warning=False)
            aux_loss = self._fix_loss_for_parallel(aux_loss, print_warning=False)
            mtp_loss = self._fix_loss_for_parallel(mtp_loss, print_warning=False)

        if not overflow:
            overflow = "False"
        if not scaling_sens:
            scaling_sens = "unavailable"

        if self.mf_support is None:
            self.mf_support = self._can_calculate_model_flops(cb_params)
        if (not self.mf_calculated or check_arf_status(cb_params)) and self.mf_support:
            self._calculate_model_flops()

        origin_epochs = self.origin_epochs

        if cb_params.get('initial_step', None) is not None:
            self.initial_step = cb_params.initial_step

        if cb_params.dataset_sink_mode:
            per_step_seconds = step_seconds / cb_params.batch_num
            steps_per_epoch = self.steps_per_epoch
            cur_epoch_num = (cb_params.cur_step_num + self.initial_step - 1) // steps_per_epoch + 1
            cur_step_num = (cb_params.cur_step_num + self.initial_step - 1) % steps_per_epoch + 1
        else:
            per_step_seconds = step_seconds
            steps_per_epoch = cb_params.batch_num
            cur_epoch_num = cb_params.cur_epoch_num
            cur_step_num = (cb_params.cur_step_num + self.initial_step - 1) % steps_per_epoch + 1

        # compute time remaining
        step_remain = (origin_epochs - cur_epoch_num + 1) * steps_per_epoch - cur_step_num
        time_remain = step_remain * per_step_seconds / 1000

        # compute throughput
        throughput = self.global_batch_size / self.device_num / (per_step_seconds / 1000)

        # compute percent
        percent = ((cur_epoch_num - 1) * steps_per_epoch + cur_step_num) / origin_epochs / steps_per_epoch * 100

        step_diff = cb_params.cur_step_num - self.last_print_time
        if step_diff >= self.per_print_times or step_diff <= 0:
            self.last_print_time = cb_params.cur_step_num
            self.print_output_info(cb_params, cur_epoch_num, origin_epochs, throughput,
                                   cur_step_num, steps_per_epoch, loss, per_step_seconds,
                                   overflow, scaling_sens, time_remain, percent, global_norm,
                                   lm_loss, aux_loss, mtp_loss)

        if auto_parallel:
            set_auto_parallel_context(parallel_mode=parallel_mode, full_batch=full_batch)

    def _fix_loss_for_parallel(self, loss, print_warning=True):
        """Fix loss value in pipeline or double parallel mode."""
        pipeline_stages = ms.context.get_auto_parallel_context("pipeline_stages")
        self.is_zbv = ms.get_auto_parallel_context("pipeline_scheduler") == "zero_bubble_v"
        if self.is_zbv and self.print_warning_flag and print_warning:
            logger.warning("When zero_bubble_v is enabled, loss is valid only on rank 0")
        else:
            if pipeline_stages > 1 and self.print_warning_flag and print_warning:
                logger.warning("pipeline stages: %s > 1, the loss on the last card is valid.",
                               pipeline_stages)

        if self.micro_batch_interleave_num > 1 and self.print_warning_flag and print_warning:
            logger.warning("micro_batch_interleave_num: %s > 1, multiple copies in parallel is open.",
                           self.micro_batch_interleave_num)

        if pipeline_stages > 1 and not self.calculate_per_token_loss:
            loss = loss / self.mirco_size
        if self.micro_batch_interleave_num > 1:
            loss = loss / self.micro_batch_interleave_num
        if self.gradient_accumulation_steps > 1 and not self.calculate_per_token_loss:
            loss = loss / self.gradient_accumulation_steps

        return loss

    @staticmethod
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
        rank_str_list = [str(r) for r in rank_list]

        rank_list_str = "-".join(rank_str_list)
        return rank_list, rank_list_str

    def _can_calculate_model_flops(self, cb_params):
        """
        Check whether the model flops can be collected
        """
        if cb_params.mode == 'train':
            network = cb_params.train_network
        elif cb_params.mode == 'eval':
            network = cb_params.eval_network
        else:
            logger.warning('Model Flops computation only support train and eval mode!')
            return False
        if ms.get_context('mode') != ms.GRAPH_MODE:
            logger.warning('Model Flops computation only support graph mode!')
            return False
        if not hasattr(network, 'current_phase'):
            logger.warning('This model dose not support collecting model flops now!')
            return False
        self.current_phase = network.current_phase
        return True

    def _calculate_model_flops(self):
        """
        Calculate the full model flops
        """
        try:
            full_model_flops, _, shard_model_flops, \
                _, is_dynamic_shape = flops_collection(self.current_phase)
        except RuntimeError as e:
            logger.warning("%s", e)
            self.mf_support = False
            return
        self.full_model_flops = full_model_flops / 1.0
        self.mf_calculated = True
        if auto_parallel_context().get_pipeline_stages() > 1:
            pipeline_group_list, pipeline_group_name = self._get_pipeline_group()
            hashed = hashlib.sha256(
                pipeline_group_name.encode()).hexdigest()[:48]
            pipeline_group_name = str(hashed)
            create_group(pipeline_group_name, pipeline_group_list)

            is_dynamic_shape = AllReduceNet(pipeline_group_name)(
                Tensor([int(is_dynamic_shape)], dtype=ms.int32)).asnumpy()[0]
            if is_dynamic_shape > 0:
                logger.warning("Model Flops computation now do not support dynamic shape.")
                self.mf_support = False
                return

            self.full_model_flops = AllReduceNet(pipeline_group_name)(
                Tensor([self.full_model_flops])).asnumpy()[0]

        if is_dynamic_shape:
            logger.warning("Model Flops computation now do not support dynamic shape.")
            self.mf_support = False
            return
        if auto_parallel_context().get_parallel_mode() != "stand_alone":
            self.full_model_flops = self.full_model_flops / get_group_size()

        logger.info("Full model flops is %d, Shard model flops is %d.",
                    full_model_flops, shard_model_flops)

    def print_output_info(self, cb_params, cur_epoch_num, origin_epochs, throughput,
                          cur_step_num, steps_per_epoch, loss, per_step_seconds,
                          overflow, scaling_sens, time_remain, percent, global_norm, main_loss, extra_loss, mtp_loss):
        """print output information."""
        if self.learning_rate is not None:
            if isinstance(self.learning_rate, (float, Tensor, np.ndarray)):
                current_lr = str(self.learning_rate)
            elif isinstance(self.learning_rate, LearningRateSchedule):
                if ms.context.get_context('device_target') == 'CPU':
                    if self.print_warning_flag:
                        logger.warning(
                            "device target not support CPU when generating the learning rate value, "
                            "please use: mindspore.context.set_context(device_target='Ascend')")
                        self.print_warning_flag = False
                    current_lr = None
                else:
                    if cb_params.optimizer is not None:
                        global_step = cb_params.optimizer.global_step
                    else:
                        global_step = cb_params.network.optimizer.global_step

                    # temporary set_train to avoid error on Atlas 800T A2
                    origin_phase = cb_params.train_network.phase
                    cb_params.train_network.set_train(False)
                    current_lr = self.learning_rate(global_step)
                    cb_params.train_network.set_train(origin_phase)

                    current_lr = np.array2string(current_lr.asnumpy())
            else:
                if self.print_warning_flag:
                    logger.warning(
                        "The current learning rate cannot be calculated in real time."
                        "Only the type of LearningRateSchedule is supported in the callback of MFLossMonitor,"
                        "but the input learning rate function type is %s", type(self.learning_rate)
                    )
                    self.print_warning_flag = False
                current_lr = None
        else:
            if self.print_warning_flag:
                logger.warning(
                    "MFLossMonitor callback is not set learning rate arguments."
                    "To display the learning rate, you must input the arguments, "
                    "which can be LearningRateSchedule or a fixed float"
                )
                self.print_warning_flag = False
            current_lr = None

        global_step = cur_step_num + (cur_epoch_num - 1) * steps_per_epoch
        if self.mf_calculated:
            throughput_per_npu = self.full_model_flops / per_step_seconds / 1e9
            throughput_info = ', train_throughput_per_npu: %.3fT' % (throughput_per_npu)
            if self.tensor_writer is not None:
                self.tensor_writer.add_scalar('model-flops-throughput-per-npu',
                                              float(throughput_per_npu),
                                              global_step=global_step)
        else:
            throughput_info = ''

        if cb_params.dataset_sink_mode:
            loss = "loss: %5.3f, " % loss
        else:
            loss = "loss:[%5.3f/%5.3f], " % (loss, np.mean(self.loss_list))
        if self.print_separate_loss:
            separate_loss = "lm_loss: %5.3f, " % main_loss
            if self.is_moe_model:
                separate_loss += "aux_loss: %5.3f, " % extra_loss
            if self.is_mtp_model:
                separate_loss += "mtp_loss: %5.3f, " % mtp_loss
        else:
            separate_loss = ""
        if current_lr is not None:
            logger.info("{ Epoch:[%3d/%3d], step:[%5d/%5d], " + loss + separate_loss +
                        "per_step_time: %dms, lr: %s, overflow cond: %s, loss_scale: %s, global_norm: %s%s",
                        cur_epoch_num, origin_epochs, cur_step_num, steps_per_epoch,
                        int(per_step_seconds), current_lr, overflow, scaling_sens, global_norm, throughput_info)
            if self.tensor_writer is not None:
                self.tensor_writer.add_scalar('learning-rate', float(current_lr), global_step=global_step)
                self.tensor_writer.add_scalar('learning-rate vs samples', float(current_lr),
                                              global_step=global_step * self.global_batch_size)
        else:
            logger.info("{ Epoch:[%3d/%3d], step:[%5d/%5d], " + loss + separate_loss +
                        "per_step_time: %dms, overflow cond: %s, loss_scale: %s, global_norm: %s%s",
                        cur_epoch_num, origin_epochs, cur_step_num, steps_per_epoch,
                        int(per_step_seconds), overflow, scaling_sens, global_norm, throughput_info)

        # print progress bar
        show_str = ('|%%-%ds|' % 50) % (int(50 * percent / 100) * "█")
        logger.info("  %4.1f%% %s %.5f samples/s/p  %s }", percent, show_str, throughput,
                    timedelta(seconds=int(time_remain)))

        # write tensorboard
        if self.tensor_writer is not None:
            self.tensor_writer.add_scalar('batch-size', self.global_batch_size, global_step=global_step)
            self.tensor_writer.add_scalar('batch-size vs samples', self.global_batch_size,
                                          global_step=global_step * self.global_batch_size)
            self.tensor_writer.add_scalar('loss', loss, global_step=global_step)
            self.tensor_writer.add_scalar('loss vs samples', loss,
                                          global_step=global_step * self.global_batch_size)
            if self.tensorboard.get('log_loss_scale_to_tensorboard', False):
                self.tensor_writer.add_scalar('loss-scale', scaling_sens, global_step=global_step)
                self.tensor_writer.add_scalar('loss-scale vs samples', scaling_sens,
                                              global_step=global_step * self.global_batch_size)
            self.tensor_writer.add_scalar('grad-norm', global_norm, global_step=global_step)
            self.tensor_writer.add_scalar('grad-norm vs samples', global_norm,
                                          global_step=global_step * self.global_batch_size)
            if self.tensorboard.get('log_timers_to_tensorboard', False):
                self.tensor_writer.add_scalar('iteration-time', int(per_step_seconds),
                                              global_step=global_step)
                self.tensor_writer.add_scalar('iteration-time vs samples', int(per_step_seconds),
                                              global_step=global_step * self.global_batch_size)
                self.tensor_writer.add_scalar('throughput', throughput, global_step=global_step)
                seconds_per_day = 86400
                billion_samples_per_day = throughput * get_group_size() * seconds_per_day / 1e9
                self.tensor_writer.add_scalar('B-samples-per-day', billion_samples_per_day, global_step=global_step)
                self.tensor_writer.add_scalar('throughput vs samples', throughput,
                                              global_step=global_step * self.global_batch_size)
            if self.print_separate_loss:
                self.tensor_writer.add_scalar('lm-loss', main_loss, global_step=global_step)
                if self.is_mtp_model:
                    self.tensor_writer.add_scalar('mtp-loss', mtp_loss, global_step=global_step)
                if self.is_moe_model:
                    self.tensor_writer.add_scalar('aux-loss', extra_loss, global_step=global_step)


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class TrainingStateMonitor(Callback):
    """
    Monitor metrics such as local norm and local loss in training process.

    Args:
        origin_epochs (int): Required. Training epoches.
        config (dict, optional): The config specified how to display metrics. Keys are shown below. Default: ``None``,
            mean that keys will be set as the default values as below.

            - target: Specify the name or regular expression of params to monitor.
              Must be list of str, e.g. ["layers.[01]", "attention"]. Default: ['*'] , all params are selected.

            - invert: Whether to invert `target`, i.e. params in `target` won't be monitored.
              Must be `bool`.  Default: `False`

            - local_norm_format: Determine where to display the local norm.
              Should be a `str` in ['tensorboard', 'log'] (mean that write data to tensorboard or log file),
              or a `list` containing them,  or ``None``. Only params specified will be monitored.
              may cause a large amount of print info if 'log' is selected.
              Set to ``None`` to ignore this metric. Default: ``None``.

            - device_local_norm_format: Determine where to display the device local norm.
              Should be a `str` in ['tensorboard', 'log'] (mean that write data to tensorboard or log file),
              or a `list` containing them,  or ``None``. Set to ``None`` to ignore this metric. Default: ``None``.

            - local_loss_format: Determine where to display the local loss.
              Should be a `str` in ['tensorboard', 'log'] (mean that write data to tensorboard or log file),
              or a `list` containing them,  or ``None``. Set to ``None`` to ignore this metric.
              Default: ``None``.

            - device_local_loss_format: Determine where to display the device local loss.
              Should be a `str` in ['tensorboard', 'log'] (mean that write data to tensorboard or log file),
              or a `list` containing them,  or ``None``. Set to ``None`` to ignore this metric.
              Default: ``None``.

            - optimizer_state_format: Determine where to display the optimizer state.
              Should be a `str` in ['tensorboard', 'log'] (mean that write data to tensorboard or log file),
              or a `list` containing them,  or ``None``. Only the optimizer state of params specified
              will be monitored, may cause a large amount of print info if 'log' is selected.
              Set to ``None`` to ignore this metric. Default: ``None``.

            - weight_state_format: Determine where to display the weight L2-norm.
              Should be a `str` in ['tensorboard', 'log'] (mean that write data to tensorboard or log file),
              or a `list` containing them,  or ``None``. Set to ``None`` to ignore this metric.
              Default: ``None``.

            - throughput_baseline: The model throughput baseline to calculate linearity. Must be a positive number.
              Will be displayed both to tensorboard and log. Set to ``None`` to ignore this metric. Default: ``None``.

            - print_struct: Whether to print the structure of model. If ``True``, callback will print the names of all
              trainable params at the first step and then quit training process. Default: ``False``.

        step_interval (int, optional): Every how many steps to display metrics. Default: ``1``.
        dataset_size (int, optional): Required in sink mode. Training dataset size. Default: ``None``.
        initial_epoch (int, optional): The beginning epoch. Default: ``0``.
        initial_step (int, optional): The beginning step. Default: ``0``.
        global_batch_size (int, optional): The total batch size. Default: ``0``.
        check_for_nan_in_loss_and_grad (bool, optional): Whether to check loss and norm of grad is Nan.
            Default: ``False``.
        use_skip_data_by_global_norm (bool, optional): Whether to use the skip data function
            by global norm. Default: ``False``.
        embedding_size (int, optional): The size of embedding norm which is get
            by hidden_size * vocab_size. Default: ``4096``.
        use_local_norm (bool, optional): Whether to turn on the local norm. Default: ``False``.
            Default: ``False``.
    """
    @args_type_check(embedding_size=int, use_skip_data_by_global_norm=bool)
    def __init__(self,
                 origin_epochs: int,
                 config: dict = None,
                 step_interval: int = 1,
                 dataset_size: int = None,
                 initial_epoch: int = 0,
                 initial_step: int = 0,
                 global_batch_size: int = 0,
                 check_for_nan_in_loss_and_grad: bool = False,
                 use_skip_data_by_global_norm: bool = False,
                 embedding_size: int = 4096,
                 use_local_norm: bool = False):
        super(TrainingStateMonitor, self).__init__()
        if not (isinstance(step_interval, int) and step_interval > 0):
            logger.warning(f"The value of 'monitor_config.step_interval' should be positive integer, "
                           f"but get {step_interval}. Use default value: 1.")
            step_interval = 1
        self.step_interval = step_interval
        self.last_print_time = 0
        self.step_time = time.time()
        self.epoch_time = time.time()
        self.run_context = None
        self.steps_per_epoch = dataset_size
        self.origin_epochs = origin_epochs
        self.initial_epoch = initial_epoch
        self.initial_step = initial_step
        self.global_batch_size = global_batch_size
        self.global_norm_spike_count = 0
        self.use_skip_data_by_global_norm = use_skip_data_by_global_norm
        self.embedding_size = embedding_size
        self.use_local_norm = use_local_norm
        self.device_num = get_real_group_size()
        self.tensor_writer = get_tensorboard_writer()
        self.outputer = {'tensorboard': self._to_tensorboard, 'log': self._to_log}
        self._init_config(config)
        self.dump_path = None
        self.check_for_nan_in_loss_and_grad = check_for_nan_in_loss_and_grad
        if get_auto_parallel_context("dump_local_norm_path"):
            self.dump_path = os.path.join(get_auto_parallel_context("dump_local_norm_path"), f'rank_{get_real_rank()}')
            self.dump_key = {0: -1}
            self.dump_step = step_interval
            if is_version_ge(ms.__version__, '2.5.0'):
                self.dump_name_mode = 0
                self.finish_pattern = 'finish_step_*_*'
                self.local_loss_pattern = re.compile('(local_loss)__(.+)_[a-z]+[0-9]+_([0-9]+)')
                self.local_norm_pattern = re.compile('(local_norm)__(.+)_[a-z]+[0-9]+_([0-9]+)')
                self.device_local_norm_pattern = re.compile('(device_local_norm)_[a-z]+[0-9]+_([0-9]+)')
            else:
                self.dump_name_mode = 1
                self.finish_pattern = '*_finish_step'
                self.local_loss_pattern = re.compile('([0-9]+)_(local_loss)__(.+)')
                self.local_norm_pattern = re.compile('([0-9]+)_(local_norm)__(.+)')
                self.device_local_norm_pattern = re.compile('([0-9]+)_(device_local_norm)')

    def on_train_epoch_begin(self, run_context):
        """
        Record time at the beginning of epoch.

        Args:
            run_context (RunContext): Context of the process running.
        """
        self.epoch_time = time.time()
        self.run_context = run_context

    def on_train_step_begin(self, run_context):
        """
        Record time at the beginning of step.

        Args:
            run_context (RunContext): Context of the process running.
        """
        self.step_time = time.time()
        self.run_context = run_context
        if self.print_struct:
            network = run_context.original_args().network
            if isinstance(network, ms.nn.TrainOneStepCell):
                network = network.network
            for param in network.trainable_params():
                logger.info(param.name)
            self.run_context.request_stop()

    def on_train_step_end(self, run_context):
        """
        Print training info at the end of step.

        Args:
            run_context (RunContext): Context of the process running.
        """
        if self.print_struct:
            self._clear_dump_path()
            return
        step_seconds = (time.time() - self.step_time) * 1000
        parallel_mode = get_auto_parallel_context("parallel_mode")
        full_batch = get_auto_parallel_context("full_batch")
        auto_parallel = parallel_mode in ['semi_auto_parallel', 'auto_parallel']
        if auto_parallel:
            set_auto_parallel_context(parallel_mode='data_parallel', full_batch=False)
        cb_params = run_context.original_args()
        if cb_params.dataset_sink_mode:
            per_step_seconds = step_seconds / cb_params.batch_num
        else:
            self.steps_per_epoch = cb_params.batch_num
            per_step_seconds = step_seconds
        step_diff = cb_params.cur_step_num - self.last_print_time
        if step_diff >= self.step_interval or step_diff <= 0:
            self.last_print_time = cb_params.cur_step_num
            if get_auto_parallel_context("dump_local_norm_path"):
                self._dump_data_in_step(cb_params.cur_step_num)
            if self.optimizer_state_format:
                self._dump_optimizer_state(cb_params)
            if self.weight_state_format:
                network = cb_params.network
                if isinstance(network, ms.nn.TrainOneStepCell):
                    network = network.network
                weight_norm = _get_weight_norm(network)
                self._output('weight_norm', weight_norm, cb_params.cur_step_num, self.weight_state_format)
            if self.throughput_baseline is not None:
                # compute throughput
                throughput = self.global_batch_size / self.device_num / (per_step_seconds / 1000)
                linearity = throughput / self.throughput_baseline
                self._output('throughput_linearity', linearity, cb_params.cur_step_num, ['log', 'tensorboard'])
            if self.device_local_loss_format:
                for loss_tag, device_local_loss in get_device_local_loss(None).items():
                    device_local_loss = np.mean(device_local_loss.asnumpy())
                    self._output(f'device_accum_local_{loss_tag}_loss', device_local_loss, cb_params.cur_step_num,
                                 self.device_local_loss_format)

        if auto_parallel:
            set_auto_parallel_context(parallel_mode=parallel_mode, full_batch=full_batch)

        if self.use_local_norm and self.embedding_size is not None:
            embedding_local_norm = get_embedding_info(cb_params, self.embedding_size)
            logger.info("embedding_local_norm: %s", embedding_local_norm)

        self.abnormal_global_norm_check(cb_params)

        # Boundary check.
        if self.check_for_nan_in_loss_and_grad:
            loss, global_norm, local_norm = self._get_loss_output(cb_params.net_outputs)
            check_device_local_loss()
            self._check_nan_or_inf(loss, 'loss')
            self._check_nan_or_inf(global_norm, 'global_norm')
            self._check_nan_or_inf(local_norm, 'local_norm')

        if self.device_local_loss_format:
            reset_device_local_loss()

    def abnormal_global_norm_check(self, cb_params):
        """Check the abnormal global_norm and raise error"""
        if cb_params.get('initial_step', None) is not None:
            self.initial_step = cb_params.initial_step

        if cb_params.dataset_sink_mode:
            steps_per_epoch = self.steps_per_epoch
            cur_epoch_num = (cb_params.cur_step_num + self.initial_step - 1) // steps_per_epoch + 1
            cur_step_num = (cb_params.cur_step_num + self.initial_step - 1) % steps_per_epoch + 1
        else:
            steps_per_epoch = cb_params.batch_num
            cur_epoch_num = cb_params.cur_epoch_num
            cur_step_num = (cb_params.cur_step_num + self.initial_step - 1) % steps_per_epoch + 1
        net_outputs = cb_params.net_outputs
        global_norm = self._get_loss_output(net_outputs)[1]
        global_step = cur_step_num + (cur_epoch_num - 1) * steps_per_epoch

        if self.check_for_global_norm and self.use_skip_data_by_global_norm:
            raise ValueError("The check_for_global_norm and use_skip_data_by_global_norm"
                             " cannot be turned on at the same time, please choose one.")

        if self.check_for_global_norm and global_norm >= self.global_norm_spike_threshold:
            if str(global_step) not in self.abnormal_global_norms:
                # Because json cannot use number as key, so we convert it to string
                self.abnormal_global_norms[str(global_step)] = [global_norm.item()]
                if get_rank() == 0:
                    parent_dirs = os.path.dirname(self.global_norm_record_path)
                    if not os.path.exists(parent_dirs):
                        os.makedirs(parent_dirs)
                    with open(self.global_norm_record_path, 'w') as file:
                        json.dump(self.abnormal_global_norms, file)
                    set_safe_mode_for_file_or_dir(self.global_norm_record_path)
                logger.info(f"Current global norm {global_norm} is greater equal than "
                            f"threshold {self.global_norm_spike_threshold}, stop training...")
                barrier_world()
                logger.info(f"Call barrier before throw TREError.")
                ms.runtime.synchronize()
                logger.info(f"All stream execution completed.")
                raise RuntimeError("TREError occurred......")
            self.abnormal_global_norms[str(global_step)].append(global_norm.item())
            logger.info(f"The global norm {global_norm} of step {global_step} is still greater or equal "
                        f"than threshold {self.global_norm_spike_threshold}, continue training.")

        if self.use_skip_data_by_global_norm:
            opt_global_step = cb_params.optimizer.global_step \
                if cb_params.optimizer is not None else cb_params.network.optimizer.global_step
            is_skip = global_norm >= self.global_norm_spike_threshold
            if is_skip:
                logger.info("opt_global_step: %d, skip_data_grad_norm_threshold: %s, is_skip: %s",
                            opt_global_step, self.global_norm_spike_threshold, is_skip)
                self.global_norm_spike_count += 1
                if self.global_norm_spike_count < self.global_norm_spike_count_threshold:
                    logger.info(f"Current global norm {global_norm} of step {global_step} "
                                f"has been {self.global_norm_spike_count} "
                                f"consecutive times greater than threshold: "
                                f"{self.global_norm_spike_threshold}")
                else:
                    raise ValueError(
                        f"Current global norm {global_norm} of step {global_step} "
                        f"has been {self.global_norm_spike_count_threshold} "
                        f"consecutive times greater than threshold "
                        f"{self.global_norm_spike_threshold}, stop training...")
            else:
                self.global_norm_spike_count = 0

    def _init_config(self, config):
        """Initialize members from config"""
        if config is None:
            logger.warning("The param `config` of TrainingStateMonitor is not set. Will use the default config.")
            config = {}
        if not isinstance(config, dict):
            raise TypeError("The param `config` of TrainingStateMonitor should be a dict.")
        self.target = config.get('target') or ['.*']
        self.invert = config.get('invert')
        if self.invert is None:
            self.invert = False
        self.target_cache = {}
        self.local_norm_format = config.get('local_norm_format', None)
        self.local_loss_format = config.get('local_loss_format', None)
        self.device_local_norm_format = config.get('device_local_norm_format', None)
        self.device_local_loss_format = \
            config.get('device_local_loss_format', None) if is_last_pipeline_stage() else None
        self.optimizer_state_format = config.get('optimizer_state_format', None)
        self.weight_state_format = config.get('weight_state_format', None)
        self.throughput_baseline = config.get('throughput_baseline', None)
        self.print_struct = config.get('print_struct')

        self.check_for_global_norm = config.get('check_for_global_norm')
        self.global_norm_record_path = os.path.join(get_output_root_path(), "abnormal_global_norm.json")
        self.global_norm_spike_threshold = config.get('global_norm_spike_threshold')
        self.global_norm_spike_count_threshold = config.get('global_norm_spike_count_threshold')
        self.abnormal_global_norms: dict[str, list[float]] = {}

        if self.print_struct is None:
            self.print_struct = False
        if not (isinstance(self.target, list) and self.target and all([isinstance(i, str) for i in self.target])):
            raise TypeError(f"The value of 'target' should be a list of str.")
        if not isinstance(self.invert, bool):
            raise TypeError(f"The value of 'invert' should be bool.")
        if (self.throughput_baseline is not None and
                not (isinstance(self.throughput_baseline, (int, float)) and self.throughput_baseline > 0)):
            raise ValueError(f"The value of 'throughput_baseline' should be None or positive number.")
        if not isinstance(self.print_struct, bool):
            raise TypeError(f"The value of 'print_struct' should be bool.")
        attrs = ['local_norm_format', 'local_loss_format', 'device_local_norm_format', 'device_local_loss_format',
                 'optimizer_state_format', 'weight_state_format']
        for attr in attrs:
            self._check_attr_formats(attr)
        if self.global_norm_record_path and os.path.exists(self.global_norm_record_path):
            # the data format might be like {"300": [3.3], "600": [4.1, 4.2],}
            # because json cannot use number as key, we convert it to string
            with open(self.global_norm_record_path, 'r') as file:
                self.abnormal_global_norms = json.load(file)

    def _check_attr_formats(self, attr):
        """Check the validation of formats in config"""
        if getattr(self, attr):
            if not isinstance(getattr(self, attr), (str, list)):
                raise TypeError(f"The value of {attr} should be a `str` in 'tensorboard' or 'log', "
                                f"or a list containing them, or None, but get type {type(getattr(self, attr))}")
            if isinstance(getattr(self, attr), str):
                setattr(self, attr, set([getattr(self, attr)]))
            else:
                setattr(self, attr, set(getattr(self, attr)))
            diff = getattr(self, attr) - {'tensorboard', 'log'}
            if diff:
                raise ValueError(f"The value of {attr} should be a `str` in 'tensorboard' or 'log', "
                                 f"or a list containing them, or None, but get unexpected value {diff}")
        else:
            setattr(self, attr, None)
        if self.tensor_writer is None and getattr(self, attr):
            logger.warning("Tensorboard config is unset. '%s' will use 'log' only.", attr)
            getattr(self, attr).discard('tensorboard')
            getattr(self, attr).add('log')

    def _parse_step(self):
        """record the finish dump id of each step"""

        def check_step(pattern, id_pos):
            search_path = glob.glob(os.path.join(self.dump_path, f'{pattern}.npy'))
            if not search_path:
                return None
            step_ids = []
            for f in search_path:
                tag, _ = os.path.splitext(os.path.basename(f))
                step_ids.append(int(tag.split('_')[id_pos]))
                os.remove(f)
            step_ids.sort()
            return step_ids

        step_ids = check_step(self.finish_pattern, self.dump_name_mode - 1)
        if step_ids is None:
            return
        cur_steps = len(self.dump_key)
        for i, step_id in enumerate(step_ids):
            self.dump_key[cur_steps + i] = step_id

    def _dump_data_in_step(self, global_step):
        """write the dumped data each step to tensorboard"""

        def match_pattern(pattern, filename):
            name, _ = os.path.splitext(os.path.basename(filename))
            parsed = re.fullmatch(pattern, name)
            if parsed is None:
                return None, None, None
            groups = parsed.groups()
            dump_id = int(groups[self.dump_name_mode - 1])
            prefix = groups[self.dump_name_mode]
            suffix = None if len(groups) < 3 else groups[self.dump_name_mode + 1]
            return dump_id, prefix, suffix

        self._parse_step()
        while self.dump_step <= global_step and self.dump_key.get(self.dump_step) is not None:
            begin_id = self.dump_key[self.dump_step - 1]
            end_id = self.dump_key[self.dump_step]
            file_list = os.listdir(self.dump_path)
            local_losses = {}
            for f in file_list:
                parsed_name = None, None, None
                if self.local_norm_format:
                    parsed_name = match_pattern(self.local_norm_pattern, f)
                if not any(parsed_name) and self.device_local_norm_format:
                    parsed_name = match_pattern(self.device_local_norm_pattern, f)
                if not any(parsed_name) and self.local_loss_format:
                    parsed_name = match_pattern(self.local_loss_pattern, f)
                if not any(parsed_name):
                    continue
                dump_id, prefix, suffix = parsed_name
                if not begin_id < dump_id < end_id:
                    continue
                data = np.load(os.path.join(self.dump_path, f), allow_pickle=False)
                if prefix == 'device_local_norm':
                    self._output(f'device_local_norm', data, self.dump_step, self.device_local_norm_format)
                elif prefix == 'local_loss':
                    # collect all local loss if there are more than one local loss within one step
                    local_losses[suffix] = local_losses.get(suffix, [])
                    local_losses[suffix].append(data)
                elif prefix == 'local_norm' and self._check_param_name(suffix):
                    self._output(f'local_norm/{suffix}', data, self.dump_step, self.local_norm_format)
            if local_losses and self.local_loss_format:
                self._dump_local_loss(local_losses)
            self._clear_dump_path()
            self.dump_step += self.step_interval

    def _dump_local_loss(self, local_losses):
        """write the local loss to log/tensorboard"""
        # log local loss of each micro
        for loss_tag, loss_list in local_losses.items():
            if 'log' in self.local_loss_format:
                for local_loss in loss_list:
                    self._output(f'micro_local_{loss_tag}_loss', local_loss, self.dump_step, ['log'])
            if 'tensorboard' in self.local_loss_format:
                self._output(f'local_{loss_tag}_loss', np.mean(loss_list), self.dump_step, ['tensorboard'])

    def _dump_optimizer_state(self, cb_params):
        """write the optimizer state to tensorboard"""
        optimizer = cb_params.optimizer
        if optimizer is None:
            optimizer = getattr(cb_params.network, "optimizer", None)

        if hasattr(optimizer, "moment1") and hasattr(optimizer, "moment2"):
            adam_m, adam_v = optimizer.moment1, optimizer.moment2
        elif hasattr(optimizer, "moments1") and hasattr(optimizer, "moments2"):
            adam_m, adam_v = optimizer.moments1, optimizer.moments2
        elif hasattr(optimizer, "exp_avg") and hasattr(optimizer, "exp_avg_sq"):
            adam_m, adam_v = optimizer.exp_avg, optimizer.exp_avg_sq
        else:
            return
        global_step = cb_params.cur_step_num
        adam_m_norms = _get_optimizer_state(adam_m, self._check_param_name)
        adam_v_norms = _get_optimizer_state(adam_v, self._check_param_name)
        for param_name, adam_m_norm in adam_m_norms.items():
            param_name = param_name.split('.', maxsplit=1)[1]
            self._output(f'adam_m_norm/{param_name}', adam_m_norm, global_step, self.optimizer_state_format)
        for param_name, adam_v_norm in adam_v_norms.items():
            param_name = param_name.split('.', maxsplit=1)[1]
            self._output(f'adam_v_norm/{param_name}', adam_v_norm, global_step, self.optimizer_state_format)

    def _check_param_name(self, param_name):
        if self.target_cache.get(param_name) is None:
            for pattern in self.target:
                if re.search(pattern, param_name) is not None:
                    self.target_cache[param_name] = not self.invert
                    return not self.invert
            self.target_cache[param_name] = self.invert
            return self.invert
        return self.target_cache[param_name]

    def _clear_dump_path(self):
        if not self.dump_path:
            return
        file_list = os.listdir(self.dump_path)
        for f in file_list:
            os.remove(os.path.join(self.dump_path, f))

    def _to_tensorboard(self, tag, data, global_step):
        """Write data to tensorboard if possible"""
        if self.tensor_writer is not None:
            self.tensor_writer.add_scalar(tag, data, global_step=global_step)

    def _to_log(self, tag, data, global_step):
        """Write data to log file"""
        cur_epoch_num = (global_step + self.initial_step - 1) // self.steps_per_epoch + 1
        cur_step_num = (global_step + self.initial_step - 1) % self.steps_per_epoch + 1
        logger.info("Epoch:[%3d/%3d], step:[%5d/%5d] %s: %.4f",
                    cur_epoch_num, self.origin_epochs, cur_step_num, self.steps_per_epoch, tag, data)

    def _output(self, tag, data, global_step, formats):
        """Write data in specified formats"""
        if formats:
            for fmt in formats:
                self.outputer[fmt](tag, data, global_step)

    def _get_loss_output(self, output):
        """Get loss, global/local norm"""
        loss = output
        global_norm = None
        local_norm = None
        if isinstance(output, (tuple, list)):
            if len(output) == 7:
                loss, global_norm, local_norm = output[0], output[4], output[5]
            elif len(output) == 5:
                loss, global_norm = output[0], output[4]
            elif isinstance(output[0], ms.Tensor) and isinstance(output[0].asnumpy(), np.ndarray):
                loss = output[0]

        if isinstance(global_norm, ms.Tensor):
            global_norm = global_norm.asnumpy()

        if isinstance(local_norm, ms.Tensor):
            local_norm = local_norm.asnumpy()

        if isinstance(loss, ms.Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        return loss, global_norm, local_norm

    @staticmethod
    def _check_nan_or_inf(indicator, indicator_name):
        """Check if Nan or Inf in indicator then terminate training"""
        if indicator is not None:
            if np.any(np.isnan(indicator)):
                raise ValueError(f"There is nan in {indicator_name} with value {indicator}, terminate training.")
            if np.any(np.isinf(indicator)):
                raise ValueError(f"There is inf in {indicator_name} with value {indicator}, terminate training.")


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class SummaryMonitor:
    """
    Summary Monitor can help you to collect some common information, such as loss,
    learning late, computational graph and so on.

    Note:
        referring to
        `note <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.SummaryCollector.html>`_ .

    Args:
        summary_dir (str, optional):
            The collected data will be persisted to this directory. If the directory does not exist,
            it will be created automatically. Default: ``None``.
        collect_freq (int, optional):
            Set the frequency of data collection, it should be greater than zero, and the unit is `step`.
            Default: ``10``.
        collect_specified_data (Union[None, dict], optional):
            Perform custom operations on the collected data. Default: ``None``.
        keep_default_action (bool, optional):
            This field affects the collection behavior of the 'collect_specified_data' field. Default: ``True``.
        custom_lineage_data (Union[dict, None], optional):
            Allows you to customize the data. In the custom data, the type of the key supports str,
            and the type of value supports str, int and float. Default: ``None`` , it means there is no custom data.
        collect_tensor_freq (Optional[int], optional):
            The same semantics as the `collect_freq`, but controls TensorSummary only. Default: ``None``.
        max_file_size (Optional[int], optional):
            The maximum size in bytes of each file that can be written to the disk. For example,
            to write not larger than 4GB, specify max_file_size=4*1024**3. Default: ``None``, which means no limit.
        export_options (Union[None, dict], optional):
            Perform custom operations on the export data. Default: ``None``, it means that the data is not exported.

    Examples:
        >>> from mindformers.core import SummaryMonitor
        >>> monitor = SummaryMonitor(summary_dir='./summary_dir')
    """

    def __new__(cls,
                summary_dir=None,
                collect_freq=10,
                collect_specified_data=None,
                keep_default_action=True,
                custom_lineage_data=None,
                collect_tensor_freq=None,
                max_file_size=None,
                export_options=None):
        if summary_dir is None:
            rank_id = get_real_rank()
            summary_dir = get_output_subpath('summary', rank_id)
        kwargs = {
            "summary_dir": summary_dir,
            "collect_freq": collect_freq,
            "collect_specified_data": collect_specified_data,
            "keep_default_action": keep_default_action,
            "custom_lineage_data": custom_lineage_data,
            "collect_tensor_freq": collect_tensor_freq,
            "max_file_size": max_file_size,
            "export_options": export_options
        }
        return SummaryCollector(**kwargs)


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class CheckpointMonitor(ModelCheckpoint):
    """
    Checkpoint Monitor For Save LossScale.

    Args:
        prefix (str, optional): The prefix name of checkpoint files. Default: ``'CKP'``.
        directory (str, optional): The path of the folder which will be saved in the checkpoint file. Default: ``None``.
        config (CheckpointConfig, optional): Checkpoint strategy configuration. Default: ``None``.
        save_checkpoint_steps (int, optional): Steps to save checkpoint. Default: ``1``.
        save_checkpoint_seconds (int, optional): Seconds to save checkpoint.
            Can't be used with save_checkpoint_steps at the same time. Default: ``0``.
        keep_checkpoint_max (int, optional): Maximum number of checkpoint files can be saved. Default: ``5``.
        keep_checkpoint_per_n_minutes (int, optional): Save the checkpoint file every "keep_checkpoint_per_n_minutes"
            minutes. Can't be used with keep_checkpoint_max at the same time. Default: ``0``.
        integrated_save (bool, optional): Whether to merge and save the split Tensor in the automatic parallel scenario.
            Integrated save function is only supported in automatic parallel scene. Default: ``True``.
        save_network_params (bool, optional): Whether to only save network weights additionally. Default: ``False``.
        save_trainable_params (bool, optional): Whether to save only weights of trainable parameters.
            Default: ``False``.
        async_save (bool, optional): Whether asynchronous execution saves the checkpoint to a file. Default: ``False``.
        saved_network (Cell, optional): Network to be saved in checkpoint file. Default: ``None``.
        append_info (list, optional): The information save to checkpoint file.
            Support "epoch_num", "step_num" and dict. Default: ``None``.
        enc_key (Union[None, bytes], optional): Byte type key used for encryption. Default: ``None``.
        enc_mode (str, optional): This parameter is valid only when "enc_key" is not set to None. Specifies the
            encryption mode, currently supports 'AES-GCM', 'AES-CBC' and 'SM4-CBC'. Default: ``'AES-GCM'``.
        exception_save (bool, optional): Whether to save the current checkpoint when an exception occurs.
            Default: ``False``.
        global_batch_size (int, optional): The total batch size. Default: ``0``.
        checkpoint_format (str, optional): The format of checkpoint to save. Support 'ckpt' or 'safetensors'.
            Default: ``'ckpt'``.
        remove_redundancy (bool, optional): Whether to remove redundancy when saving checkpoint. Default: ``False``.
        embedding_size (int, optional): The size of embedding norm which is get
            by hidden_size * vocab_size. Default: ``4096``.
        use_checkpoint_health_monitor (bool, optional): Whether to use the checkpoint health
            monitor function by embedding norm. Default: ``False``.
        embedding_local_norm_threshold (float, optional): The threshold of the embedding norm. Default: ``1.0``.
        health_ckpts_record_dir (str, optional): The path of the file which is used to record the health of checkpoint.
            Default: ``./output``.
        use_legacy_format (bool, optional): Whether to use the legacy 'save_checkpoint' process, Default: ``True``.
        save_optimizer (bool, optional): Whether to save optimizer weights,
            only used in megatron-format weight save scene. Legacy scene will be set to ``None``.
            Default: ``True``.
        save_checkpoint_path (str, optional): Users can specify the path to store weights.
            If None, the checkpoints will be saved at './output_dir/checkpoint'. Default: ``None``.

    Raises:
        ValueError: If `prefix` is not str or contains the '/' character.
        ValueError: If `directory` is not str.
        TypeError: If the config is not CheckpointConfig type.

    Examples:
        >>> from mindformers.core import CheckpointMonitor
        >>> monitor = CheckpointMonitor(directory='./checkpoint_dir')
    """

    @args_type_check(embedding_local_norm_threshold=float, use_checkpoint_health_monitor=bool)
    def __init__(self, prefix='CKP',
                 directory=None,
                 config=None,
                 save_checkpoint_steps=1,
                 save_checkpoint_seconds=0,
                 keep_checkpoint_max=5,
                 keep_checkpoint_per_n_minutes=0,
                 integrated_save=True,
                 save_network_params=False,
                 save_trainable_params=False,
                 async_save=False,
                 saved_network=None,
                 append_info=None,
                 enc_key=None,
                 enc_mode='AES-GCM',
                 exception_save=False,
                 global_batch_size=None,
                 checkpoint_format='ckpt',
                 remove_redundancy=False,
                 embedding_size=4096,
                 embedding_local_norm_threshold=1.0,
                 use_checkpoint_health_monitor=False,
                 health_ckpts_record_dir="./output",
                 use_legacy_format=True,
                 save_optimizer=True,
                 save_checkpoint_path=None):

        self.config = config
        self.save_network_params = save_network_params
        self.save_trainable_params = save_trainable_params
        self.rank_id = get_real_rank()
        self.embedding_local_norm_threshold = embedding_local_norm_threshold
        self.use_checkpoint_health_monitor = use_checkpoint_health_monitor
        self.embedding_size = embedding_size
        self.health_ckpts_record_dir = health_ckpts_record_dir
        self.use_legacy_format = use_legacy_format
        # Ensure that 'save_optimizer' only use in the sense of 'use_legacy_format == False'
        self.save_optimizer = save_optimizer if not use_legacy_format else None
        self.origin_prefix = prefix
        self.save_checkpoint_path = save_checkpoint_path
        self.need_remove_redundancy = remove_redundancy

        prefix = prefix + "_rank_{}".format(self.rank_id)

        self.global_batch_size = global_batch_size
        # this list records parameters which will be ignored when saving ckpt.
        self.filter_list = ['accu_grads', 'fi_parameter', 'zeros_k_pe', 'zeros_k_nope', 'zeros_value_states', '_cache',
                            '_device_local_norm', '_device_local_loss', 'expert_load']

        self.save_info_list = defaultdict(
            lambda: {
                'ckpt': {'ckpt_file_path': None, 'save_start_time': None, 'save_end_time': None},
                'network': {'ckpt_file_path': None, 'save_start_time': None, 'save_end_time': None},
                'trainable_params': {'ckpt_file_path': None, 'save_start_time': None, 'save_end_time': None},
            }
        )

        if append_info is None:
            append_info = [{
                "epoch_num": 0,
                "step_num": 0,
                "global_step": 0,
                "loss_scale": 1
            }]
        ckpt_directory = os.path.join(directory, f"checkpoint/rank_{self.rank_id}") \
            if directory else get_output_subpath('checkpoint', self.rank_id)
        self.network_directory = os.path.join(directory, f"checkpoint_network/rank_{self.rank_id}") \
            if directory else get_output_subpath('checkpoint_network', self.rank_id)
        self.trainable_directory = os.path.join(directory, f"checkpoint_trainable/rank_{self.rank_id}") \
            if directory else get_output_subpath('checkpoint_trainable', self.rank_id)
        if context.get_auto_parallel_context('parallel_mode') in \
                ['semi_auto_parallel', 'auto_parallel', 'hybrid_parallel']:
            logger.info("Integrated_save is changed to False when using auto_parallel.")
            integrated_save = False
        config_ck = CheckpointConfig(save_checkpoint_steps=save_checkpoint_steps,
                                     save_checkpoint_seconds=save_checkpoint_seconds,
                                     keep_checkpoint_max=keep_checkpoint_max,
                                     keep_checkpoint_per_n_minutes=keep_checkpoint_per_n_minutes,
                                     integrated_save=integrated_save,
                                     async_save=async_save,
                                     saved_network=saved_network,
                                     append_info=append_info,
                                     enc_key=enc_key,
                                     enc_mode=enc_mode,
                                     format=checkpoint_format,
                                     exception_save=exception_save,
                                     remove_redundancy=remove_redundancy)
        super(CheckpointMonitor, self).__init__(prefix, ckpt_directory if self.use_legacy_format else None,
                                                config=config_ck)
        self.meta_json = os.path.join(self._directory, "meta.json")
        if self._config.async_save:
            self.last_epoch_num = None
            self.last_step_num_in_epoch = None
            self.last_ckpoint_file = None
            self.meta_updated = True
            self.async_save_manager = AsyncSaveManager(self._config.async_save)

        if self.save_network_params:
            self._network_manager = CheckpointManager(config_ck.format)

        if self.save_trainable_params:
            self._trainable_manager = CheckpointManager(config_ck.format)

        self.need_remove_extra_ckpt = False
        self.common_info = CommonInfo()

    def print_savetime(self, record_step, batch_num):
        """print the time cost of saving checkpoint files."""
        epoch = int((record_step - 1) // batch_num + 1)
        step = int((record_step - 1) % batch_num + 1)

        def output_if_exists(key):
            save_info = self.save_info_list[record_step][key]
            file = save_info['ckpt_file_path']
            if file is not None and os.path.exists(file):
                save_info['save_end_time'] = os.path.getmtime(file)
                cost_time = save_info['save_end_time'] - save_info['save_start_time']
                logger.info(f'Finish saving {key} of epoch {epoch} step {step}'
                            f' using {cost_time:.3f} seconds')
                save_info['ckpt_file_path'] = None

        output_if_exists('ckpt')
        output_if_exists('network')
        output_if_exists('trainable_params')

    def _save_ckpt(self, cb_params, force_to_save=False):
        """Save checkpoint files."""
        # pylint: disable=E0203
        if cb_params.cur_step_num == self._last_triggered_step:
            return

        # if param is cache enable, flush data from cache to host before save_ckpt
        if self._need_flush_from_cache:
            self._flush_from_cache(cb_params)

        save_ckpt = self._check_save_ckpt(cb_params, force_to_save)
        # if async_save is True, check whether saving processes are completed each step
        if self._config.async_save:
            keys = list(self.save_info_list.keys())
            for record_step in keys:
                self.print_savetime(record_step, cb_params.batch_num)
                if not any([self.save_info_list[record_step][key]['ckpt_file_path']
                            for key in ['ckpt', 'network', 'trainable_params']]):
                    self.save_info_list.pop(record_step)

        if self._config.async_save and not ms.async_ckpt_thread_status() and \
                self.last_epoch_num and self.last_step_num_in_epoch and self.last_ckpoint_file and \
                not self.meta_updated:
            self.record_last_ckpt_to_json(self.last_epoch_num, self.last_step_num_in_epoch, self.last_ckpoint_file)
            self.meta_updated = True

        if save_ckpt:
            # NOTE: origin checkpoint processes are remained here
            self.save_checkpoint(cb_params)
            self.save_checkpoint_network(cb_params)

            # If async_save is False, output the time cost directly
            if not self._config.async_save:
                self.print_savetime(cb_params.cur_step_num, cb_params.batch_num)

    def get_checkpoint_health_info(self, cb_params):
        """get the health of checkpoint."""
        embedding_local_norm = get_embedding_info(cb_params, self.embedding_size)

        stage_nums = auto_parallel_context().get_pipeline_stages()
        device_nums = get_group_size()
        per_stage_device_nums = device_nums // stage_nums
        health_flag = ms.Tensor([0], dtype=ms.float32)
        is_health = 0
        if stage_nums > 1:
            parallel_mode = ms.get_auto_parallel_context("parallel_mode")
            ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE)
            if get_rank() < per_stage_device_nums:
                rank_list = list(range(0, per_stage_device_nums))
                if embedding_local_norm >= self.embedding_local_norm_threshold:
                    health_flag = ms.Tensor([1], dtype=ms.float32)
                group_name = self.create_group_pipeline(rank_list)
                final_health = AllReduceNet(group_name)(health_flag)
                if final_health.asnumpy() != 0:
                    is_health = 1
            ms.set_auto_parallel_context(parallel_mode=parallel_mode)
        return is_health

    def create_group_pipeline(self, rank_list):
        rank_str_list = [str(r) for r in rank_list]
        rank_list_str = "-".join(rank_str_list)
        # To make the name of group unique.
        hashed = hashlib.sha256(
            rank_list_str.encode()).hexdigest()[:48]
        pipeline_group_name = str(hashed)
        create_group(pipeline_group_name, rank_list)
        return pipeline_group_name

    def save_checkpoint(self, cb_params):
        """save checkpoint suitable for resume training."""
        logger.info('......Saving ckpt......')
        self.save_info_list[cb_params.cur_step_num]['ckpt']['save_start_time'] = time.time()
        step_num_in_epoch = int((cb_params.cur_step_num - 1) % cb_params.batch_num + 1)
        cur_ckpoint_file = (f"{self._prefix}-{str(cb_params.cur_epoch_num)}"
                            f"_{str(step_num_in_epoch)}.{self._config.format}")
        # update checkpoint file list.
        self._manager.update_ckpoint_filelist(self._directory, self._prefix)
        # keep checkpoint files number equal max number.
        if self._config.keep_checkpoint_max and 0 < self._config.keep_checkpoint_max <= self._manager.ckpoint_num:
            self._manager.remove_oldest_ckpoint_file()
            self.need_remove_extra_ckpt = True
        elif self._config.keep_checkpoint_per_n_minutes and self._config.keep_checkpoint_per_n_minutes > 0:
            # pylint: disable=E0203
            self._cur_time_for_keep = time.time()
            if (self._cur_time_for_keep - self._last_time_for_keep) \
                    < self._config.keep_checkpoint_per_n_minutes * 60:
                self._manager.keep_one_ckpoint_per_minutes(self._config.keep_checkpoint_per_n_minutes,
                                                           self._cur_time_for_keep)

        # generate the new checkpoint file and rename it.
        global SAVE_DIR
        SAVE_DIR = self._directory
        cur_file = os.path.join(self._directory, cur_ckpoint_file)
        self._last_time_for_keep = time.time()
        self._last_triggered_step = cb_params.cur_step_num

        if self.use_checkpoint_health_monitor:
            is_health = self.get_checkpoint_health_info(cb_params)
            # check the health of checkpoint and save the record file
            if get_rank() == 0:
                dump_health_json_path = os.path.join(self.health_ckpts_record_dir, "health_ckpts.json")
                health_step_data = {
                    'is_health': is_health,
                    'ckpt_name': cur_ckpoint_file
                }
                all_step_health_data = []
                if os.path.exists(dump_health_json_path):
                    with open(dump_health_json_path, 'r') as file:
                        data = json.load(file)
                        all_step_health_data = list(data)
                all_step_health_data.append(health_step_data)
                with open(dump_health_json_path, 'w') as file:
                    json.dump(all_step_health_data, file, indent=4)
                set_safe_mode_for_file_or_dir(dump_health_json_path)

        if "epoch_num" in self._append_dict:
            self._append_dict["epoch_num"] = cb_params.cur_epoch_num
        if "step_num" in self._append_dict:
            self._append_dict["step_num"] = self._append_step_num + cb_params.cur_step_num
        if cb_params.optimizer is not None:
            self._append_dict["global_step"] = cb_params.optimizer.global_step
        else:
            self._append_dict["global_step"] = cb_params.network.optimizer.global_step
        if "loss_scale" in self._append_dict:
            outputs = cb_params.net_outputs
            if isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
                self._append_dict["loss_scale"] = outputs[2]
        if self.global_batch_size is not None:
            self._append_dict["global_batch_size"] = self.global_batch_size
            logger.info("global_batch_size: %d", self._append_dict["global_batch_size"])
        logger.info("epoch_num: %d", self._append_dict["epoch_num"])
        logger.info("step_num: %d", self._append_dict["step_num"])
        logger.info("global_step: %d", self._append_dict["global_step"])
        network = self._config.saved_network if self._config.saved_network is not None else cb_params.train_network

        self.remove_redundancy(network, cur_file, self._append_dict, None)

        self._latest_ckpt_file_name = cur_file
        self.save_info_list[cb_params.cur_step_num]['ckpt']['ckpt_file_path'] = cur_file

        if self._config.async_save:
            self.last_epoch_num = cb_params.cur_epoch_num
            self.last_step_num_in_epoch = step_num_in_epoch
            self.last_ckpoint_file = cur_ckpoint_file
            self.meta_updated = False
        else:
            if "__exception_save__" not in self._append_dict:
                self.record_last_ckpt_to_json(cb_params.cur_epoch_num, step_num_in_epoch, cur_ckpoint_file)

    def _get_cur_dp(self, cur_rank, parameter_redundancy_dict):
        """get the current dp"""
        value_len = sys.maxsize
        min_value = ()
        min_value_set = set()
        for key, value in parameter_redundancy_dict.items():
            if key.startswith("accu_grads") or key.startswith("inputs"):
                continue
            for item in value:
                if cur_rank not in item:
                    continue
                # if item is subset of min_value_set, update min_value_set and min_value
                if len(item) < value_len:
                    if min_value_set and not set(item).issubset(min_value_set):
                        return (cur_rank,)
                    value_len = len(item)
                    min_value_set = set(item)
                    min_value = item
                # if value is not smaller than len of min_value len,
                # check if min_value_set is subset of current item
                elif not min_value_set.issubset(set(item)):
                    return (cur_rank,)

        return min_value

    def _filter_ckpt_not_save(self, x, filter_list):
        return all(not x.startswith(item) and item not in x for item in filter_list)

    def _tft_save_ckpt(self, param_layout_set, save_param_names, cur_file, append_dict, network):
        """save checkpoint with remove redundancy for TFT training."""
        def choice_func(x):
            return (x not in param_layout_set or (save_param_names is not None and x in save_param_names)) \
                and self._filter_ckpt_not_save(x, self.filter_list)

        ms.save_checkpoint(network, cur_file, False, False,
                           append_dict, self._config.enc_key, self._config.enc_mode,
                           format=self._config.format, choice_func=choice_func,
                           remove_redundancy=self._config.remove_redundancy)

    # pylint: disable=W0640
    def _do_remove_redundancy_for_tft(self, redundancy_info, cur_file, network, append_dict):
        """save checkpoint with remove redundancy for TFT training."""
        rank_id, param_redundancy_dict, single_params, param_layout = redundancy_info

        pattern = rf'_(\d+)\.{self._config.format}$'
        match = re.search(pattern, cur_file)
        cur_step_in_epoch = int(match.group(1))

        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        cur_dp = self._get_cur_dp(rank_id, param_redundancy_dict)
        # loop through all ranks in the current dp
        for rank in cur_dp:
            save_param_names = single_params.get(rank)
            if save_param_names == param_layout.keys():
                logger.warning(
                    f"For remove_redundancy save checkpoint, the saved parameters are non-redundant.")
            param_layout_set = set(param_layout.keys()) if parallel_mode else set()
            cur_file = re.sub(r'rank_\d+', f'rank_{rank}', cur_file)
            self._tft_save_ckpt(param_layout_set, save_param_names, cur_file, append_dict, network)
            append_dict["__exception_save__"] = True
            self.meta_json = re.sub(r'rank_\d+', f'rank_{rank}', self.meta_json)
            self.record_last_ckpt_to_json(append_dict["epoch_num"], cur_step_in_epoch, os.path.basename(cur_file))

    def _check_if_skip_trainable_params(self, value):
        """
        Checks if a trainable parameter should be skipped based on execution mode and parameter properties.
        """
        is_graph_mode = context.get_context('mode') == context.GRAPH_MODE
        in_auto_parallel = ms.get_auto_parallel_context("parallel_mode") in [
            ms.ParallelMode.SEMI_AUTO_PARALLEL,
            ms.ParallelMode.AUTO_PARALLEL,
        ]
        skip_for_parallel = is_graph_mode and in_auto_parallel and ((not value.sliced) or value.has_init)

        cur_param_info = value.param_info
        is_pipeline_shared = getattr(cur_param_info, 'is_pipeline_shared_param', False)

        return skip_for_parallel or is_pipeline_shared

    def remove_redundancy(self, network, cur_file, append_dict, train_network):
        """remove redundancy when saving checkpoint files."""
        if self._config.remove_redundancy:
            logger.info('......Removing redundancy......')
            parallel_mode = context.get_auto_parallel_context("parallel_mode")
            if parallel_mode == "stand_alone":
                raise TypeError(f"The deduplication feature for saving checkpoint can only be used "
                                f"in parallel scenarios, but got {parallel_mode}.")

            if train_network:
                param_layout = train_network.parameter_layout_dict
            else:
                param_layout = network.parameter_layout_dict
            rank_id = get_real_rank()
            if param_layout:
                device_num = get_real_group_size()
                stage_num = get_auto_parallel_context("pipeline_stages")
                chunk_size = device_num // stage_num
                initial_rank = (rank_id // chunk_size) * chunk_size
                param_redundancy_dict = get_parameter_redundancy(param_layout, initial_rank)
                single_params = remove_param_redundancy(param_redundancy_dict)
                save_param_names = single_params.get(rank_id)
                param_layout_set = set(param_layout.keys())
                if save_param_names == param_layout.keys():
                    logger.warning(
                        f"For remove_redundancy save checkpoint, the saved parameters are non-redundant.")

                def choice_func(x):
                    return (x not in param_layout_set or (save_param_names is not None and x in save_param_names)) \
                        and self._filter_ckpt_not_save(x, self.filter_list)
            else:
                param_redundancy_dict = get_parameter_redundancy(network)
                single_params = remove_param_redundancy(param_redundancy_dict)
                save_param_names = single_params.get(rank_id)

                def choice_func(x):
                    return save_param_names is not None and x in save_param_names \
                        and self._filter_ckpt_not_save(x, self.filter_list)

            # __exception_save__ is used to indicate that the checkpoint is saved by the TFT process
            if "__exception_save__" in append_dict:
                redundancy_info = (rank_id, param_redundancy_dict, single_params, param_layout)
                self._do_remove_redundancy_for_tft(redundancy_info, cur_file, network, append_dict)
                return

            ms.save_checkpoint(network, cur_file, False, self._config.async_save,
                               append_dict, self._config.enc_key, self._config.enc_mode,
                               format=self._config.format, choice_func=choice_func,
                               remove_redundancy=self._config.remove_redundancy)
        else:
            ms.save_checkpoint(network, cur_file, self._config.integrated_save, self._config.async_save,
                               append_dict, self._config.enc_key, self._config.enc_mode,
                               format=self._config.format,
                               choice_func=lambda x: self._filter_ckpt_not_save(x, self.filter_list),
                               remove_redundancy=self._config.remove_redundancy)

    def save_checkpoint_network(self, cb_params):
        """save checkpoint only network params, which is suitable for train, evaluate and predict."""
        save_obj = cb_params.network
        network = self._config.saved_network if self._config.saved_network is not None else cb_params.train_network

        if hasattr(save_obj, 'optimizer') and save_obj.optimizer is not None:
            save_obj = save_obj.network
        step_num_in_epoch = int((cb_params.cur_step_num - 1) % cb_params.batch_num + 1)

        if self.save_trainable_params:
            self.save_info_list[cb_params.cur_step_num]['trainable_params']['save_start_time'] = time.time()
            save_obj.init_parameters_data()
            param_dict = OrderedDict()
            for param in save_obj.trainable_params():
                param_dict[param.name] = param
            param_list = []
            for (key, value) in param_dict.items():
                if self._check_if_skip_trainable_params(value):
                    continue

                each_param = {"name": key}
                param_data = Tensor(value.data.asnumpy())

                # in automatic model parallel scenario, some parameters were split to all the devices,
                # which should be combined before saving
                if key in save_obj.parameter_layout_dict:
                    param_data = _get_merged_param_data(save_obj, key, param_data, self._config.integrated_save)

                each_param["data"] = param_data
                param_list.append(each_param)
            save_obj = param_list
            cb_cur_ckpoint_file = (f"{self._prefix}-trainable_params-{str(cb_params.cur_epoch_num)}"
                                   f"_{str(step_num_in_epoch)}.{self._config.format}")
            cb_cur_file = os.path.join(self.trainable_directory, cb_cur_ckpoint_file)
            os.makedirs(self.trainable_directory, exist_ok=True)

            # update checkpoint file list.
            self._trainable_manager.update_ckpoint_filelist(
                self.trainable_directory, f"{self._prefix}-trainable_params"
            )
            # keep checkpoint files number equal max number.
            if self.need_remove_extra_ckpt:
                self._trainable_manager.remove_oldest_ckpoint_file()

            self.remove_redundancy(save_obj, cb_cur_file, {}, network)
            self.save_info_list[cb_params.cur_step_num]['trainable_params']['ckpt_file_path'] = cb_cur_file
            return

        if self.save_network_params:
            self.save_info_list[cb_params.cur_step_num]['network']['save_start_time'] = time.time()
            cb_cur_ckpoint_file = (f"{self._prefix}-network-{str(cb_params.cur_epoch_num)}"
                                   f"_{str(step_num_in_epoch)}.{self._config.format}")
            cb_cur_file = os.path.join(self.network_directory, cb_cur_ckpoint_file)
            os.makedirs(self.network_directory, exist_ok=True)

            # update checkpoint file list.
            self._network_manager.update_ckpoint_filelist(self.network_directory, f"{self._prefix}-network")
            # keep checkpoint files number equal max number.
            if self.need_remove_extra_ckpt:
                self._network_manager.remove_oldest_ckpoint_file()

            self.remove_redundancy(save_obj, cb_cur_file, {}, network)
            self.save_info_list[cb_params.cur_step_num]['network']['ckpt_file_path'] = cb_cur_file

        self.need_remove_extra_ckpt = False

    def _save_megatron_ckpt_file_format(self, cb_params):
        """Save the checkpoints like megatron format."""
        # Get current step as iteration
        iteration = self._append_step_num + cb_params.cur_step_num

        # Get common info
        self.common_info.step_num = iteration
        self.common_info.epoch_num = cb_params.cur_epoch_num
        self.common_info.global_step = int(cb_params.network.optimizer.global_step)
        self.common_info.loss_scale = None
        if isinstance(cb_params.net_outputs, (tuple, list)) and len(cb_params.net_outputs) >= 3:
            self.common_info.loss_scale = float(cb_params.net_outputs[2])
        self.common_info.global_batch_size = self.global_batch_size

        from mindspore.parallel.strategy import get_strategy_metadata
        # Get all strategy info of this network to save 'metadata.json'
        global_strategy_info = get_strategy_metadata(network=cb_params.network)

        save_checkpoint(
            iteration=iteration,
            network=cb_params.network.network,
            optimizer=cb_params.network.optimizer if self.save_optimizer else None,
            async_save_manager=self.async_save_manager if self._config.async_save else None,
            common_info=self.common_info,
            keep_max_num=self._config.keep_checkpoint_max,
            user_prefix=self.origin_prefix,
            save_checkpoint_path=self.save_checkpoint_path,
            global_strategy_info=global_strategy_info,
            remove_redundancy=self.need_remove_redundancy
        )

    def record_last_ckpt_to_json(self, epoch, step, ckpt_file):
        """record last ckpt info to json"""
        meta_data = {
            "last_epoch": epoch,
            "last_step": step,
            "last_ckpt_file": ckpt_file
        }
        with tempfile.NamedTemporaryFile('w', delete=False, dir=self._directory) as temp_file:
            json.dump(meta_data, temp_file)
            temp_file_path = temp_file.name
        os.replace(temp_file_path, self.meta_json)
        set_safe_mode_for_file_or_dir(self.meta_json)

    def step_end(self, run_context):
        """
        Save the checkpoint at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        if self.use_legacy_format:
            super().step_end(run_context)
        else:
            cb_params = run_context.original_args()
            force_to_save = False
            if cb_params.cur_step_num == self._last_triggered_step:
                return

            # If param is cache enable, flush data from cache to host before save_ckpt
            if self._need_flush_from_cache:
                self._flush_from_cache(cb_params)

            cur_step_need_save_ckpt = self._check_save_ckpt(cb_params, force_to_save)
            # Save checkpoint
            if cur_step_need_save_ckpt:
                self._save_megatron_ckpt_file_format(cb_params)

                # If async_save is False, output the time cost directly
                if not self._config.async_save:
                    self.print_savetime(cb_params.cur_step_num, cb_params.batch_num)
                self._last_triggered_step = cb_params.cur_step_num

    def end(self, run_context):
        """
        Save the last checkpoint after training finished.

        Args:
            run_context (RunContext): Context of the train running.
        """
        if self.use_legacy_format:
            super().end(run_context)

    def on_train_step_begin(self, run_context):
        """Called before each training step."""
        super().on_train_step_begin(run_context)
        if not self.use_legacy_format and self._config.async_save:
            logger.info("(on_train_step_begin) Try to execute finalize func.")
            self.async_save_manager.maybe_finalize(wait_finish=False)

    def on_train_end(self, run_context):
        """Called after the end of training."""
        super().on_train_end(run_context)
        if not self.use_legacy_format:
            cb_params = run_context.original_args()
            # Need to save the last step checkpoint
            self._save_megatron_ckpt_file_format(cb_params)

            if self._config.async_save:
                logger.info("(on_train_end) Wait all ranks and execute finalize func.")
                self.async_save_manager.maybe_finalize(wait_finish=True)


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class ProfileMonitor(Callback):
    """
    Profile analysis in training.

    Args:
        start_step (int, optional): The step to start profiling. Default: ``1``.
        stop_step (int, optional): The step to stop profiling. Default: ``10``.
        output_path (str, optional): The result of profiling will be saved in this path. Default: ``None``.
        start_profile (str, optional): Whether to enable profiling. Default: ``True``.
        profile_rank_ids (list, optional): Specify rank ids to enable profiling. Default: ``None`` (All rank ids
            are enabled).
        profile_pipeline (str, optional): Whether to enable profiling on one card of each parallel stage.
            Default: ``False``.
        profile_communication (str, optional): Whether to collect communication performance data
            during multi-device training. Default: ``False``.
        profile_memory (str, optional): Whether to collect Tensor memory data. Default: ``False``.
        config (dict, optional): Configuration items, used to profile relevant configuration information,
            such as parallel configuration. Default: ``None``.
        profiler_level (int, optional): Collection level of profiling data(0, 1, 2). Default: ``0``.

            - 0: The most streamlined level of performance data collection,
              only collecting execution time data for computational operators and
              basic data for large communication operators.
            - 1: In addition to level 0, extra data is collected for CANN layer AscendCL,
              AICORE performance data, and small communication operators.
            - 2: In addition to level 1, extra data is collected for graph compile level O2
              and Runtime in the CANN layer.

        with_stack (str, optional): Whether to collect Python-side stack trace data. Default: ``False``.
        data_simplification (str, optional): Whether to enable data simplification, which will delete the FRAMEWORK
            directory and other extraneous data after exporting profiling data. Default: ``True``.
        mstx (bool, optional): Whether to enable mstx step-time recording. Default: ``False``.

    Examples:
        >>> from mindformers.core import ProfileMonitor
        >>> monitor = ProfileMonitor(output_path='./profile_dir')
    """

    def __init__(self, start_step=1, stop_step=10, output_path=None,
                 start_profile=True, profile_rank_ids=None, profile_pipeline=False,
                 profile_communication=False, profile_memory=False, config=None,
                 profiler_level=0, with_stack=False, data_simplification=True, mstx=False, **kwargs):
        super(ProfileMonitor, self).__init__()
        self.mstx_range_id = None
        self.mstx_enabled = is_version_ge(ms.__version__, '2.5.0') and not _check_mspti_is_on()
        self.stop_step = stop_step
        self.profile_rank_ids = profile_rank_ids
        self.profile_pipeline = profile_pipeline
        self.profiler = None

        # check start_profile
        start_profile = self._check_start_profile(start_profile, start_step)

        # check step
        self.start_step, stop_step = self._check_step(start_step, stop_step)

        if profile_communication:
            if profiler_level == 0:
                profiler_level = 1
                logger.warning(
                    "When profile_communication is True, profiler_level must be greater than 0, reset "
                    "profiler_level to 1")
        # convert profiler_level
        profiler_level = self._get_profiler_level(profiler_level)

        rank_id = get_real_rank()
        self.pipeline_rank_ids = get_pipeline_rank_ids() if self.profile_pipeline else None
        if self.pipeline_rank_ids == [-1]:
            raise ValueError(f"Device num should be divided by pipeline stage num.")

        if self._is_profile_required(rank_id):
            if not output_path:
                output_path = get_output_subpath('profile', rank_id)
            else:
                output_path = os.path.join(output_path, 'profile', 'rank_{}'.format(rank_id))
            logger.info("Profile save path: %s", output_path)

            if ms.get_context("device_target") == "GPU" and profile_memory:
                logger.warning("The parameter profile_memory is not supported on GPU currently, "
                               "so is changed to False. ")
                profile_memory = False

            # get schedule config
            schedule_config = self._get_schedule(start_profile, start_step, stop_step)
            if is_version_ge(ms.__version__, '2.6.0'):
                from mindspore.profiler import profile, _ExperimentalConfig, tensorboard_trace_handler

                experimental_config = _ExperimentalConfig(profiler_level=profiler_level,
                                                          data_simplification=data_simplification,
                                                          mstx=mstx)
                self.profiler = profile(
                    profile_memory=profile_memory,
                    start_profile=False,
                    with_stack=with_stack,
                    schedule=schedule_config,
                    on_trace_ready=tensorboard_trace_handler(dir_name=output_path),
                    experimental_config=experimental_config,
                    **kwargs
                )
                self.is_profiler_start = False
            # compatible to old version mindspore
            else:
                from mindspore.profiler import Profiler, tensor_board_trace_handler
                self.profiler = Profiler(
                    start_profile=False,
                    profile_memory=profile_memory,
                    profiler_level=profiler_level,
                    with_stack=with_stack,
                    data_simplification=data_simplification,
                    mstx=mstx,
                    schedule=schedule_config,
                    on_trace_ready=tensor_board_trace_handler(dir_name=output_path),
                    **kwargs
                )
                self.is_profiler_start = False
            self._record_metadata(config)
            self.run_context = None
            self.output_path = output_path

    @staticmethod
    def _check_step(start_step, stop_step):
        """
        Check start_step and stop_step.

        Args:
            start_step: start step number.
            stop_step: stop step number.
        """
        if start_step < 0:
            start_step = 1
            logger.warning("start_step must bo greater than 0, but got %s, reset to default 1")
        if stop_step < 0:
            stop_step = 10
            logger.warning("stop_step must bo greater than 0, but got %s, reset to default 10")
        if start_step > stop_step:
            start_step = 1
            stop_step = 10
            logger.warning("stop_step must bo greater than start_step, but get start_step = %d, stop_step = %d, "
                           "now start_step and stop_step are reset to 1 and 10.", start_step, stop_step)
        return start_step, stop_step

    @staticmethod
    def _check_start_profile(start_profile, start_step):
        """
        Check start_step and stop_step.

        Args:
            start_profile: Whether to collect after initialization.
            start_step: start step number.
        """
        if start_step != 1 and start_profile:
            logger.warning("If the parameters start_step and init_start_profile are set simultaneously, "
                           "the init_start_profile parameter will not take effect, reset init_start_profile to False.")
            return False
        return start_profile

    @staticmethod
    def _get_schedule(start_profile, start_step, stop_step):
        """
        Get schedule by start_step and stop_step.

        Args:
            start_profile: Whether to start the profiler from the first step.
            start_step: start step number.
            stop_step: stop step number.
        """
        if start_profile:
            schedule_config = schedule(wait=0, active=stop_step, warmup=0, repeat=1, skip_first=1)
        else:
            schedule_config = schedule(wait=0, active=stop_step - start_step + 1, warmup=0, repeat=1,
                                       skip_first=start_step)
        return schedule_config

    def on_train_step_begin(self, run_context):
        """
        Start profile at the beginning of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if self.profiler and not self.is_profiler_start:
            self.profiler.start()
            self.is_profiler_start = True

        if self.mstx_enabled:
            self.mstx_range_id = ms.profiler.mstx.range_start(f'step {step_num}', ms.runtime.current_stream())

    def on_train_step_end(self, run_context):
        """
        Stop profile at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if self.mstx_enabled:
            ms.profiler.mstx.range_end(self.mstx_range_id)
        if self.profiler:
            self.profiler.step()
        if step_num == self.stop_step and self.profiler:
            logger.info("End of Profiling, please analyze it using MindStudio Insight. "
                        "See https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/"
                        "GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html for details.")

    def _record_metadata(self, config):
        """
        Record metadata from config.

        Args:
            config (dict): config of the train running.
        """
        if config is None:
            return

        parallel = config.parallel
        parallel_config = config.parallel_config.to_dict()

        try:
            self.profiler.add_metadata_json('distributed_args', json.dumps({
                'tensor_model_parallel_size': parallel_config.get('model_parallel', 1),
                'pipeline_model_parallel_size': parallel_config.get('pipeline_stage', 1),
                'data_parallel_size': parallel_config.get('data_parallel', 1),
                'expert_model_parallel_size': parallel_config.get('expert_parallel', 1),
                'sequence_parallel': parallel_config.get('use_seq_parallel', False),
                'parallel_mode': parallel.get('parallel_mode', None),
                'world_size': parallel.get('device_num', None)
            }))
        except AttributeError as e:
            logger.warning("Profiler failed to record distributed args,  %s", e)

    def _is_profile_required(self, rank_id):
        """
        Determine whether current rank id needs to enable profiling.

        Args:
            rank_id (int): current rank id.
        """
        if not self.profile_rank_ids and not self.pipeline_rank_ids:
            return True

        profile_ids = self.profile_rank_ids if isinstance(self.profile_rank_ids, list) else []
        pipeline_ids = self.pipeline_rank_ids if isinstance(self.pipeline_rank_ids, list) else []

        if rank_id in profile_ids or rank_id in pipeline_ids:
            return True

        return False

    @staticmethod
    def _get_profiler_level(level):
        """
        Obtain profiler level based on the level value with integer type.

        Args:
            level (int): the value of profiler_level in MF config.
        """
        if level is None:
            return ProfilerLevel.Level0

        max_level = len(ProfilerLevel.__members__) - 1
        if level < 0 or level > max_level:
            logger.warning("Invalid profiler_level: %s, return None.", level)
            return None
        profiler_level = getattr(ProfilerLevel, f"Level{level}")
        return profiler_level


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class EvalCallBack(Callback):
    """
    Evaluate Callback used in training progress.

    Args:
        eval_func (Callable): The function used to evaluate the model results
            and can be customized according to specific task.
        step_interval (int, optional): Determine the num of step intervals between each eval.
            Default ``100``. Note that it will not take effects when running in data sink mode.
        epoch_interval (int, optional): Determine the num of epoch intervals between each eval.
            Default ``-1``, means eval on every epoch end.

    Examples:
        >>> from mindformers.core.callback import EvalCallBack
        >>> def eval_func():
        ...     print("output result")
        >>> eval_callback = EvalCallBack(eval_func=eval_func)
        >>> type(eval_callback)
    """

    def __init__(self, eval_func: Callable, step_interval: int = 100, epoch_interval: int = -1):
        self.eval_func = eval_func
        self.step_interval = step_interval
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, run_context):
        # if not use epoch end
        if self.epoch_interval <= 0:
            return
        callback_params = run_context.original_args()
        cur_epoch_num = callback_params.cur_epoch_num
        if cur_epoch_num % self.epoch_interval == 0:
            self._execute_eval()

    def on_train_step_end(self, run_context):
        # if not use step end
        if self.step_interval <= 0:
            return
        callback_params = run_context.original_args()
        cur_step_num = callback_params.cur_step_num
        if cur_step_num % self.step_interval == 0:
            self._execute_eval()

    def _execute_eval(self):
        start_time = time.time()
        output = self.eval_func()
        eval_time = time.time() - start_time
        logger.info("Eval result: %s, eval time is %f s.", output, eval_time)


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class ColdHotExpertMonitor(Callback):
    """
        ColdHotExpertMonitor Callback used in MoE model training progress.

        Args:
            config : Read config from configuration file.

        Examples:
            >>> from mindformers.core.callback import ColdHotExpertMonitor
            >>> callback = ColdHotExpertMonitor(config)
            >>> type(callback)
            <class 'mindformers.core.callback.callback.ColdHotExpertMonitor'>
    """

    def __init__(self, moe_config=None, hidden_size=None, ffn_hidden_size=None, expert_parallel=None,
                 model_parallel=None, save_checkpoint_steps=None):
        self.update_step = moe_config.update_step if hasattr(moe_config, "update_step") else 10000
        self.expert_num = moe_config.expert_num
        self.hot_expert_num = moe_config.hot_expert_num
        self.moe_module_name = moe_config.moe_module_name
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.ep = expert_parallel
        self.mp = model_parallel
        self.save_checkpoint_steps = save_checkpoint_steps
        self.rank_id = int(os.getenv("RANK_ID"))
        self.local_expert_num = self.expert_num // self.ep
        start_index = (self.rank_id // self.mp) * self.local_expert_num
        end_index = start_index + self.local_expert_num
        self.local_expert_index = [i for i in range(start_index, end_index)]
        self.rank_size = int(os.getenv("RANK_SIZE"))

    def on_train_step_end(self, run_context):
        """
        Switch popular expert copies when there is a change in popular experts at the step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        if self.update_step <= 0:
            return
        callback_params = run_context.original_args()
        cur_step_num = callback_params.cur_step_num
        if ((cur_step_num < self.update_step and cur_step_num & (cur_step_num - 1) == 0) or
                (cur_step_num == self.save_checkpoint_steps) or (cur_step_num % self.update_step == 0)):
            total_start = time.time()
            train_network = callback_params.train_network
            if train_network is None:
                return
            blocks = self.get_attribute_by_path(train_network, self.moe_module_name)
            for block in blocks:
                if cur_step_num > 1:
                    self.return_back_hot_expert(block)
                self.switch_hot_expert(block, cur_step_num)
            total_end = time.time()
            logger.info("switch hot experts spent time is %f s.", total_end - total_start)

    def on_train_end(self, run_context):
        """
        Switch popular expert copies when there is a change in popular experts at the step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        callback_params = run_context.original_args()
        cur_step_num = callback_params.cur_step_num
        train_network = callback_params.train_network
        if train_network is None:
            return
        blocks = self.get_attribute_by_path(train_network, self.moe_module_name)
        for block in blocks:
            if cur_step_num > 1:
                self.return_back_hot_expert(block)

    def get_attribute_by_path(self, obj, path):
        """
        Obtains MoE blocks modules in obj by path.

        Args:
            obj : Model.
            path(str) : Path of the MoE layer in the model
        """
        for attr in path.split('.'):
            obj = getattr(obj, attr)
        return obj

    def return_back_hot_expert(self, block):
        """
        When the popular experts change, return the replica parameters to the old popular experts.

        Args:
            block : MoE layer.
        """
        old_hot_expert_index = block.output.hot_expert_index.value()[0]
        if self.hot_expert_num == 1:
            if old_hot_expert_index[0] in self.local_expert_index:
                ffn_index = old_hot_expert_index[0] - (self.rank_id // self.mp) * self.local_expert_num
                block.output.ffn.mapping.weight[ffn_index] = block.output.mlp.mapping.weight
                block.output.ffn.mapping.bias[0][ffn_index][0] = block.output.mlp.mapping.bias
                block.output.ffn.projection.weight[ffn_index] = block.output.mlp.projection.weight
                block.output.ffn.projection.bias[0][ffn_index][0] = block.output.mlp.projection.bias
        elif self.hot_expert_num > 1:
            for i in range(self.hot_expert_num):
                if old_hot_expert_index[i] in self.local_expert_index:
                    ffn_index = old_hot_expert_index[i] - (self.rank_id // self.mp) * self.local_expert_num
                    block.output.ffn.mapping.weight[ffn_index] = block.output.mlp.mapping.weight[i]
                    block.output.ffn.mapping.bias[0][ffn_index][0] = block.output.mlp.mapping.bias[0][i][0]
                    block.output.ffn.projection.weight[ffn_index] = block.output.mlp.projection.weight[i]
                    block.output.ffn.projection.bias[0][ffn_index][0] = block.output.mlp.projection.bias[0][i][0]

    def switch_hot_expert(self, block, cur_step_num):
        """
        Switch popular expert copies when there is a change in popular experts at the step.

        Args:
            block : MoE layer.
            cur_step_num : Current training step
        """
        old_hot_expert_index = block.output.hot_expert_index.value()[0]
        cumsum_tensor = block.output.router.router.cumsum_value.value()
        _, new_expert_index = cumsum_tensor.topk(self.expert_num, largest=True)
        new_hot_expert_index = new_expert_index[0:self.hot_expert_num]
        new_cold_expert_index = new_expert_index[self.hot_expert_num:self.expert_num]
        broadcasts = [self.BroadcastCell(i) for i in range(self.rank_size)]
        if self.hot_expert_num == 1:
            if cur_step_num > 1 and old_hot_expert_index[0] == new_hot_expert_index[0]:
                return
            # Broadcast new hot expert and copy the weights of new hot experts to mlp
            for i in range(self.mp):
                ffn_index = new_hot_expert_index[0] % self.local_expert_num
                rank_id = new_hot_expert_index[0] // self.local_expert_num * self.mp + i
                expert_part = broadcasts[rank_id]((block.output.ffn.mapping.weight[ffn_index],
                                                   block.output.ffn.mapping.bias[0][ffn_index][0],
                                                   block.output.ffn.projection.weight[ffn_index],
                                                   block.output.ffn.projection.bias[0][ffn_index][0]))
                if self.rank_id % self.mp == i:
                    block.output.mlp.mapping.weight = expert_part[0]
                    block.output.mlp.mapping.bias = expert_part[1]
                    block.output.mlp.projection.weight = expert_part[2]
                    block.output.mlp.projection.bias = expert_part[3]
        elif self.hot_expert_num > 1:
            new_hot_expert_index, _ = new_hot_expert_index.topk(self.hot_expert_num, largest=False)
            if cur_step_num > 1 and old_hot_expert_index.equal(new_hot_expert_index).all():
                return
            # Broadcast new hot expert and copy the weights of new hot experts to mlp
            for index in range(self.hot_expert_num):
                for i in range(self.mp):
                    ffn_index = new_hot_expert_index[index] % self.local_expert_num
                    rank_id = new_hot_expert_index[index] // self.local_expert_num * self.mp + i
                    expert_part = broadcasts[rank_id]((block.output.ffn.mapping.weight[ffn_index],
                                                       block.output.ffn.mapping.bias[0][ffn_index][0],
                                                       block.output.ffn.projection.weight[ffn_index],
                                                       block.output.ffn.projection.bias[0][ffn_index][0]))
                    if self.rank_id % self.mp == i:
                        block.output.mlp.mapping.weight[index] = expert_part[0]
                        block.output.mlp.mapping.bias[0][index][0] = expert_part[1]
                        block.output.mlp.projection.weight[index] = expert_part[2]
                        block.output.mlp.projection.bias[0][index][0] = expert_part[3]
        block.output.hot_expert_index = new_hot_expert_index.reshape((1, -1))
        block.output.cold_expert_index = new_cold_expert_index.reshape((1, -1))
        del broadcasts

    class BroadcastCell(Cell):
        def __init__(self, rank_id):
            super().__init__(auto_prefix=False)
            self.broadcast = Broadcast(rank_id)
            self.add_flags(skip_auto_parallel_compile=True)

        @jit()
        def construct(self, x):
            x = self.broadcast(x)
            return x


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class TrainCallBack(Callback):
    """
    Train Callback used in training progress.

    Args:
        stop_step (int): The function stop train process at the step.
                             Default None, set in yaml.
    Examples:
        >>> from mindformers.core.callback import TrainCallBack
        >>> stop_step = TrainCallBack(stop_step=10)
        <class 'mindformers.core.callback.callback.TrainCallBack'>
    """

    def __init__(self, stop_step: int = None):
        self.stop_step = stop_step

    def on_train_step_end(self, run_context):
        """
        Print training info at the end of epoch.

        Args:
            run_context (RunContext): Context of the process running.
        """
        cb_params = run_context.original_args()
        if self.stop_step is not None and cb_params.cur_step_num >= self.stop_step:
            run_context.request_stop()
            logger.info("set train process early stop at %s steps in yaml", self.stop_step)


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class StressDetectCallBack(Callback):
    """
    Stress Detect Callback used in training progress.

    Args:
        detection_interval (int): (int, optional): The number of steps between each hardware precision stress detection.
            Default: ``None``.
        num_detections (int, optional): The number of consecutive hardware precision stress detections for each round.
            Default: ``None``.
        dataset_size (int, optional): Training dataset size. Default: ``None``.

    Examples:
        >>> from mindformers.core.callback import StressDetectCallBack
        >>> stress_detect_callback = StressDetectCallBack(detection_interval=10, num_detections=3, dataset_size=1024)
        >>> type(stress_detect_callback)
    """

    def __init__(self, detection_interval: int = None, num_detections: int = None, dataset_size: int = None):
        logger.warning('StressDetectCallBack serves as an experimental interface and its functionality is '
                       'not yet stable.')
        self.detection_interval = detection_interval
        self.num_detections = num_detections
        self.steps_per_epoch = dataset_size
        self.ms_version_valid = check_stress_detect_valid()

        if self.detection_interval > self.steps_per_epoch:
            logger.warning(f"detection_interval = {self.detection_interval} is bigger than "
                           f"steps_per_epoch = {self.steps_per_epoch}")


    def on_train_step_end(self, run_context):
        """
        Stress detect at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        callback_params = run_context.original_args()
        cur_step_num = callback_params.cur_step_num
        # stress detect
        detect_ret_list = []
        if self.ms_version_valid:
            from mindspore.utils import stress_detect

            if cur_step_num % self.detection_interval == 0:
                logger.info("Start to stress detect")
                for _ in range(self.num_detections):
                    ret = stress_detect()
                    detect_ret_list.append(ret)

            self.log_stress_detect_result(detect_ret_list)


    @staticmethod
    def log_stress_detect_result(detect_ret_list):
        """print output information."""
        for ret in detect_ret_list:
            if ret == 0:
                logger.info("Stress detection passed")
            elif ret == VOLTAGE_ERROR_CODE:
                raise RuntimeError(f"Voltage recovery failed with error code: {ret}")
            else:
                logger.warning(f"Stress detection failed with error code: {ret}")


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class TopkBiasBalanceCallback(Callback):
    """
    Callback for topk bias balance feature in moe module.
    Arguments below, except `gradient_accumulation_steps`, take effects only when use legacy models.

    Args:
        balance_via_topk_bias (bool, optional):
            Whether to use topk bias update, should be consistent with moe config. Defaults to False.
        topk_bias_update_rate (float, optional): How fast is the bias updated. Defaults to 0.0.
        expert_num (int, optional): How many experts in the moe module. Defaults to 1.
        micro_batch_num (int, optional): Micro batch number in pipeline parallel. Default to 1.
        gradient_accumulation_steps (int, optional): Gradient accumulation steps for training. Default to 1.
    """
    def __init__(self,
                 balance_via_topk_bias: bool = False,
                 topk_bias_update_rate: float = 0.0,
                 expert_num: int = 1,
                 micro_batch_num: int = 1,
                 gradient_accumulation_steps: int = 1):
        # for aux loss free
        # this process is to update the expert load
        self.update_topk_bias_flag = balance_via_topk_bias
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.write_expert_load_to_tensorboard = get_tensorboard_args()['log_expert_load_to_tensorboard']
        self.tensor_writer = get_tensorboard_writer()
        if self.update_topk_bias_flag and self.tensor_writer is not None and self.write_expert_load_to_tensorboard:
            logger.info('The expert loads will be written to tensorboard.')
        self.cur_step = 0
        if self.update_topk_bias_flag:
            self.assign = P.Assign()
            self.assign.recompute(False)
            self.afb_sub = P.Sub()
            self.afb_add = P.Add()
            self.sign = P.Sign()
            self.afb_mul = P.Mul()
            self.afb_div = P.Div()
            self.pipeline_stages = ms.context.get_auto_parallel_context("pipeline_stages")
            self.micro_batch_num = micro_batch_num if self.pipeline_stages > 1 else 1
            self.acc_step_over_expert_num = \
                Tensor([micro_batch_num * gradient_accumulation_steps / expert_num], ms.float32)
            self.topk_bias_update_rate = topk_bias_update_rate
            self.zeros_tensor = ms.Tensor(np.zeros([expert_num]), ms.float32)

    def _update_topk_bias(self, network):
        """update topk bias tensor during training."""
        while hasattr(network, "network"):
            network = network.network
        if hasattr(network, "update_topk_bias"):
            expert_loads = network.update_topk_bias(self.gradient_accumulation_steps)
            if self.tensor_writer is not None and self.write_expert_load_to_tensorboard:
                for layer, expert_load in expert_loads:
                    if expert_load.sum() > 0:
                        expert_load_dict = {f"ep_{i}": load_i.asnumpy() for i, load_i in enumerate(expert_load)}
                        self.tensor_writer.add_scalars(
                            f"expert_load/{layer}",
                            expert_load_dict,
                            global_step=self.cur_step
                        )
            return
        if self.update_topk_bias_flag:
            for layer in network.model.layers:
                if hasattr(layer.feed_forward, "routed_experts"):
                    if hasattr(layer.feed_forward.routed_experts, "router"):
                        expert_load_data = \
                            layer.feed_forward.routed_experts.router.router.expert_load.value()
                        if expert_load_data.sum() > 0:
                            err = self.afb_sub(self.acc_step_over_expert_num, expert_load_data)
                            topk_bias_new = self.afb_add(
                                layer.feed_forward.routed_experts.router.router.topk_bias.value(),
                                self.afb_mul(self.sign(err), self.topk_bias_update_rate)
                            )
                            self.assign(layer.feed_forward.routed_experts.router.router.topk_bias,
                                        topk_bias_new)
                            self.assign(layer.feed_forward.routed_experts.router.router.expert_load,
                                        self.zeros_tensor)
                    else:
                        expert_load_data = layer.feed_forward.routed_experts.expert_load.value()
                        if expert_load_data.sum() > 0:
                            err = self.afb_sub(self.acc_step_over_expert_num, expert_load_data)
                            topk_bias_new = self.afb_add(
                                layer.feed_forward.routed_experts.topk_bias.value(),
                                self.afb_mul(self.sign(err), self.topk_bias_update_rate)
                            )
                            self.assign(layer.feed_forward.routed_experts.topk_bias, topk_bias_new)
                            self.assign(layer.feed_forward.routed_experts.expert_load, self.zeros_tensor)

    def on_train_step_end(self, run_context):
        """update expert bias at the end of step."""
        cb_params = run_context.original_args()
        self.cur_step = cb_params.cur_step_num
        # pylint: disable=W0212
        network = cb_params.train_network
        while hasattr(network, 'network'):
            network = network.network
        parallel_mode = get_auto_parallel_context("parallel_mode")
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"] and ms.get_context('mode') == 0:
            network = network._backbone
        self._update_topk_bias(network)


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class MoEDropRateCallback(Callback):
    """Callback drop rate in moe module.

    Args:
        expert_num (int): How many experts in the moe module.
        capacity_factor (float): Capcity factor in the moe module.
        num_layers (int): How many layers in the model.
        mtp_depth (int): How many layers in the mtp module.

    Examples:
        >>> from mindformers.core.callback import MoEDropRateCallback
        >>> stop_step = MoEDropRateCallback(expert_num=8, capacity_factor=1.5, num_layers=4, mtp_depth=1)
        <class 'mindformers.core.callback.callback.MoEDropRateCallback'>
    """
    def __init__(self,
                 expert_num: int,
                 capacity_factor: float,
                 num_layers: int,
                 mtp_depth: int):
        self.capacity_factor_over_expert_num = capacity_factor / expert_num
        self.num_layers = num_layers + mtp_depth

    def _callback_droprate(self, network):
        """callback drop rate."""
        for i in range(self.num_layers):
            while hasattr(network, "network"):
                network = network.network
            if hasattr(network.model.layers[i].feed_forward, "routed_experts"):
                if hasattr(network.model.layers[i].feed_forward.routed_experts, "router"):
                    fi = network.model.layers[i].feed_forward.routed_experts.router.router.fi_parameter.value()
                    if fi.sum() > 0:
                        delta = fi - self.capacity_factor_over_expert_num
                        droprate = ms.ops.sum(delta * (delta > 0))
                        logger.info("layer: %d, drop_rate: %.5f" % (i, droprate))
            else:
                if hasattr(network.model.layers[i].feed_forward, "router"):
                    fi = network.model.layers[i].feed_forward.router.router.fi_parameter.value()
                    if fi.sum() > 0:
                        delta = fi - self.capacity_factor_over_expert_num
                        droprate = ms.ops.sum(delta * (delta > 0))
                        logger.info("layer: %d, drop_rate: %.5f" % (i, droprate))

    def on_train_step_end(self, run_context):
        """get expert drop rate at the end of step."""
        cb_params = run_context.original_args()
        # pylint: disable=W0212
        network = cb_params.train_network
        while hasattr(network, 'network'):
            network = network.network
        parallel_mode = get_auto_parallel_context("parallel_mode")
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"] and ms.get_context('mode') == 0:
            network = network._backbone
        self._callback_droprate(network)


def get_embedding_info(cb_params, embedding_size):
    """print embedding info and get the health of checkpoint."""
    if len(cb_params.net_outputs) < 7:
        raise ValueError("You should turn on the local norm while using the skip data by global norm function.")
    embedding_local_norm = 0
    pipeline_stages = ms.context.get_auto_parallel_context("pipeline_stages")
    device_nums = get_group_size()
    rank = get_rank()
    if rank < device_nums // pipeline_stages:
        for local_norm, local_norm_size in zip(cb_params.net_outputs[5], cb_params.net_outputs[6]):
            if local_norm_size == embedding_size:
                embedding_local_norm = local_norm
                break
    return embedding_local_norm


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class StressTestModelMonitor(Callback):
    """Initialize the StressTestModelMonitor.

    Args:
        interval_steps (int, optional): Number of steps after which to check the model.
        stress_model_dir (str, optional): The directory where the model ymal file is stored.
        stress_dataset_dir (str): The directory where the stress test dataset is stored.
        compare_interval_steps (int, optional): Number of interval steps where the stress test result is compared.
        stress_master_port (int, optional): The master port of stress test.
        stress_test_log_dir (str optional): The directory where the stress test training log is stored.
        check_stresslog_interval_time (int, optional): Time interval where the stress test log is checked.
    """
    def __init__(self,
                 interval_steps=10,
                 stress_model_dir=None,
                 stress_dataset_dir=None,
                 compare_interval_steps=None,
                 stress_master_port=8338,
                 stress_test_log_dir="test_output/stress_test_output1/msrun_log",
                 check_stresslog_interval_time=60):
        logger.warning('StressTestModelMonitor serves as an experimental interface and its functionality is '
                       'not yet stable.')
        super(StressTestModelMonitor, self).__init__()

        self.interval_steps = interval_steps
        self.last_checked_step = 0
        self.model_dir = stress_model_dir
        if not self.model_dir or not os.path.exists(self.model_dir):
            raise ValueError(f"model_dir {self.model_dir} was not found for StressTestModelMonitor.")
        self.dataset_dir = stress_dataset_dir
        self.stress_master_port = stress_master_port
        self.main_master_port = int(os.getenv("MS_SCHED_PORT"))
        logger.info(f"The main model is using master port {self.main_master_port}")
        if not isinstance(self.stress_master_port, int) or self.stress_master_port < 1:
            logger.warning(f"For StressTestMonitor, stress_master_port must be an integer greater than or equal "
                           f"to 1, but got {self.stress_master_port}. Setting to default value 8338")
            self.stress_master_port = 8338
        if self.stress_master_port == self.main_master_port:
            logger.warning(f"For StressTestMonitor, stress_master_port must be different from the main task "
                           f"but both got {self.stress_master_port}. Setting to {self.stress_master_port+1}")
            self.stress_master_port += 1
            logger.warning(f"Make sure that the new port {self.stress_master_port} is unoccupied.")
        self.worker_num = ms.communication.get_local_rank_size()
        logger.info(f"Local worker number for each stress test is {self.worker_num}.")
        self.compare_interval_steps = compare_interval_steps
        if not isinstance(self.compare_interval_steps, int) or self.compare_interval_steps < 1:
            logger.warning(f"For StressTestMonitor, compare_interval_steps must be an integer greater than or equal"
                           f" to 1, but got {self.compare_interval_steps}.")
            logger.warning(f"Skipping interval steps comparison, only the last step result will be compared."
                           f" compare_interval_steps is set to None")
            self.compare_interval_steps = None
        self.stress_test_log_dir = stress_test_log_dir
        self.check_stresslog_interval_time = check_stresslog_interval_time
        if not isinstance(self.check_stresslog_interval_time, int) or self.check_stresslog_interval_time < 1:
            logger.warning(f"For StressTestMonitor, check_stresslog_interval_time must be an integer greater than or "
                           f"equal to 1, but got {self.check_stresslog_interval_time}. Setting to default value 60")
            self.check_stresslog_interval_time = 60

    def on_train_step_end(self, run_context):
        """Perform actions after each training step and check the criteria."""
        cb_params = run_context.original_args()
        current_step = cb_params.cur_step_num  # Retrieve the current step number

        # Check if interval_steps is set and enough steps have passed
        if self.interval_steps and (current_step - self.last_checked_step >= self.interval_steps):
            self.check_stress_test_model(current_step)
            self.last_checked_step = current_step  # Update the last checked step

    def check_stress_test_model(self, current_step):
        """Perform stress test on current step"""
        logger.info(f"On Step {current_step}, Main Process paused. Running the stress test models...")
        logger.info(f"Stress test model directory is: '{self.model_dir}'")
        logger.info(f"Check stress test logs at {self.stress_test_log_dir} for details.")
        if not self.dataset_dir or not os.path.exists(self.dataset_dir):
            logger.error(f"dataset_dir: {self.dataset_dir} was not found for StressTestModelMonitor, "
                         f"Exiting Stress test.")
            return
        num_cores = os.cpu_count()
        cpu_cores = f"0-{num_cores - 1}"
        logger.debug(f"CPU cores assigned to the stress test task: {cpu_cores}")

        rank_id = get_rank()
        if rank_id % self.worker_num == 0:
            node_num = rank_id//self.worker_num
            saved_dir = os.path.join(self.stress_test_log_dir, "node"+str(node_num))
            command = f"""taskset -c {cpu_cores} bash scripts/msrun_launcher.sh "run_mindformer.py \
                        --config {self.model_dir} \
                        --use_parallel True\
                        --run_mode train \
                        --train_data {self.dataset_dir}" \
                        {self.worker_num} {self.stress_master_port} {saved_dir} True 7200"""

            logger.info(f"Running stress test on node {node_num}, RANK {rank_id} with logs in {saved_dir}")
            log_file_path = os.path.join(saved_dir, "worker_0.log")
            # Start the subprocess
            command = shlex.split(command)
            result_1 = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Monitor the log file
            while result_1.poll() is None:  # While the subprocess is running
                time.sleep(self.check_stresslog_interval_time)
                log_msg = self.readlog(log_file_path)
                logger.info(f"Checking stress test log every {self.check_stresslog_interval_time} seconds")
                logger.info(f"Current state of stress_test: {log_msg}")

            # Once the subprocess has finished, check the result
            if result_1.returncode != 0:
                logger.warning(f"An error occurred while running the stress test model on rank {rank_id}: \
                               {result_1.stderr.read().decode('utf-8')}")
                logger.warning(f"Check the sub task workers log for rank {rank_id} for more details.")
        barrier()

        logger.info(f"Stress tests ended, now starting to collect and compare results")

        # If compare_interval_steps is None, only compare the last step result, and check for its validity.
        if not self.compare_interval_steps:
            logger.warning("For StressTestMonitor, compare_interval_steps is set to None, "
                           "so only the last step result is compared.")
        else:
            interval_results = None
            logger.info(f"Test results are compared every {self.compare_interval_steps} steps")
            for i in range(self.worker_num):
                if get_rank() % self.worker_num == i:
                    node_num = get_rank() // self.worker_num
                    log_dir = os.path.join(self.stress_test_log_dir, "node" + str(node_num))
                    log_file_path = os.path.join(log_dir, f"worker_{i}.log")
                    logger.info(f"log_file_path created with {log_file_path}")
                    interval_results, subtask_global_step_num = self.extract_interval_step_results(log_file_path)
            barrier()

            # Check if the compare_interval_steps is larger than the total steps in the stress test task
            if interval_results is None:
                logger.warning(f"compare_interval_steps {self.compare_interval_steps} is larger than the total number"
                               f" of steps {subtask_global_step_num}, so only the last step result is compared.")
            else:
                gathered_interval_results, _ = all_gather_into_tensor(interval_results)
                gathered_interval_results = gathered_interval_results.asnumpy()
                logger.info("Stress tests interval results collected, now starting to compare interval results")
                logger.debug(f"Collected interval results are {gathered_interval_results}")
                _ = self.compare_gathered_results(gathered_interval_results)

        # Now compare the results from the last step, this is executed regardless of compare_interval_steps setting
        last_step_results = None
        for i in range(self.worker_num):
            if get_rank() % self.worker_num == i:
                node_num = get_rank() // self.worker_num
                log_dir = os.path.join(self.stress_test_log_dir, "node" + str(node_num))
                log_file_path = os.path.join(log_dir, f"worker_{i}.log")
                logger.info(f"log_file_path created with {log_file_path}")
                last_step_results = self.extract_last_step_result(log_file_path)
        barrier()

        gathered_results, _ = all_gather_into_tensor(last_step_results) # <class 'mindspore.common.tensor.Tensor'>
        gathered_results = gathered_results.asnumpy()   # <class 'numpy.ndarray'>
        logger.debug(f"Collected last step results are gathered_results.")
        logger.info("Last step results are collected from each rank, now starting to compare last step results")

        rank0_result = gathered_results[0]
        comparison = np.all(gathered_results == rank0_result, axis=1)
        if np.all(comparison):
            logger.info(f"STRESS TEST PASSED. ALL Results aligned at step {current_step}: "
                        f"[loss, global_norm] = {rank0_result}")
        else:
            unmatched_rank = np.where(~comparison)[0]  # Indices of rows that do not match
            discrepancies = gathered_results[unmatched_rank]  # Get the discrepancies
            logger.warning(f"STRESS TEST FAILED at step {current_step}. Discrepancies found at rank: "
                           f"{unmatched_rank}, values: {discrepancies}.")

        logger.info(f"On Step {current_step}: Stress test ended! Resume training of the main model.")
        return

    def extract_last_step_result(self, log_file):
        """Extract the last step's results from the log file."""
        loss_value = None
        global_norm_value = None
        with open(log_file, 'r') as file:
            lines = file.readlines()
            for line in reversed(lines):
                if "INFO - {" in line:
                    # Extract loss and global_norm values
                    loss_value = self.get_value_from_line(line, r"loss: (\d+\.\d+)")
                    global_norm_value = self.get_value_from_line(line, r"global_norm: \[(\d+\.\d+)\]")
                    if loss_value is not None and global_norm_value is not None:
                        break
        return Tensor([[loss_value, global_norm_value]], ms.float32)

    def extract_interval_step_results(self, log_file):
        """Extract results from specific steps in the middle of log file"""
        last_recorded_step = 0
        results = []
        steps_per_epoch = None
        with open(log_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if "INFO - {" in line:
                    # Get the number of steps per epoch to calculate global step number
                    if not steps_per_epoch:
                        step_info = re.search(r"step:\[\s*\d+/\s*(\d+)\]", line)
                        steps_per_epoch = int(step_info.group(1))

                    # Calculate the global step number
                    epoch_match = re.search(r"Epoch:\[\s*(\d+)", line)
                    step_match = re.search(r"step:\[\s*(\d+)", line)
                    epoch_number = int(epoch_match.group(1))
                    step_number = int(step_match.group(1))
                    global_step_number = (epoch_number - 1) * steps_per_epoch + step_number

                    # Consider logging only if it matches the interval
                    if global_step_number >= (self.compare_interval_steps+last_recorded_step):
                        loss_value = self.get_value_from_line(line, r"loss: (\d+\.\d+)")
                        global_norm_value = self.get_value_from_line(line, r"global_norm: \[(\d+\.\d+)\]")
                        results.append(Tensor([[epoch_number, step_number, loss_value, global_norm_value]], ms.float32))
                        last_recorded_step = global_step_number

        # if results is empty, it means that compare_interval_steps is larger than the total step number
        if not results:
            return None, global_step_number

        results = Tensor(results)
        return results, global_step_number

    def compare_gathered_results(self, gathered_interval_results):
        """Compares results from different ranks at the same epoch and step number."""
        results_dict: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        for result in gathered_interval_results:
            epoch_number, step_number, loss_value, global_norm_value = result[0]
            epoch_step_key = (int(epoch_number), int(step_number))

            # Organize results by epoch and step
            if epoch_step_key not in results_dict:
                results_dict[epoch_step_key] = []
            results_dict[epoch_step_key].append((loss_value, global_norm_value))
        consistent = True
        discrepancies = {}

        # Now iterate through the results_dict to check for consistency
        for epoch_step, values in results_dict.items():
            # Retrieve loss-global pairs for comparison
            loss_global_pairs = [(val[0], val[1]) for val in values]
            # Check if all pairs are consistent
            if all(pair == loss_global_pairs[0] for pair in loss_global_pairs):
                logger.info(f"Results consistent for epoch {epoch_step[0]}, step {epoch_step[1]}: "
                            f"(loss, global_norm) = {loss_global_pairs[0]}")
            else:
                consistent = False
                discrepancies[epoch_step] = []
                # Collect the ranks associated with discrepancies
                for idx, val in enumerate(loss_global_pairs):
                    if val != loss_global_pairs[0]:
                        # Store the index in the original gathered_interval_results
                        discrepancies[epoch_step].append((idx, val))

        if consistent:
            logger.info("ALL INTERVAL TESTS PASSED. All results aligned across all intervals.")
            return True

        for epoch_step, disc_values in discrepancies.items():
            indices = [val[0] for val in disc_values]
            value_pairs = [val[1] for val in disc_values]
            logger.warning(f"STRESS TEST FAILED. DISCREPANCIES found in epoch {epoch_step[0]}, "
                           f"step {epoch_step[1]}: ranks {indices}, (loss, global_norm) = {value_pairs}")
        logger.warning(f"Check the workers log of the problematic rank for detailed results")
        return False

    def get_value_from_line(self, line, pattern):
        """Extracts a numerical value from a line based on a regex pattern."""
        match = re.search(pattern, line)
        if match:
            return float(match.group(1))
        return None

    def readlog(self, file_path):
        """
        Search for the latest line indicating training has started, based on key identifiers.
        """
        with open(file_path, 'r', errors='ignore') as f:
            lines = f.readlines()

        # Define the keywords indicating training has started
        keywords = ['Epoch', 'step', 'loss', 'global_norm']

        # Search backwards for the latest line containing all keywords
        for line in reversed(lines):
            if all(keyword in line for keyword in keywords):
                parts = line.split("- INFO -")
                if len(parts) > 1:
                    return parts[1].strip()  # Return the part after '- INFO -'
                return line.strip()

        # If no such line, training hasn't started
        return "Training has not started yet."


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class SDCMonitor(Callback):
    """Monitor SDC (Silent Data Corruption) by SilentCheck and CheckSum.

    Args:
        initial_step (int, optional): The beginning step. Default: ``0``.
        step_interval (int, optional): The interval of steps to monitor SilentCheck errors in device logs.
            Default: ``10``.
        strike_window_time (int, optional): The window time (minutes) to monitor SilentCheck error. Default: ``480``.
        strike_num (int, optional): The number of SilentCheck error to strike out and start CheckSum. Default: ``3``.
        checksum_time (int, optional): The duration (minutes) of CheckSum. Default: ``5``.
        checksum_cooldown_time (int, optional): The cooldown time (minutes) of CheckSum after it stops.
            Default: ``180``.
    """
    def __init__(self,
                 initial_step: int = 0,
                 step_interval: int = 10,
                 strike_window_time: int = 480,
                 strike_num: int = 3,
                 checksum_time: int = 5,
                 checksum_cooldown_time: int = 180):
        super(SDCMonitor, self).__init__()
        logger.warning('SDCMonitor serves as an experimental interface and its functionality is not yet stable.')

        npu_asd_enable = int(os.getenv('NPU_ASD_ENABLE', '0'))
        ms_sdc_detect_enable = int(os.getenv('MS_SDC_DETECT_ENABLE', '0'))
        if npu_asd_enable != 1 or ms_sdc_detect_enable != 1 or not is_version_ge(ms.__version__, '2.7.0'):
            raise ValueError("SDCMonitor needs mindspore >= 2.7.0, and only works when environment variable "
                             "'NPU_ASD_ENABLE' and 'MS_SDC_DETECT_ENABLE' are set to 1.")

        self.initial_step = initial_step
        self.step_interval = step_interval
        self.step_times = {datetime.now(): initial_step} # {timestamp: step}
        self.silent_check_error_times = {} # {timestamp: step}
        self.strike_window_time = timedelta(minutes=strike_window_time)
        self.strike_num = strike_num
        self.checksum_enable = False
        self.prev_checksum_time = datetime.min # start/stop time
        self.checksum_time = timedelta(minutes=checksum_time)
        self.checksum_cooldown_time = timedelta(minutes=checksum_cooldown_time)

        self.device_log_path = os.path.join(get_ascend_log_path(), 'debug', f'device-{get_real_local_rank()}')
        # device log file: device-<pid>_<timestamp>.log, e.g. device-311523_20250225184632284.log
        self.prev_log_file_time = "0"
        pid = os.getpid()
        self.log_file_pattern = re.compile(rf'device-{pid}_(\d{{17}})\.log$')
        self.silent_check_error_pattern = re.compile(r'^\[ERROR\].*silent_check_v3\.cc:.*SilentCheckV3', re.MULTILINE)
        # device log time: YYYY-MM-DD-HH:MM:SS.SSS.SSS
        self.log_time_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}\.\d{3}\.\d{3})')
        logger.info(f"Device log path: {self.device_log_path}, pid: {pid}")

        self.all_reduce_net = AllReduceNet(GlobalComm.WORLD_COMM_GROUP) # AllReduce status and result of CheckSum

    def _get_log_files_to_check(self):
        """Get device log filenames after last check and sort them by timestamp."""
        log_files = []
        if not os.path.exists(self.device_log_path):
            return log_files
        for f in os.listdir(self.device_log_path):
            match = self.log_file_pattern.match(f)
            if match and match.group(1) >= self.prev_log_file_time:
                log_files.append(f)
        log_files.sort()
        return log_files

    def _parse_silent_check_error_times(self, log_files):
        """Parse SilentCheck error times of step from device logs"""
        # parse error log times, log file size < 20MB
        error_log_times = []
        for file in log_files:
            file_path = os.path.join(self.device_log_path, file)
            if not os.path.exists(file_path):
                continue
            with open(file_path, 'r') as f:
                logs = f.read()
            error_logs = self.silent_check_error_pattern.findall(logs)
            for log in error_logs:
                match = self.log_time_pattern.search(log)
                if match:
                    log_time = match.group(1)
                    # merge ms and us in str then convert to datetime
                    log_time = re.sub(r'\.(\d{3})\.(\d{3})', lambda m: f".{m.group(1)}{m.group(2)}", log_time)
                    log_time = datetime.strptime(log_time, "%Y-%m-%d-%H:%M:%S.%f")
                    error_log_times.append(log_time)
        if not error_log_times:
            return {}
        # process from latest to earliest, stop early if error num reaches strike num
        error_times = {} # {timestamp: step}
        step_time_list = list(self.step_times.keys())
        index = len(step_time_list) - 1
        for log_time in reversed(error_log_times):
            while index > 0 and log_time <= step_time_list[index - 1]:
                index -= 1
            if index == 0 or len(error_times) == self.strike_num:
                break
            left, right = step_time_list[index - 1], step_time_list[index]
            # all SilentCheck errors in a step is treated as one error
            if left < log_time <= right:
                step = self.step_times[right]
                logger.warning(f"SilentCheck detect SDC at step: {step}")
                error_times[log_time] = step
                index -= 1
        return dict(reversed(list(error_times.items()))) # order from earliest to latest

    def _update_silent_check_error_times(self, new_silent_check_error_times, now):
        """Add new SilentCheck error times and remove expired ones."""
        self.silent_check_error_times.update(new_silent_check_error_times)
        expired_error_times = []
        for error_time, _ in self.silent_check_error_times.items():
            if now - error_time > self.strike_window_time:
                expired_error_times.append(error_time)
        for error_time in expired_error_times:
            self.silent_check_error_times.pop(error_time)

    def _start_checksum(self, step):
        """Sync CheckSum enable status on all ranks and start CheckSum."""
        # set context to skip pp validation during global AllReduce
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE)
        self.checksum_enable = bool(
            self.all_reduce_net(ms.Tensor([self.checksum_enable], ms.int32)).asnumpy()[0])
        ms.set_auto_parallel_context(parallel_mode=parallel_mode)
        if self.checksum_enable:
            logger.info(f"Start CheckSum at step: {step}")
            self.prev_checksum_time = datetime.now()
            ms.sdc_detect_start()

    def _stop_checksum(self, step):
        """Stop CheckSum and aggregate SDC detection result."""
        logger.warning(f"Stop CheckSum at step: {step}")
        ms.sdc_detect_stop()
        self.checksum_enable = False
        now = datetime.now()
        self.prev_checksum_time = now
        has_sdc = ms.get_sdc_detect_result()
        if has_sdc:
            logger.warning(f"CheckSum detects SDC on rank {get_real_rank()}")
        # set context to skip pp validation during global AllReduce
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE)
        has_sdc = bool(self.all_reduce_net(ms.Tensor([has_sdc], ms.int32)).asnumpy()[0])
        ms.set_auto_parallel_context(parallel_mode=parallel_mode)
        if has_sdc:
            logger.warning("Detect SDC by SilentCheck and CheckSum, which means training may be unstable. "
                           "Check training logs and device logs of each rank for more details.")
        self.silent_check_error_times.clear()
        self.step_times = {now: step}

    def on_train_step_end(self, run_context):
        """Monitor SilentCheck errors and manage CheckSum if strike out."""
        cb_params = run_context.original_args()
        cur_step_num = cb_params.cur_step_num + self.initial_step

        now = datetime.now()
        # stop CheckSum and clear previous SilentCheck errors
        if self.checksum_enable:
            if now - self.prev_checksum_time >= self.checksum_time:
                self._stop_checksum(cur_step_num)
            return

        self.step_times[now] = cur_step_num
        # parse device logs and start CheckSum if strike out
        if cb_params.cur_step_num % self.step_interval == 0:
            logger.info(f"Checking device logs at step: {cur_step_num}...")
            log_files = self._get_log_files_to_check()
            new_silent_check_error_times = self._parse_silent_check_error_times(log_files)
            self._update_silent_check_error_times(new_silent_check_error_times, now)
            if now - self.prev_checksum_time >= self.checksum_cooldown_time:
                if len(self.silent_check_error_times) >= self.strike_num:
                    self.checksum_enable = True
                    logger.warning(f"SDC {self.strike_num} strikes and out on rank: {get_real_rank()}, "
                                   f"SilentCheck error steps: {list(self.silent_check_error_times.values())}")
                # any rank stikes out will enable CheckSum in all ranks by AllReduce
                self._start_checksum(cur_step_num)
            self.step_times = {now: cur_step_num}
            if log_files:
                self.prev_log_file_time = re.search(r'_(\d{17})\.log$', log_files[-1]).group(1)
