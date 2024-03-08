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
import time
import datetime

from collections import OrderedDict
from copy import deepcopy
from typing import Callable, Optional, Union

import numpy as np
import mindspore as ms
from mindspore import Callback, Profiler, ModelCheckpoint, CheckpointConfig, context, save_checkpoint, Tensor
from mindspore.train.callback import SummaryCollector
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.train.callback._callback import set_cur_net
from mindspore.train.serialization import _get_merged_param_data
from mindspore.nn.cell import Cell
from mindspore.ops.operations.comm_ops import Broadcast
from mindspore.common import jit

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.cloud_adapter.cloud_adapter import Local2ObsMonitor
from mindformers.tools.logger import logger
from mindformers.tools.utils import get_output_root_path, get_output_subpath, get_remote_save_url, check_in_modelarts,\
    get_real_rank, get_real_group_size

__all__ = ['ObsMonitor', 'MFLossMonitor', 'CheckpointMointor', 'SummaryMonitor', 'ProfileMonitor', 'EvalCallBack']

_cur_dir = os.getcwd()
SAVE_DIR = _cur_dir

@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class ObsMonitor:
    """
    Obs Monitor For Local and AICC.

    Args:
        src_dir (str): The output path in Local/AICC. Default: None.
        target_dir (str): The remote url to save files. Default: None.
        step_upload_frequence (int): The step interval of uploading. Default: 100.
        epoch_upload_frequence (int): The epoch interval of uploading. Default: -1, means epoch upload is disabled.
        keep_last (bool): Check the consistency of obs files and AICC. Default: True.
    Examples:
        >>> from mindformers.core.callback import ObsMonitor
        >>> monitor = ObsMonitor(src_dir='./root_path', target_dir='./remote_url')
    """

    def __new__(cls,
                src_dir: str = None,
                target_dir: str = None,
                step_upload_frequence: int = 100,
                epoch_upload_frequence: int = -1,
                keep_last: bool = True):
        if src_dir is None:
            src_dir = get_output_root_path()
        if target_dir is None:
            target_dir = get_remote_save_url()
        return Local2ObsMonitor(src_dir, target_dir, step_upload_frequence, epoch_upload_frequence, keep_last)


def _get_loss_output(output):
    """Get output of task for MFLossMonitor."""
    overflow = False
    scaling_sens = False
    loss = output
    learning_rate = None
    if isinstance(output, (tuple, list)):
        if len(output) == 3:
            loss, overflow, scaling_sens = output
            if isinstance(scaling_sens, ms.Tensor):
                scaling_sens = scaling_sens.asnumpy()
        elif len(output) == 4:
            loss, overflow, scaling_sens, learning_rate = output
            if isinstance(scaling_sens, ms.Tensor):
                scaling_sens = scaling_sens.asnumpy()
        else:
            if isinstance(output[0], ms.Tensor) and isinstance(output[0].asnumpy(), np.ndarray):
                loss = output[0]

    if isinstance(loss, ms.Tensor) and isinstance(loss.asnumpy(), np.ndarray):
        loss = np.mean(loss.asnumpy())

    # Boundary check.
    if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
        invalid_loss_info = "NaN" if np.isnan(loss) else "Inf"
        raise ValueError(f"The current value of loss is {invalid_loss_info}, terminate training.")

    return loss, overflow, scaling_sens, learning_rate


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class MFLossMonitor(Callback):
    """
    Loss Monitor for classification.

    Args:
        learning_rate (Union[float, LearningRateSchedule], optional): The learning rate schedule. Default: None.
        per_print_times (int): Every how many steps to print the log information. Default: 1.
        micro_batch_num (int): MicroBatch size for Pipeline Parallel. Default: 1.
        micro_batch_interleave_num (int): split num of batch size. Default: 1.
        origin_epochs (int): Training epoches. Default: None.
        dataset_size (int): Training dataset size. Default: None.
        initial_epoch (int): The beginning epoch. Default: 0.
        global_batch_size (int): The total batch size. Default: 0.
        device_num (int): The number of device in use. Default: 0.
    Examples:
        >>> from mindformers.core.callback import MFLossMonitor
        >>> lr = [0.01, 0.008, 0.006, 0.005, 0.002]
        >>> monitor = MFLossMonitor(per_print_times=10)
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
                 gradient_accumulation_steps: int = 1):
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

    def epoch_begin(self, run_context):
        """
        Record time at the beginning of epoch.

        Args:
            run_context (RunContext): Context of the process running.
        """
        self.loss_list = []
        self.epoch_time = time.time()
        self.run_context = run_context

    def epoch_end(self, run_context):
        """
        Print training info at the end of epoch.

        Args:
            run_context (RunContext): Context of the process running.
        """

    def step_begin(self, run_context):
        """
        Record time at the beginning of step.

        Args:
            run_context (RunContext): Context of the process running.
        """
        self.step_time = time.time()
        self.run_context = run_context

    def step_end(self, run_context):
        """
        Print training info at the end of step.

        Args:
            run_context (RunContext): Context of the process running.
        """
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        full_batch = ms.get_auto_parallel_context("full_batch")
        auto_parallel = parallel_mode in ['semi_auto_parallel', 'auto_parallel']
        if auto_parallel:
            ms.context.set_auto_parallel_context(parallel_mode='data_parallel', full_batch=False)
        cb_params = run_context.original_args()
        step_seconds = (time.time() - self.step_time) * 1000
        net_outputs = cb_params.net_outputs
        loss, overflow, scaling_sens, learning_rate = _get_loss_output(net_outputs)
        if learning_rate is not None:
            self.learning_rate = learning_rate
        loss = self._fix_loss_for_parallel(loss)
        self.loss_list.append(loss)

        if not overflow:
            overflow = "False"
        if not scaling_sens:
            scaling_sens = "unavailable"

        if cb_params.dataset_sink_mode:
            origin_epochs = self.origin_epochs
            per_step_seconds = step_seconds / cb_params.batch_num
            steps_per_epoch = self.steps_per_epoch
            cur_epoch_num = (cb_params.cur_step_num + self.initial_step - 1) // steps_per_epoch + 1
            cur_step_num = (cb_params.cur_step_num + self.initial_step - 1) % steps_per_epoch + 1
        else:
            origin_epochs = self.origin_epochs
            per_step_seconds = step_seconds
            steps_per_epoch = cb_params.batch_num
            cur_epoch_num = cb_params.cur_epoch_num
            cur_step_num = (cb_params.cur_step_num + self.initial_step - 1) % cb_params.batch_num + 1

        # compute time remaining
        step_remain = (origin_epochs - cur_epoch_num + 1) * steps_per_epoch - cur_step_num
        time_remain = step_remain * per_step_seconds / 1000

        # compute throughput
        throughput = self.global_batch_size / self.device_num / (per_step_seconds / 1000)

        # compute percent
        percent = ((cur_epoch_num - 1) * steps_per_epoch +  cur_step_num) / origin_epochs / steps_per_epoch * 100

        if (cb_params.cur_step_num - self.last_print_time) >= self.per_print_times:
            self.last_print_time = cb_params.cur_step_num
            self.print_output_info(cb_params, cur_epoch_num, origin_epochs, throughput,
                                   cur_step_num, steps_per_epoch, loss, per_step_seconds,
                                   overflow, scaling_sens, time_remain, percent)


        if check_in_modelarts() and get_real_rank() == get_real_group_size() - 1:
            self.dump_info_to_modelarts(ma_step_num=cur_step_num, ma_loss=loss)

        if auto_parallel:
            ms.context.set_auto_parallel_context(parallel_mode=parallel_mode, full_batch=full_batch)

    def _fix_loss_for_parallel(self, loss):
        """Fix loss value in pipeline or double parallel mode."""
        pipeline_stages = ms.context.get_auto_parallel_context("pipeline_stages")
        if pipeline_stages > 1 and self.print_warning_flag:
            logger.warning("pipeline stages: %s > 1, the loss on the last card is valid.",
                           pipeline_stages)

        if self.micro_batch_interleave_num > 1 and self.print_warning_flag:
            logger.warning("micro_batch_interleave_num: %s > 1, multiple copies in parallel is open.")

        if pipeline_stages > 1:
            loss = loss / self.mirco_size
        if self.micro_batch_interleave_num > 1:
            loss = loss / self.micro_batch_interleave_num
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps

        return loss

    def print_output_info(self, cb_params, cur_epoch_num, origin_epochs, throughput,
                          cur_step_num, steps_per_epoch, loss, per_step_seconds,
                          overflow, scaling_sens, time_remain, percent):
        """print output information."""
        if self.learning_rate is not None:
            if isinstance(self.learning_rate, (float, Tensor)):
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

        if current_lr is not None:
            if cb_params.dataset_sink_mode:
                logger.info("{ Epoch:[%3d/%3d], step:[%5d/%5d], loss: %5.3f, "
                            "per_step_time: %dms, lr: %s, overflow cond: %s, loss_scale: %s",
                            cur_epoch_num, origin_epochs, cur_step_num, steps_per_epoch, loss,
                            int(per_step_seconds), current_lr, overflow, scaling_sens)
            else:
                logger.info("{ Epoch:[%3d/%3d], step:[%5d/%5d], loss:[%5.3f/%5.3f], "
                            "per_step_time: %dms, lr: %s, overflow cond: %s, loss_scale: %s",
                            cur_epoch_num, origin_epochs, cur_step_num, steps_per_epoch, loss, np.mean(self.loss_list),
                            int(per_step_seconds), current_lr, overflow, scaling_sens)
            show_str = ('|%%-%ds|' % 50) % (int(50 * percent / 100) * "█")
            logger.info("  %4.1f%% %s %.5f samples/s/p  %s }", percent, show_str, throughput,
                        datetime.timedelta(seconds=int(time_remain)))
        else:
            if cb_params.dataset_sink_mode:
                logger.info("{ Epoch:[%3d/%3d], step:[%5d/%5d], loss: %5.3f, "
                            "per_step_time: %dms, overflow cond: %s, loss_scale: %s",
                            cur_epoch_num, origin_epochs, cur_step_num, steps_per_epoch, loss,
                            int(per_step_seconds), overflow, scaling_sens)
            else:
                logger.info("{ Epoch:[%3d/%3d], step:[%5d/%5d], loss:[%5.3f/%5.3f], "
                            "per_step_time: %dms, overflow cond: %s, loss_scale: %s",
                            cur_epoch_num, origin_epochs, cur_step_num, steps_per_epoch, loss, np.mean(self.loss_list),
                            int(per_step_seconds), overflow, scaling_sens)
            show_str = ('|%%-%ds|' % 50) % (int(50 * percent / 100) * "█")
            logger.info("  %4.1f%% %s %.5f samples/s/p  %s }", percent, show_str, throughput,
                        datetime.timedelta(seconds=int(time_remain)))

    def dump_info_to_modelarts(self, ma_step_num, ma_loss):
        """dump modelarts info to display evaluation result page"""
        ma_loss = float(ma_loss)
        obj = None
        modelarts_dir = os.path.join(get_output_root_path(), "modelarts")
        if not os.path.exists(modelarts_dir):
            os.mkdir(modelarts_dir)
        if not os.path.exists(os.path.join(modelarts_dir, "model_analysis_results.json")):
            obj = {
                "en-us": {
                    "common": {},
                    "precision_performance": {
                        "pr": {
                            "title": "loss", "description": "loss of model", "value": {"current_loss": 0},
                            "line_chart": {
                                "pr_line_chart": {
                                    "name": "loss line chart of model",
                                    "x_axis_name": "step",
                                    "y_axis_name": "loss",
                                    "curve": {"loss": []}}}}},
                    "feature_sensitivity": {},
                    "computational_performance": {},
                    "abstract_feature": {},
                    "adversary": {}
                },
                "zh-cn": {
                    "common": {},
                    "precision_performance": {
                        "pr": {
                            "title": "loss", "description": "模型损失", "value": {"当前loss": 0},
                            "line_chart": {
                                "pr_line_chart": {
                                    "name": "loss line chart of model",
                                    "x_axis_name": "step",
                                    "y_axis_name": "loss",
                                    "curve": {"loss": []}}}}},
                    "feature_sensitivity": {},
                    "computational_performance": {},
                    "abstract_feature": {},
                    "adversary": {}
                }
            }
        else:
            with open(os.path.join(modelarts_dir, "model_analysis_results.json"), "r") as fp:
                obj = json.load(fp)

        if obj is not None:
            en_precision_performance = obj["en-us"]["precision_performance"]
            en_precision_performance["pr"]["value"]["loss_value"] = ma_loss
            en_loss_list = en_precision_performance["pr"]["line_chart"]["pr_line_chart"]["curve"]["loss"]
            en_loss_list.append([ma_step_num, ma_loss])

            zh_precision_performance = obj["zh-cn"]["precision_performance"]
            zh_precision_performance["pr"]["value"]["当前loss"] = ma_loss
            zh_loss_list = zh_precision_performance["pr"]["line_chart"]["pr_line_chart"]["curve"]["loss"]
            zh_loss_list.append([ma_step_num, ma_loss])

            with open(os.path.join(modelarts_dir, "model_analysis_results.json"), "w") as fp:
                json.dump(obj, fp)


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class SummaryMonitor:
    """
    Summary Monitor For AICC and Local.

    Args:
        summary_dir (str):
            The collected data will be persisted to this directory. Default: None.
        collect_freq (int):
            Set the frequency of data collection, it should be greater than zero, and the unit is `step`.
            Default: 10.
        collect_specified_data (Union[None, dict]):
            Perform custom operations on the collected data. Default: None.
        keep_default_action (bool):
            This field affects the collection behavior of the 'collect_specified_data' field. Default: True.
        custom_lineage_data (Union[dict, None]):
            Allows you to customize the data and present it on the MingInsight lineage page. Default: None.
        collect_tensor_freq (Optional[int]):
            The same semantics as the `collect_freq`, but controls TensorSummary only. Default: None.
        max_file_size (Optional[int]):
            The maximum size in bytes of each file that can be written to the disk. Default: None.
        export_options (Union[None, dict]):
            Perform custom operations on the export data. Default: None.
    Examples:
        >>> from mindformers.core.callback import SummaryMonitor
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
class CheckpointMointor(ModelCheckpoint):
    """
    Checkpoint Monitor For Save LossScale.

    Args:
        prefix (str): The prefix name of checkpoint files. Default: 'CKP'.
        directory (str): The path of the folder which will be saved in the checkpoint file. Default: None.
        config (CheckpointConfig): Checkpoint strategy configuration. Default: None.
        save_checkpoint_steps (int): Steps to save checkpoint. Default: 1.
        save_checkpoint_seconds (int): Seconds to save checkpoint.
                                       Can't be used with save_checkpoint_steps at the same time. Default: 0.
        keep_checkpoint_max (int): Maximum number of checkpoint files can be saved. Default: 5.
        keep_checkpoint_per_n_minutes (int): Save the checkpoint file every "keep_checkpoint_per_n_minutes" minutes.
                                             Can't be used with keep_checkpoint_max at the same time. Default: 0.
        integrated_save (bool): Whether to merge and save the split Tensor in the automatic parallel scenario.
                                Integrated save function is only supported in automatic parallel scene. Default: True.
        save_network_params (bool): Whether to only save network weights additionally. Default: True.
        save_trainable_params (bool): Whether to save fine-tuned weights additionally. Default: False.
        async_save (bool): Whether asynchronous execution saves the checkpoint to a file. Default: False.
        saved_network (Cell): Network to be saved in checkpoint file. Default: None.
        append_info (list): The information save to checkpoint file.
                            Support "epoch_num", "step_num" and dict. Default: None.
        enc_key (Union[None, bytes]): Byte type key used for encryption. Default: None.
        enc_mode (str): This parameter is valid only when "enc_key" is not set to None. Specifies the encryption
                        mode, currently supports 'AES-GCM', 'AES-CBC' and 'SM4-CBC'. Default: 'AES-GCM'.
        exception_save (bool): Whether to save the current checkpoint when an exception occurs. Default: False.

    Raises:
        ValueError: If `prefix` is not str or contains the '/' character.
        ValueError: If `directory` is not str.
        TypeError: If the config is not CheckpointConfig type.

    Examples:
        >>> from mindformers.core.callback import CheckpointMointor
        >>> monitor = CheckpointMointor(directory='./checkpoint_dir')
    """

    def __init__(self, prefix='CKP',
                 directory=None,
                 config=None,
                 save_checkpoint_steps=1,
                 save_checkpoint_seconds=0,
                 keep_checkpoint_max=5,
                 keep_checkpoint_per_n_minutes=0,
                 integrated_save=True,
                 save_network_params=True,
                 save_trainable_params=False,
                 async_save=False,
                 saved_network=None,
                 append_info=None,
                 enc_key=None,
                 enc_mode='AES-GCM',
                 exception_save=False):

        self.config = config
        self.save_network_params = save_network_params
        self.save_trainable_params = save_trainable_params
        self.rank_id = get_real_rank()
        prefix = prefix + "_rank_{}".format(self.rank_id)

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
                                     exception_save=exception_save)
        super(CheckpointMointor, self).__init__(prefix, ckpt_directory, config=config_ck)

    def _save_ckpt(self, cb_params, force_to_save=False):
        """Save checkpoint files."""
        # pylint: disable=E0203
        if cb_params.cur_step_num == self._last_triggered_step:
            return

        # if param is cache enable, flush data from cache to host before save_ckpt
        if self._need_flush_from_cache:
            self._flush_from_cache(cb_params)

        save_ckpt = self._check_save_ckpt(cb_params, force_to_save)
        step_num_in_epoch = int((cb_params.cur_step_num - 1) % cb_params.batch_num + 1)

        if save_ckpt:
            logger.info('......Saving ckpt......')
            cur_ckpoint_file = self._prefix + "-" + str(cb_params.cur_epoch_num) + "_" \
                               + str(step_num_in_epoch) + ".ckpt"
            # update checkpoint file list.
            self._manager.update_ckpoint_filelist(self._directory, self._prefix)
            # keep checkpoint files number equal max number.
            if self._config.keep_checkpoint_max and 0 < self._config.keep_checkpoint_max <= self._manager.ckpoint_num:
                self._manager.remove_oldest_ckpoint_file()
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

            if context.get_context("enable_ge") and os.getenv("MS_ENABLE_REF_MODE", "0") == "0":
                set_cur_net(cb_params.train_network)
                cb_params.train_network.exec_checkpoint_graph()
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
            network = self._config.saved_network if self._config.saved_network is not None else cb_params.train_network
            save_checkpoint(network, cur_file, self._config.integrated_save, self._config.async_save,
                            self._append_dict, self._config.enc_key, self._config.enc_mode)

            def save_only_network_params():
                save_obj = cb_params.network
                if hasattr(save_obj, 'optimizer') and save_obj.optimizer is not None:
                    save_obj = save_obj.network

                if self.save_network_params:
                    cb_cur_ckpoint_file = self._prefix + "-" + "network" + ".ckpt"
                    cb_cur_file = os.path.join(self.network_directory, cb_cur_ckpoint_file)
                    os.makedirs(self.network_directory, exist_ok=True)
                    save_checkpoint(save_obj, cb_cur_file, self._config.integrated_save, self._config.async_save,
                                    {}, self._config.enc_key, self._config.enc_mode)

                if self.save_trainable_params:
                    save_obj.init_parameters_data()
                    param_dict = OrderedDict()
                    for param in save_obj.trainable_params():
                        param_dict[param.name] = param
                    param_list = []
                    for (key, value) in param_dict.items():
                        each_param = {"name": key}
                        param_data = Tensor(value.data.asnumpy())

                        # in automatic model parallel scenario, some parameters were split to all the devices,
                        # which should be combined before saving
                        if key in save_obj.parameter_layout_dict:
                            param_data = _get_merged_param_data(save_obj, key, param_data, self._config.integrated_save)

                        each_param["data"] = param_data
                        param_list.append(each_param)
                    save_obj = param_list
                    cb_cur_ckpoint_file = self._prefix + "-" + "trainable_params" + ".ckpt"
                    cb_cur_file = os.path.join(self.trainable_directory, cb_cur_ckpoint_file)
                    os.makedirs(self.trainable_directory, exist_ok=True)
                    save_checkpoint(save_obj, cb_cur_file, self._config.integrated_save,
                                    self._config.async_save, {}, self._config.enc_key, self._config.enc_mode)

            save_only_network_params()

            self._latest_ckpt_file_name = cur_file


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class ProfileMonitor(Callback):
    """
    Profile analysis in training.

    Args:
        start_step (int): The step to start profiling. Default: 1.
        stop_step (int): The step to stop profiling. Default: 10.
        output_path (str): The result of profiling will be saved in this path. Default: None.
        start_profile (str): Whether to enable profiling. Default: True.
        profile_communication (str): Whether to collect communication performance data
                                     during multi-device training. Default: False.
        profile_memory (str): Whether to collect Tensor memory data. Default: True.
    Examples:
        >>> from mindformers.core.callback import ProfileMonitor
        >>> monitor = ProfileMonitor(output_path='./profile_dir')
    """

    def __init__(self, start_step=1, stop_step=10,
                 output_path=None, start_profile=True,
                 profile_communication=False, profile_memory=True, **kwargs):
        super(ProfileMonitor, self).__init__()
        self.start_step = start_step
        self.stop_step = stop_step
        self.start_profile = start_profile
        self.profile_communication = profile_communication

        if profile_communication and not start_profile:
            raise ValueError("When profile_communication is True, start_profile must also be True")

        if output_path is None:
            rank_id = get_real_rank()
            output_path = get_output_subpath('profile', rank_id)

        if ms.get_context("device_target") == "GPU" and profile_memory:
            logger.warning("The parameter profile_memory is not supported on GPU currently, so is changed to False. ")
            profile_memory = False

        self.profiler = Profiler(
            start_profile=start_profile, output_path=output_path,
            profile_communication=profile_communication, profile_memory=profile_memory, **kwargs)

        self.run_context = None
        self.output_path = output_path

    def step_begin(self, run_context):
        """
        Start profile at the begin of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.start_step and not self.start_profile:
            self.profiler.start()

    def step_end(self, run_context):
        """
        Stop profile at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.stop_step:
            self.profiler.stop()
            self.profiler.analyse()
            logger.info("End of Profiling, please view the profile data under %s and analyze it using mindinsight."
                        "MindInsight order as follow: "
                        "mindinsight start --summary-base-dir %s", self.output_path, self.output_path)


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class EvalCallBack(Callback):
    """
    Evaluate Callback used in training progress.

    Args:
        eval_func (Callable): The function calculates eval result and is task specific.
        step_interval (int): Determine the num of step intervals between each eval.
                             Default 100, means only eval on epoch end, do not eval between steps.
                             Note that it will not take effects when running in data sink mode.
        epoch_interval (int): Determine the num of epoch intervals between each eval.
                              Default -1, means eval on every epoch end.
    Examples:
        >>> from mindformers.core.callback import EvalCallBack
        >>> def eval_func():
        ...     print("output result")
        >>> eval_callback = EvalCallBack(eval_func=eval_func)
        >>> type(eval_callback)
        <class 'mindformers.core.callback.callback.EvalCallBack'>
    """

    def __init__(self, eval_func: Callable, step_interval: int = 100, epoch_interval: int = -1):
        self.eval_func = eval_func
        self.step_interval = step_interval
        self.epoch_interval = epoch_interval

    def epoch_end(self, run_context):
        # if not use epoch end
        if self.epoch_interval <= 0:
            return
        callback_params = run_context.original_args()
        cur_epoch_num = callback_params.cur_epoch_num
        if cur_epoch_num % self.epoch_interval == 0:
            self._execute_eval()

    def step_end(self, run_context):
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
class ColdHotExpertMointor(Callback):
    """
        ColdHotExpertMointor Callback used in MoE model training progress.

        Args:
            config : Read config from configuration file.

        Examples:
            >>> from mindformers.core.callback import ColdHotExpertMointor
            >>> callback = ColdHotExpertMointor(config)
            >>> type(callback)
            <class 'mindformers.core.callback.callback.ColdHotExpertMointor'>
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
        self.local_expert_index = [i for i in range(
            (self.rank_id // self.mp) * self.local_expert_num,
            (self.rank_id // self.mp) * self.local_expert_num + self.local_expert_num)]
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
        Obtains MoE blocks modules in obj by path..

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
