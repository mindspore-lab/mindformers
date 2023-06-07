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
import os
import time
from copy import deepcopy
from typing import Callable, Optional, Union

import numpy as np
import mindspore as ms
from mindspore import Callback, Profiler, ModelCheckpoint, CheckpointConfig, context, save_checkpoint
from mindspore.train.callback import SummaryCollector
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.train.callback._callback import set_cur_net

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.cloud_adapter.cloud_adapter import Local2ObsMonitor
from mindformers.tools.logger import logger
from mindformers.tools.utils import get_output_root_path, get_output_subpath, get_remote_save_url

__all__ = ['ObsMonitor', 'MFLossMonitor', 'CheckpointMointor', 'SummaryMonitor', 'ProfileMonitor', 'EvalCallBack']

_cur_dir = os.getcwd()
SAVE_DIR = _cur_dir


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class ObsMonitor:
    """Obs Monitor For AICC and Local"""
    def __new__(cls,
                src_dir: str = None,
                target_dir: str = None,
                rank_id: int = None,
                upload_frequence: int = -1,
                keep_last: bool = True):
        if src_dir is None:
            src_dir = get_output_root_path()
        if target_dir is None:
            target_dir = get_remote_save_url()
        return Local2ObsMonitor(src_dir, target_dir, rank_id, upload_frequence, keep_last)


def _get_loss_output(output):
    """Get output of task for MFLossMonitor."""
    overflow = False
    scaling_sens = False
    loss = output
    if isinstance(output, (tuple, list)):
        if len(output) == 3:
            loss, overflow, scaling_sens = output
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

    return loss, overflow, scaling_sens


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
                 initial_epoch: int = 0):
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
        callback_params = run_context.original_args()
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / callback_params.batch_num
        logger.info(
            "Epoch time: %5.3f ms, "
            "per step time: %5.3f ms, "
            "avg loss: %5.3f", epoch_mseconds, per_step_mseconds, np.mean(self.loss_list))

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
        loss, overflow, scaling_sens = _get_loss_output(net_outputs)
        loss = self._fix_loss_for_parallel(loss)
        self.loss_list.append(loss)

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if not overflow:
            overflow = "False"
        if not scaling_sens:
            scaling_sens = "unavailable"

        if cb_params.dataset_sink_mode:
            origin_epochs = self.origin_epochs
            steps_per_epoch = self.steps_per_epoch
            cur_epoch_num = (cb_params.cur_step_num - 1) // steps_per_epoch \
                            + self.initial_epoch * cb_params.batch_num // steps_per_epoch + 1
            cur_step_num = (cb_params.cur_step_num - 1) % steps_per_epoch + 1
        else:
            origin_epochs = self.origin_epochs
            steps_per_epoch = cb_params.batch_num
            cur_step_num = cur_step_in_epoch
            cur_epoch_num = cb_params.cur_epoch_num

        if (cb_params.cur_step_num - self.last_print_time) >= self.per_print_times:
            self.last_print_time = cb_params.cur_step_num
            self.print_output_info(cb_params, cur_epoch_num, origin_epochs,
                                   cur_step_num, steps_per_epoch, loss, step_seconds,
                                   overflow, scaling_sens)

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
            loss = loss / (self.mirco_size * self.micro_batch_interleave_num)
        elif self.micro_batch_interleave_num > 1:
            loss = loss / self.micro_batch_interleave_num

        return loss

    def print_output_info(self, cb_params, cur_epoch_num, origin_epochs,
                          cur_step_num, steps_per_epoch, loss, step_seconds,
                          overflow, scaling_sens):
        """print output information."""
        if self.learning_rate is not None:
            if isinstance(self.learning_rate, float):
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
                    current_lr = self.learning_rate(global_step)
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
            logger.info(
                "Epoch:[%3d/%3d], step:[%5d/%5d], "
                "loss:[%5.3f/%5.3f], time:%5.3f ms, "
                "lr:%s, overflow cond: %s, loss_scale: %s", cur_epoch_num, origin_epochs,
                cur_step_num, steps_per_epoch, loss, np.mean(self.loss_list),
                step_seconds, current_lr, overflow, scaling_sens)
        else:
            logger.info(
                "Epoch:[%3d/%3d], step:[%5d/%5d], "
                "loss:[%5.3f/%5.3f], time:%5.3f ms, "
                "overflow cond: %s, loss_scale: %s", cur_epoch_num, origin_epochs,
                cur_step_num, steps_per_epoch, loss, np.mean(self.loss_list),
                step_seconds, overflow, scaling_sens)


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class SummaryMonitor:
    """Summary Monitor For AICC and Local"""
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
            rank_id = int(os.getenv("RANK_ID", '0'))
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
    """Checkpoint Monitor For Save LossScale"""
    def __init__(self, prefix='CKP',
                 directory=None,
                 config=None,
                 save_checkpoint_steps=1,
                 save_checkpoint_seconds=0,
                 keep_checkpoint_max=5,
                 keep_checkpoint_per_n_minutes=0,
                 integrated_save=True,
                 async_save=False,
                 saved_network=None,
                 append_info=None,
                 enc_key=None,
                 enc_mode='AES-GCM',
                 exception_save=False):

        self.config = config
        self.rank_id = int(os.getenv("RANK_ID", '0'))
        prefix = prefix + "_rank_{}".format(self.rank_id)

        if append_info is None:
            append_info = [{
                "epoch_num": 0,
                "step_num": 0,
                "global_step": 0,
                "loss_scale": 1
            }]
        directory = get_output_subpath('checkpoint', self.rank_id)
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
        super(CheckpointMointor, self).__init__(prefix, directory, config=config_ck)

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

            if context.get_context("enable_ge"):
                set_cur_net(cb_params.train_network)
                cb_params.train_network.exec_checkpoint_graph()
            if "epoch_num" in self._append_dict:
                self._append_dict["epoch_num"] = self._append_epoch_num + cb_params.cur_epoch_num
            if "step_num" in self._append_dict:
                self._append_dict["step_num"] = self._append_step_num + cb_params.cur_epoch_num * cb_params.batch_num
            if cb_params.optimizer is not None:
                self._append_dict["global_step"] = cb_params.optimizer.global_step
            else:
                self._append_dict["global_step"] = cb_params.network.optimizer.global_step
            if "loss_scale" in self._append_dict:
                outputs = cb_params.net_outputs
                if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
                    self._append_dict["loss_scale"] = outputs[2]
            network = self._config.saved_network if self._config.saved_network is not None else cb_params.train_network
            save_checkpoint(network, cur_file, self._config.integrated_save, self._config.async_save,
                            self._append_dict, self._config.enc_key, self._config.enc_mode)

            self._latest_ckpt_file_name = cur_file


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class ProfileMonitor(Callback):
    """
    Profile analysis in training.
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
            rank_id = int(os.getenv("RANK_ID", '0'))
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
    """Evaluate Callback used in training progress.

    Args:
        eval_func (Callable): the function to calculate eval result, task specific.
        step_interval (int): determine the num of step intervals between each eval.
            Default -1, means only eval on epoch end, do not eval between steps.
            Note that it will not take effects when running in data sink mode.
        epoch_interval (int): determine the num of epoch intervals between each eval.
            Default 1, means eval on every epoch end.
    """
    def __init__(self, eval_func: Callable, step_interval: int = -1, epoch_interval: int = 1):
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
