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
from typing import Optional, Union

import numpy as np
import mindspore as ms
from mindspore import Callback, Profiler
from mindspore.train.callback import SummaryCollector
from mindspore.nn.learning_rate_schedule import LearningRateSchedule

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.cloud_adapter.cloud_adapter import Local2ObsMonitor, CheckpointCallBack
from mindformers.tools.logger import logger
from mindformers.tools.utils import LOCAL_DEFAULT_PATH


__all__ = ['ObsMonitor', 'MFLossMonitor', 'CheckpointMointor', 'SummaryMonitor', 'ProfileMonitor']


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class ObsMonitor:
    """Obs Monitor For AICC and Local"""
    def __new__(cls,
                src_dir: str = None,
                target_dir: str = None,
                rank_id: int = None,
                upload_frequence: int = 1,
                keep_last: bool = True):
        is_cfts = MindFormerRegister.is_exist(
            module_type=MindFormerModuleType.TOOLS, class_name="cfts")
        if is_cfts:
            cfts = MindFormerRegister.get_cls(
                class_name="cfts", module_type=MindFormerModuleType.TOOLS)
            return cfts.obs_monitor()
        return Local2ObsMonitor(src_dir, target_dir, rank_id, upload_frequence, keep_last)


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class MFLossMonitor(Callback):
    """
    Loss Monitor for classification.

    Args:
        learning_rate (Union[float, LearningRateSchedule], optional): The learning rate schedule. Default: None.
        per_print_times (int): Every how many steps to print the log information. Default: 1.

    Examples:
        >>> from mindformers.common.callback import MFLossMonitor
        >>> lr = [0.01, 0.008, 0.006, 0.005, 0.002]
        >>> monitor = MFLossMonitor(per_print_times=10)
    """

    def __init__(self,
                 learning_rate: Optional[Union[float, LearningRateSchedule]] = None,
                 per_print_times: int = 1):
        super(MFLossMonitor, self).__init__()
        self.per_print_times = per_print_times
        self.learning_rate = deepcopy(learning_rate)
        self.last_print_time = 0
        self.print_warning_flag = True
        self.loss_list = []
        self.step_time = time.time()
        self.epoch_time = time.time()

    # pylint: disable=unused-argument
    def epoch_begin(self, run_context):
        """
        Record time at the beginning of epoch.

        Args:
            run_context (RunContext): Context of the process running.
        """
        self.loss_list = []
        self.epoch_time = time.time()

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

    # pylint: disable=unused-argument
    def step_begin(self, run_context):
        """
        Record time at the beginning of step.

        Args:
            run_context (RunContext): Context of the process running.
        """
        self.step_time = time.time()
        self.print_warning_flag = True

    # pylint: disable=missing-docstring
    def step_end(self, run_context):
        """
        Print training info at the end of step.

        Args:
            run_context (RunContext): Context of the process running.
        """
        cb_params = run_context.original_args()
        step_mseconds = (time.time() - self.step_time) * 1000
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], ms.Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, ms.Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        self.loss_list.append(loss)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        # Boundary check.
        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("Invalid loss, terminate training.")

        def print_output_info():
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
                        current_lr = 'CPU Mode Not Support Compute LR Now.'
                    else:
                        current_step = ms.Tensor(cb_params.cur_step_num - 1, ms.int32)
                        current_lr = self.learning_rate(current_step)
                        current_lr = np.array2string(current_lr.asnumpy())
                else:
                    current_lr = "Not support LR %s type compute!" % type(self.learning_rate)
            else:
                current_lr = 'Not Set LR.'
            logger.info(
                "Epoch:[%3d/%3d], step:[%5d/%5d], "
                "loss:[%5.3f/%5.3f], time:%5.3f ms, "
                "lr:%s", cb_params.cur_epoch_num - 1, cb_params.epoch_num,
                cur_step_in_epoch, cb_params.batch_num, loss, np.mean(self.loss_list),
                step_mseconds, current_lr)

        if (cb_params.cur_step_num - self.last_print_time) >= self.per_print_times:
            self.last_print_time = cb_params.cur_step_num
            print_output_info()


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
            rank_id = os.getenv("RANK_ID", "0")
            summary_dir = os.path.join(
                LOCAL_DEFAULT_PATH, 'rank_{}'.format(rank_id), 'summary')
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
        is_cfts = MindFormerRegister.is_exist(
            module_type=MindFormerModuleType.TOOLS, class_name="cfts")
        if is_cfts:
            cfts = MindFormerRegister.get_cls(
                class_name="cfts", module_type=MindFormerModuleType.TOOLS)
            return cfts.summary_monitor(**kwargs)
        return SummaryCollector(**kwargs)


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class CheckpointMointor:
    """Checkpoint Monitor For AICC and Local"""
    def __new__(cls,
                prefix='CKP',
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

        rank_id = int(os.getenv("DEVICE_ID", '0'))
        prefix = prefix + "_rank_{}".format(rank_id)

        kwargs = {
            "prefix": prefix,
            "directory": directory,
            "config": config,
            "save_checkpoint_steps": save_checkpoint_steps,
            "save_checkpoint_seconds": save_checkpoint_seconds,
            "keep_checkpoint_max": keep_checkpoint_max,
            "keep_checkpoint_per_n_minutes": keep_checkpoint_per_n_minutes,
            "integrated_save": integrated_save,
            "async_save": async_save,
            "saved_network": saved_network,
            "append_info": append_info,
            "enc_key": enc_key,
            "enc_mode": enc_mode,
            "exception_save": exception_save
        }
        is_cfts = MindFormerRegister.is_exist(
            module_type=MindFormerModuleType.TOOLS, class_name="cfts")
        if is_cfts:
            cfts = MindFormerRegister.get_cls(
                class_name="cfts", module_type=MindFormerModuleType.TOOLS)
            return cfts.checkpoint_monitor(**kwargs)
        checkpoint_cb = CheckpointCallBack(**kwargs)
        return checkpoint_cb.save_checkpoint()


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class ProfileMonitor(Callback):
    """
    Profile analysis in training.
    """
    def __init__(self, start_step=1, stop_step=10, output_path=None, profile_communication=False):
        super(ProfileMonitor, self).__init__()
        self.start_step = start_step
        self.stop_step = stop_step
        if output_path is not None:
            assert isinstance(output_path, str) and os.path.realpath(output_path), \
                f"output path must be real path, but get {output_path}"
            self.profiler = Profiler(
                start_profile=False, output_path=output_path, profile_communication=profile_communication)
        else:
            self.profiler = Profiler(start_profile=False)
        self.run_context = None

    def step_begin(self, run_context):
        """
        Start profile at the begin of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.start_step:
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
