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

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class ObsMonitor:
    """Obs Monitor For AICC and Local"""
    def __new__(cls):
        cfts = MindFormerRegister.get_cls(
            class_name="cfts", module_type=MindFormerModuleType.COMMON)
        return cfts.obs_monitor()


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class MFLossMonitor:
    """Loss Monitor For AICC and Local"""
    def __new__(cls, per_print_times=1):
        cfts = MindFormerRegister.get_cls(
            class_name="cfts", module_type=MindFormerModuleType.COMMON)

        return cfts.loss_monitor(per_print_times)


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
        cfts = MindFormerRegister.get_cls(
            class_name="cfts", module_type=MindFormerModuleType.COMMON)
        return cfts.summary_monitor(**kwargs)


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
        cfts = MindFormerRegister.get_cls(
            class_name="cfts", module_type=MindFormerModuleType.COMMON)

        return cfts.checkpoint_monitor(**kwargs)
