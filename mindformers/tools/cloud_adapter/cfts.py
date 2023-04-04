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
"""CFTS"""

import os
import pathlib

from mindspore.train.callback import SummaryCollector

from mindformers.core.callback import ProfileMonitor
from mindformers.tools.logger import logger
from ..utils import LOCAL_DEFAULT_PATH, PROFILE_INFO_PATH,\
    check_obs_url, check_in_modelarts, format_path, Validator
from .cloud_adapter import Obs2Local, Local2ObsMonitor, mox_adapter,\
    CheckpointCallBack


class CFTS:
    """File interaction interface for AI computing centers.

    Args:
        obs_path (str or None): A path starting with S3 or obs is used to save files in AI computing center.
            Default: None, which means not AI computing center platform.
        root_path (str): Path of the AI computing center cluster container.
            Default: '/cache', All files will be saved in there..
        rank_id (int): If you specify its value, the device's contents will be saved according to the actual rank_id.
            Default: None, means only the contents of the first device of each node are saved.
        upload_frequence (int): How often files are saved in AI computing center platform.
            Default: 1.
        keep_last (bool): Check whether files in the OBS are consistent with AI computing center platform.
            Default: True, means old file will be removed.
        retry (int): The number of attempts to save again if the first attempt fails.
            Default: 3, will be try three times.
        retry_time: The time of resaving the previously dormant program, after each attempt fails.
            Default: 5, will sleep five seconds.

    Examples:
        >>> cfts = CFTS(obs_path="s3://aicc_test", upload_frequence=1, keep_last=False)
    """
    def __init__(self,
                 obs_path=None,
                 root_path='/cache',
                 rank_id=None,
                 upload_frequence=1,
                 keep_last=True,
                 log=logger,
                 retry=3,
                 retry_time=5):
        Validator.check_type(upload_frequence, int)
        Validator.check_type(retry, int)
        Validator.check_type(retry_time, int)
        Validator.check_type(keep_last, bool)
        Validator.check_type(root_path, str)
        root_path = format_path(root_path)

        self.root = root_path
        self.obs_path = obs_path
        self.local_path = os.path.join(root_path, 'ma-user-work')
        self.checkpoint_path = None
        self.strategy_path = None
        self.dataset_path = None
        self.special_id = None
        self.log = log
        self.keep_last = keep_last
        self.rank_id = int(os.getenv('RANK_ID', '0'))
        self.device_num = int(os.getenv('DEVICE_NUM', '1'))
        self.load_file = Obs2Local(self.rank_id, retry, retry_time, self.log)
        self.save_cb = Local2ObsMonitor(
            self.local_path, obs_path, rank_id, upload_frequence, keep_last, retry, retry_time, self.log)

        if rank_id is not None:
            Validator.check_type(rank_id, int)
            self.special_id = rank_id
            os.environ.setdefault('SPECIAL_ID', str(rank_id))

        if check_in_modelarts():
            os.environ.setdefault('OBS_PATH', obs_path)

    @property
    def get_log_system(self):
        """Log system for AI computing center platform."""
        return self.log

    @property
    def get_custom_path(self, directory='custom', file_name=None):
        """Save path for custom monitor function."""
        Validator.check_type(directory, str)
        if check_in_modelarts():
            return self._generate_cfts_path(directory, file_name)
        self.log.warning("This function(get_custom_path) should be used with ModelArts Platform.")
        return self.local_path

    def get_dataset(self, dataset_path, rank_id=None):
        """Load dataset from dataset path."""
        Validator.check_type(dataset_path, str)
        if check_in_modelarts():
            return self._pull_dataset(dataset_path=dataset_path, rank_id=rank_id)
        return dataset_path

    def get_checkpoint(self, checkpoint_path, rank_id=None):
        """Load checkpoint from checkpoint path."""
        Validator.check_type(checkpoint_path, str)
        if check_in_modelarts():
            return self._pull_checkpoint(checkpoint_path=checkpoint_path, rank_id=rank_id)
        return checkpoint_path

    def get_strategy(self, strategy_path, rank_id=None):
        """Load strategy file from strategy path."""
        Validator.check_type(strategy_path, str)
        if check_in_modelarts():
            return self._pull_strategy(strategy_path=strategy_path, rank_id=rank_id)
        return strategy_path

    def summary_monitor(self, summary_dir=None, **kwargs):
        """Record summary information in training."""
        if check_in_modelarts():
            summary_dir = os.path.join(self.local_path, 'rank_{}'.format(self.rank_id))
            summary_dir = os.path.join(summary_dir, 'summary')
        elif summary_dir is None:
            summary_dir = os.path.join(LOCAL_DEFAULT_PATH, 'rank_{}'.format(self.rank_id))
            summary_dir = os.path.join(summary_dir, 'summary')
        Validator.check_type(summary_dir, str)
        format_path(summary_dir)
        return SummaryCollector(summary_dir, **kwargs)

    def checkpoint_monitor(self, directory=None, prefix='CKP', **kwargs):
        """Save checkpoint in training for network."""
        if check_in_modelarts():
            directory = os.path.join(self.local_path, 'rank_{}'.format(self.rank_id))
            directory = os.path.join(directory, 'checkpoint')
        elif directory is None:
            directory = os.path.join(LOCAL_DEFAULT_PATH, 'rank_{}'.format(self.rank_id))
            directory = os.path.join(directory, 'checkpoint')
        Validator.check_type(directory, str)
        format_path(directory)
        ckpt_cb = CheckpointCallBack(prefix=prefix, directory=directory, **kwargs)
        return ckpt_cb.save_checkpoint()

    def profile_monitor(self, start_step=1, stop_step=10,
                        start_profile=True, profile_communication=False,
                        profile_memory=True, **kwargs):
        """Profile Monitor."""
        if check_in_modelarts():
            output_path = os.path.join(PROFILE_INFO_PATH)
        else:
            output_path = os.path.join(LOCAL_DEFAULT_PATH, 'profile')
        format_path(output_path)
        if self.device_num > 1:
            logger.info("Device number is %s > 1, so profile_communication and start_profile will be set True ")
            profile_communication = True
            start_profile = True
        profile_cb = ProfileMonitor(
            start_step, stop_step, start_profile=start_profile,
            output_path=output_path, profile_communication=profile_communication,
            profile_memory=profile_memory, **kwargs)
        return profile_cb

    def obs_monitor(self):
        """Save all files in training to OBS path."""
        return self.save_cb

    def send2obs(self, src_url=None, obs_url=None):
        """Send files to obs."""
        if check_in_modelarts():
            self._send_file(src_url, obs_url)
        else:
            self.log.warning("This function(send2obs) should be used with ModelArts Platform.")

    def _pull_dataset(self, dataset_path, rank_id):
        """Pull dataset."""
        if check_in_modelarts():
            check_obs_url(dataset_path)
            local_path = os.path.join(self.root, 'dataset')
            return self.load_file.obs2local(dataset_path, local_path, rank_id)
        return dataset_path

    def _pull_checkpoint(self, checkpoint_path, rank_id):
        """Pull checkpoint."""
        if check_in_modelarts():
            check_obs_url(checkpoint_path)
            local_path = os.path.join(self.root, 'checkpoint')
            return self.load_file.obs2local(checkpoint_path, local_path, rank_id)
        return checkpoint_path

    def _pull_strategy(self, strategy_path, rank_id):
        """Pull strategy file."""
        if check_in_modelarts():
            check_obs_url(strategy_path)
            local_path = os.path.join(self.root, 'strategy')
            return self.load_file.obs2local(strategy_path, local_path, rank_id)
        return strategy_path

    def _generate_cfts_path(self, directory='custom', file_name=None):
        """Generate cfts save path."""
        if self.special_id is not None:
            cfts_path = os.path.join(
                self.local_path, 'rank_{}'.format(self.special_id), directory)

            pathlib.Path(cfts_path).mkdir(parents=True, exist_ok=True)
            cfts_path = os.path.join(cfts_path, file_name) if file_name is not None else cfts_path
        else:
            cfts_path = os.path.join(
                self.local_path, 'rank_{}'.format(self.rank_id), directory
            )
            pathlib.Path(cfts_path).mkdir(parents=True, exist_ok=True)
            cfts_path = os.path.join(cfts_path, file_name) if file_name is not None else cfts_path
        return cfts_path

    def _send_file(self, src_url, obs_url):
        """Send File each obs with local."""
        if src_url is not None and obs_url is not None:
            Validator.check_type(src_url, str)
            src_url = format_path(src_url)
            check_obs_url(obs_url)
            if self.special_id is not None:
                mox_adapter(src_url, obs_url, log=self.log)
            else:
                if self.rank_id % 8 == 0:
                    mox_adapter(src_url, obs_url, log=self.log)
        if src_url is not None and obs_url is None:
            Validator.check_type(src_url, str)
            src_url = format_path(src_url)
            final = src_url.split('/')[-1]
            if self.special_id is not None:
                target_dir = os.path.join(self.obs_path, 'rank_{}'.format(self.special_id), final)
                mox_adapter(src_url, target_dir, log=self.log)
            else:
                if self.rank_id % 8 == 0:
                    target_dir = os.path.join(self.obs_path, 'rank_{}'.format(self.rank_id), final)
                    mox_adapter(src_url, target_dir, log=self.log)
        if src_url is None and obs_url is None:
            if self.special_id is not None:
                src_url = os.path.join(self.local_path, 'rank_{}'.format(self.special_id))
                target_dir = os.path.join(self.obs_path, 'rank_{}'.format(self.special_id))
                mox_adapter(src_url, target_dir, log=self.log)
            else:
                if self.rank_id % 8 == 0:
                    src_url = os.path.join(self.local_path, 'rank_{}'.format(self.rank_id))
                    target_dir = os.path.join(self.obs_path, 'rank_{}'.format(self.rank_id))
                    mox_adapter(src_url, target_dir, log=self.log)
