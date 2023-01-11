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
import os

from fm.src.aicc_tools.ailog.log import get_logger
from fm.src.aicc_tools.utils.validator import check_obs_url, check_in_modelarts, format_path, Validator
from fm.src.aicc_tools.utils.cloud_adapter import Obs2Local, mox_adapter


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
        >> import fm.src.aicc_tools as ac
        >> CFTS = ac.CFTS(obs_path='s3://aicc_test', upload_frequence=1, keep_last=False)
    """

    def __init__(self, obs_path=None, root_path='/cache', rank_id=None,
                 upload_frequence=1, keep_last=True, retry=3, retry_time=5):
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
        self.log = get_logger()
        self.keep_last = keep_last
        self.rank_id = int(os.getenv('RANK_ID', '0'))
        self.load_file = Obs2Local(self.rank_id, retry, retry_time, self.log)

        if check_in_modelarts():
            if rank_id is not None:
                Validator.check_type(rank_id, int)
                self.special_id = rank_id
                os.environ.setdefault('SPECIAL_ID', str(rank_id))
            os.environ.setdefault('OBS_PATH', obs_path)

    def get_dataset(self, dataset_path):
        """Load dataset from dataset path."""
        Validator.check_type(dataset_path, str)
        if check_in_modelarts():
            return self._pull_dataset(dataset_path=dataset_path)
        return dataset_path

    def get_checkpoint(self, checkpoint_path):
        """Load checkpoint from checkpoint path."""
        Validator.check_type(checkpoint_path, str)
        if check_in_modelarts():
            return self._pull_checkpoint(checkpoint_path=checkpoint_path)
        return checkpoint_path

    def get_strategy(self, strategy_path):
        """Load strategy file from strategy path."""
        Validator.check_type(strategy_path, str)
        if check_in_modelarts():
            return self._pull_strategy(strategy_path=strategy_path)
        return strategy_path

    def send2obs(self, src_url=None, obs_url=None):
        """Send files to obs."""
        if check_in_modelarts():
            self._send_file(src_url, obs_url)
        else:
            self.log.warning('This function(send2obs) should be used with ModelArts Platform.')

    def _pull_dataset(self, dataset_path):
        """Pull dataset."""
        if check_in_modelarts():
            check_obs_url(dataset_path)
            local_path = os.path.join(self.root, 'dataset')
            return self.load_file.obs2local(dataset_path, local_path)
        return dataset_path

    def _pull_checkpoint(self, checkpoint_path):
        """Pull checkpoint."""
        if check_in_modelarts():
            check_obs_url(checkpoint_path)
            local_path = os.path.join(self.local_path, 'checkpoint')
            return self.load_file.obs2local(checkpoint_path, local_path)
        return checkpoint_path

    def _pull_strategy(self, strategy_path):
        """Pull strategy file."""
        if check_in_modelarts():
            check_obs_url(strategy_path)
            local_path = os.path.join(self.local_path, 'strategy')
            return self.load_file.obs2local(strategy_path, local_path)
        return strategy_path

    def _send_file(self, src_url, obs_url):
        if src_url is not None and obs_url is not None:
            Validator.check_type(src_url, str)
            src_url = format_path(src_url)
            check_obs_url(obs_url)
            mox_adapter(src_url, obs_url, log=self.log)
        if src_url is not None and obs_url is None:
            Validator.check_type(src_url, str)
            src_url = format_path(src_url)
            mox_adapter(src_url, self.obs_path, log=self.log)
        if src_url is None and obs_url is None:
            if self.special_id:
                src_url = os.path.join(self.local_path, 'rank_{}'.format(self.special_id))
                mox_adapter(src_url, self.obs_path, log=self.log)
            else:
                if self.rank_id % 8 == 0:
                    src_url = os.path.join(self.local_path, 'rank_{}'.format(self.rank_id))
                    mox_adapter(src_url, self.obs_path, log=self.log)
