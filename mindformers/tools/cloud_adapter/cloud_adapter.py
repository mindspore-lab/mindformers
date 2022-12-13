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
"""Cloud Adapter."""
import os
import time

from mindspore.train.callback import Callback, ModelCheckpoint, CheckpointConfig

from ..logger import logger
from ..utils import check_obs_url, check_in_modelarts, \
    Validator, sync_trans

if check_in_modelarts():
    import moxing as mox


__all__ = ['Local2ObsMonitor', 'Obs2Local', 'CheckpointCallBack', 'mox_adapter', 'obs_register']


class Local2ObsMonitor(Callback):
    """File saved from local system to OBS server.

    Args:
        src_dir (str): Local system path, which means path of AI computing center platform.
        target_dir (str): OBS path starting with S3 or obs is used to save files.
        rank_id (int): the device's contents will be saved according to the actual rank_id.
            Default: None, means only the contents of the first device of each node are saved.
        upload_frequence (int): How often files are saved in AI computing center platform.
            Default: 1.
        keep_last (bool): Check whether files in the OBS are consistent with AI computing center platform.
            Default: True, means old file will be removed.
        retry (int): The number of attempts to save again if the first attempt fails.
            Default: 3, will be try three times.
        retry_time: The time of resaving the previously dormant program, after each attempt fails.
            Default: 5, will sleep five seconds.
        log (logger): Use the log system to print information.
            Default: logging class for Python.
    """
    def __init__(self,
                 src_dir,
                 target_dir,
                 rank_id=None,
                 upload_frequence=10,
                 keep_last=True,
                 retry=3,
                 retry_time=5,
                 log=logger):
        super(Local2ObsMonitor, self).__init__()
        self.src_dir = src_dir
        self.target_dir = target_dir
        self.upload_frequence = upload_frequence
        self.keep_last = keep_last
        self.is_special = False
        if rank_id is not None:
            self.is_special = True
            self.special_id = int(rank_id) if isinstance(rank_id, str) else rank_id
        self.rank_id = int(os.getenv('RANK_ID', '0'))
        self.retry_time = retry_time
        self.retry = retry
        self.log = log
        self.cb_params = None
        self.pro = None
        self.on_modelarts = check_in_modelarts()
        if self.on_modelarts:
            check_obs_url(target_dir)

    def step_end(self, run_context):
        """Print training loss at the end of step."""
        if self.on_modelarts:
            self.cb_params = run_context.original_args()
            if self.cb_params.cur_step_num % self.upload_frequence == 0 and os.listdir(self.src_dir):
                self.log.info("Starting upload output file to obs!")
                self.upload()

    def upload(self):
        """Upload Files to OBS."""
        if self.is_special:
            if self.rank_id == self.special_id:
                if self.pro:
                    self.pro.join()
                self.pro = self.sync2obs(self.src_dir, self.target_dir)
        else:
            if self.rank_id % 8 == 0:
                if self.pro:
                    self.pro.join()
                self.pro = self.sync2obs(self.src_dir, self.target_dir)

    @sync_trans
    def sync2obs(self, src_dir, target_dir):
        """Asynchronous transfer to OBS."""
        src_dir = os.path.join(src_dir, "rank_{}".format(self.rank_id))
        target_dir = os.path.join(target_dir, "rank_{}".format(self.rank_id))
        if self.keep_last and mox.file.exists(target_dir):
            mox.file.remove(target_dir, recursive=True)
        mox_adapter(src_dir, target_dir, self.retry, self.retry_time, self.log)


class Obs2Local:
    """File saved from OBS server to local system of AI computing center platform.

    Args:
        rank_id (int): The obs's contents will be upload according to the actual rank_id value.
            Default: 0, means stored only one OBS file each node.
        retry (int): The number of attempts to save again if the first attempt fails.
            Default: 3, will be try three times.
        retry_time: The time of resaving the previously dormant program, after each attempt fails.
            Default: 5, will sleep five seconds.
        log (logger): Use the log system to print information.
            Default: logging class for Python.
    """
    def __init__(self, rank_id=0, retry=3, retry_time=5, log=logger):
        self.rank_id = int(rank_id)
        self.retry_time = retry_time
        self.retry = retry
        self.log = log

    def obs2local(self, obs_url, local_url, special_id=None):
        """Pull Obs Files to Local."""
        obs_name = obs_url.split("/")[-1]
        mox_lock = os.path.join(local_url, "mox_copy_{}.lock".format(obs_name))
        local_url = os.path.join(local_url, obs_name)
        if special_id is None:
            if self.rank_id % 8 == 0:
                mox_adapter(obs_url, local_url, self.retry, self.retry_time, self.log)
                try:
                    os.mknod(mox_lock)
                except IOError:
                    pass
            else:
                self.log.info("programming sleep for waiting download file from obs to local.")
                while True:
                    if os.path.exists(mox_lock):
                        break
                    time.sleep(1)
        else:
            Validator.check_type(special_id, int)
            if self.rank_id == special_id:
                mox_adapter(obs_url, local_url, self.retry, self.retry_time, self.log)
                try:
                    os.mknod(mox_lock)
                except IOError:
                    pass
            else:
                self.log.info("programming sleep for waiting download file from obs to local.")
                while True:
                    if os.path.exists(mox_lock):
                        break
                    time.sleep(1)
        return local_url


class CheckpointCallBack:
    """
    Args:
        prefix (str): The prefix name of checkpoint files. Default: "CKP".
        directory (str): The path of the folder which will be saved in the checkpoint file.
            By default, the file is saved in the current directory. Default: None.
        config (CheckpointConfig): Checkpoint strategy configuration. Default: None.
        save_checkpoint_steps (int): Steps to save checkpoint. Default: 1.
        save_checkpoint_seconds (int): Seconds to save checkpoint.
            Can't be used with save_checkpoint_steps at the same time. Default: 0.
        keep_checkpoint_max (int): Maximum number of checkpoint files can be saved. Default: 5.
        keep_checkpoint_per_n_minutes (int): Save the checkpoint file every `keep_checkpoint_per_n_minutes` minutes.
            Can't be used with keep_checkpoint_max at the same time. Default: 0.
        integrated_save (bool): Whether to merge and save the split Tensor in the automatic parallel scenario.
            Integrated save function is only supported in automatic parallel scene, not supported
            in manual parallel. Default: True.
        async_save (bool): Whether asynchronous execution saves the checkpoint to a file. Default: False.
        saved_network (Cell): Network to be saved in checkpoint file. If the saved_network has no relation
            with the network in training, the initial value of saved_network will be saved. Default: None.
        append_info (list): The information save to checkpoint file. Support "epoch_num", "step_num" and dict.
            The key of dict must be str, the value of dict must be one of int float and bool. Default: None.
        enc_key (Union[None, bytes]): Byte type key used for encryption. If the value is None, the encryption
                                      is not required. Default: None.
        enc_mode (str): This parameter is valid only when enc_key is not set to None. Specifies the encryption
                        mode, currently supports 'AES-GCM' and 'AES-CBC'. Default: 'AES-GCM'.
        exception_save (bool): Whether to save the current checkpoint when an exception occurs. Default: False.

    Raises:
        ValueError: If input parameter is not the correct type.
        ValueError: If the prefix is invalid.
        TypeError: If the config is not CheckpointConfig type.
    """
    def __init__(self,
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
                 exception_save=False
                 ):
        self.prefix = prefix
        self.directory = directory
        self.config = config
        self.save_checkpoint_steps = save_checkpoint_steps
        self.save_checkpoint_seconds = save_checkpoint_seconds
        self.keep_checkpoint_max = keep_checkpoint_max
        self.keep_checkpoint_per_n_minutes = keep_checkpoint_per_n_minutes
        self.integrated_save = integrated_save
        self.async_save = async_save
        self.saved_network = saved_network
        self.append_info = append_info
        self.enc_key = enc_key
        self.enc_mode = enc_mode
        self.exception_save = exception_save

    def save_checkpoint(self):
        """Save Checkpoint."""
        config_ck = CheckpointConfig(save_checkpoint_steps=self.save_checkpoint_steps,
                                     save_checkpoint_seconds=self.save_checkpoint_seconds,
                                     keep_checkpoint_max=self.keep_checkpoint_max,
                                     keep_checkpoint_per_n_minutes=self.keep_checkpoint_per_n_minutes,
                                     integrated_save=self.integrated_save,
                                     async_save=self.async_save,
                                     saved_network=self.saved_network,
                                     append_info=self.append_info,
                                     enc_key=self.enc_key,
                                     enc_mode=self.enc_mode,
                                     exception_save=self.exception_save)
        ckpoint_cb = ModelCheckpoint(prefix=self.prefix,
                                     directory=self.directory,
                                     config=config_ck)
        return ckpoint_cb


def mox_adapter(src_dir, target_dir, retry=3, retry_time=5, log=logger):
    """File interaction with Moxing."""
    success = False
    for i in range(retry + 1):
        start = time.time()
        try:
            mox.file.copy_parallel(src_url=src_dir, dst_url=target_dir)
        except (FileNotFoundError, RuntimeError) as e:
            log.info("%s, from %s download to %s failed, will retry(%d) again.",
                     e, src_dir, target_dir, i)
            # sleep due to restriction of obs
            log.info("sleep time %d for waiting download file from obs.", retry_time)
            continue
        end = time.time()
        if Validator.is_obs_url(target_dir):
            if mox.file.exists(target_dir):
                success = True
                log.info("Pull/Push file %s success, cost time: %f", target_dir, end - start)
                break
        else:
            if os.path.exists(target_dir):
                success = True
                log.info("Pull/Push file %s success, cost time: %f", target_dir, end - start)
                break
    return success


def obs_register(ak=None, sk=None, server=None):
    """OBS register with Moxing."""
    if check_in_modelarts():
        os.environ.pop('CREDENTIAL_PROFILES_FILE', None)
        os.environ.pop('AWS_SHARED_CREDENTIALS_FILE', None)
        mox.file.set_auth(ak=ak, sk=sk, server=server)
