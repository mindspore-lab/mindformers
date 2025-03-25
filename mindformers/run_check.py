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
"""run_check module"""
import os
import platform
import re
import subprocess
import sys
import time
import json
from io import StringIO
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore import JitConfig
from mindspore.dataset import GeneratorDataset

import mindformers as mf
from mindformers.models.llama import LlamaForCausalLM, LlamaConfig
from mindformers.trainer.optimizer_grouped_parameters import get_optimizer_grouped_parameters
from mindformers import Trainer, TrainingArguments, AdamW
from mindformers.tools.logger import logger


class BaseCheck:
    """The base check class, needs to be implemented."""

    def __init__(self, start, version_mapping=None):
        """
        Init function.

        Args:
            start (float): The start time of run_check.
            version_mapping (dict): The version compatibility dict. Default is None.
        """
        self.start = start
        self.version_mapping = version_mapping

    def set_next(self, next_check):
        """
        Set next check.

        Args:
            next_check : The instance of next check.
        """
        self.next_check = next_check

    def check(self):
        """The actual checking flow control."""

    def _next(self):
        self.next_check.check()

    def _error(self, **kwargs):
        pass

    def _success(self, **kwargs):
        pass


class MSCheck(BaseCheck):
    """Test MindSpore's run_check API."""

    def check(self):
        """MindSpore check"""
        logger.info('------------------------------Starting MindSpore Run Check------------------------------')
        buffer = StringIO()
        sys.stdout = buffer

        ms.run_check()
        result = buffer.getvalue()

        sys.stdout = sys.__stdout__
        if re.search('failed', result):
            self._error(result=result)
            version_checker = VersionCheck(self.start, error_flag='MS')
            version_checker.check()
        else:
            self._success(result=result.split('\n')[1])
            self._next()

    def _error(self, **kwargs):
        logger.error('MindSpore failed!')
        logger.error(kwargs.get('result', None))
        logger.info('The MindSpore is not installed correctly, please refer to https://www.mindspore.cn/install/')

    def _success(self, **kwargs):
        logger.info(kwargs.get('result', None))


class MFCheck(BaseCheck):
    """Mindformers run test."""

    def __init__(self, start, version_mapping=None, batch_size=1, num_train_epochs=1, num_layers=2, seq_length=2,
                 step_num=1, vocab_size=32000):
        """
        init function.

        Args:
            start (float): The start time of run_check.
            version_mapping (dict): The version compatibility dict. Default is None.
            batch_size (int): The batch size.
            num_train_epochs (int): The train epochs.
            num_layers (int): The number of layers.
            seq_length (int): The sequence length.
            step_num (int): The step number.
            vocab_size (int): The vocab size.
        """
        super().__init__(start, version_mapping)

        self.step_num = step_num
        self.vocab_size = vocab_size
        self.args = TrainingArguments(batch_size=batch_size, num_train_epochs=num_train_epochs, sink_mode=False,
                                      loss_scale_value=1024)
        self.model_config = LlamaConfig(num_layers=num_layers, seq_length=seq_length, use_flash_attention=True)

    def generator_train(self):
        """train dataset generator."""
        size = (self.step_num * self.args.batch_size, self.model_config.seq_length + 1,)
        input_ids = np.random.randint(low=0, high=self.vocab_size, size=size).astype(np.int32)
        for _, input_id in enumerate(input_ids):
            yield input_id

    def _train(self, jit_level='O1'):
        """
        Construct trainer and start training.

        Args:
            jit_level (str): The jit level, could be O0, O1 or O2.
        """
        train_dataset = GeneratorDataset(self.generator_train, column_names=["input_ids"])
        self.train_dataset = train_dataset.batch(batch_size=self.args.batch_size)
        self.model = LlamaForCausalLM(self.model_config)
        self.model.set_jit_config(JitConfig(jit_level=jit_level))
        group_params = get_optimizer_grouped_parameters(model=self.model)
        trainer = Trainer(task='text_generation',
                          model=self.model,
                          args=self.args,
                          train_dataset=self.train_dataset,
                          callbacks=None,
                          optimizers=AdamW(params=group_params))
        trainer.config.callbacks = trainer.config.callbacks[:1]
        trainer.train()

    def check(self):
        """Run mindformers test."""
        logger.info('------------------------------Starting Pretrain Test------------------------------')
        try:
            self._train(jit_level='O1')

        # pylint: disable=W0702
        except Exception as e:
            self._error(error_flag='Pretrain')
            version_checker = VersionCheck(self.start, error_flag='Pretrain')
            version_checker.check()
            raise RuntimeError("The run check failed, please see more information above. "
                               "Exception is {}".format(e))

        else:
            self._success(test='Pretrain')

        logger.info('------------------------------Starting Predict Test------------------------------')
        try:
            self.model.generate([1], do_sample=False, use_past=False)
            self.model.generate([2], do_sample=False, use_past=True)

        # pylint: disable=W0702
        except:
            self._error(error_flag='Predict')
            version_checker = VersionCheck(self.start, error_flag='Predict')
            version_checker.check()

        else:
            self._success(test='Predict')
            self._next()

    def _error(self, **kwargs):
        sys.stdout = sys.__stdout__
        error_flag = kwargs.get('error_flag', None)
        logger.error(f'{error_flag} test failed!', exc_info=True)
        logger.info('If you need any help, please open an issue in the MindFormers repository: '
                    'https://gitee.com/mindspore/mindformers/issues')

    def _success(self, **kwargs):
        test = kwargs.get('test', None)
        logger.info(f'{test} test passed!')


class VersionCheck(BaseCheck):
    """Version check"""

    def __init__(self, start, version_mapping=None, error_flag=None):
        super().__init__(start, version_mapping)
        self.error_flag = error_flag

    @staticmethod
    def get_recommend_versions(mf_version=None, version_mapping=None):
        """
        Check whether the given mf_version is in record.

        Args:
            mf_version (str): The mf_version needs to be searched. Default is None.
            version_mapping (dict): The version mapping dict. Default is None.

        Returns:
            dict:
                - is_mf_version_in_record (bool): Whether the MindFormers version is in record.
                - mf_version (str): The latest MindFormers version or the input if the version is in record.
                - ms_version (str): The corresponding MindSpore version.
                - cann_version (str): The corresponding cann version.
                - driver_version (str): The corresponding driver version.
        """
        matched_vs = version_mapping['mf'].get(mf_version, None)
        if matched_vs:
            is_mf_version_in_record = True
        else:
            is_mf_version_in_record = False
            mf_version = max(version_mapping['mf'].keys(), key=lambda x: tuple(map(int, x.split('.'))))
        ms_version = version_mapping['mf'][mf_version]['prefer']
        cann_version = version_mapping['ms'][ms_version]['prefer']
        driver_version = version_mapping['cann'][cann_version]['prefer']
        return {'is_mf_version_in_record': is_mf_version_in_record,
                'mf_version': mf_version,
                'ms_version': ms_version,
                'cann_version': cann_version,
                'driver_version': driver_version}

    def _print_matched_logs(self, end):
        """
        The installed versions are matched.

        Args:
            end (float): The end time of run_check.
        """
        if self.error_flag:
            logger.error(f'The run check failed in {end - self.start:.2f} seconds, '
                         f'please check the above info for more details')
        else:
            logger.info(
                f'All checks passed, used {end - self.start:.2f} seconds, the environment is correctly set up!')

    def _print_unmatched_logs(self, end, mf_version, ms_version, cann_version, driver_version):
        """
        The installed versions are unmatched.

        Args:
            end (float): The end time of run_check.
            mf_version (str): The recommended mindformers version.
            ms_version (str): The recommended mindspore version.
            cann_version (str): The recommended cann-toolkit version.
            driver_version (str): The recommended driver version.
        """
        if self.error_flag:
            logger.error(f"The run check failed in {end - self.start:.2f} "
                         f"seconds, It's recommended to install cann-toolkit=={cann_version} "
                         f"driver=={driver_version} mindspore=={ms_version} mindformers=={mf_version}")
        else:
            logger.warning(f'The installed software are unmatched '
                           f'but all checks passed in {end - self.start:.2f} seconds')
            logger.info(f"It's recommended to install cann-toolkit=={cann_version} "
                        f"driver=={driver_version} mindspore=={ms_version} mindformers=={mf_version}")

    def check(self):
        """Check whether the software versions are matched."""
        logger.info('------------------------------Searching Environment Info------------------------------')
        try:
            mf_version = mf.__version__
            ms_version = ms.__version__
            logger.info(f'MindFormers version: {mf_version}')
            logger.info(f'MindSpore version: {ms_version}')

            # one of ['aarch64', 'x86_64']
            arch = platform.machine()

            cann_pth = os.getenv('ASCEND_HOME_PATH')
            if not cann_pth:
                cann_pth = Path('/usr/local/Ascend/ascend-toolkit/latest')

            cann_info_file = f'{str(cann_pth)}/{arch}-linux/ascend_toolkit_install.info'
            driver_info_file = '/usr/local/Ascend/driver/version.info'
            is_cann_info_file_exist = Path(cann_info_file).exists()
            is_driver_info_file_exist = Path(driver_info_file).exists()

            if is_cann_info_file_exist:
                cann_info = subprocess.run(["cat", cann_info_file], shell=False, capture_output=True,
                                           check=True).stdout.decode('utf-8')
                cann_version = re.search(r'=(\d[\d.\w]+)', cann_info).group(1)
                logger.info(f'Ascend-cann-toolkit version: {cann_version}')
            else:
                logger.warning('The environment variable "ASCEND_HOME_PATH" is not set, please check whether the CANN '
                               'Toolkit is installed. Try to execute "source ${HOME}/Ascend/ascend-toolkit/set_env.sh".'
                               ' For installation manual, please refer to '
                               'https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit')

            if is_driver_info_file_exist:
                driver_info = subprocess.run(["cat", driver_info_file], shell=False, capture_output=True,
                                             check=True).stdout.decode('utf-8')
                driver_version = re.search(r'=(\d[\d.\w]+)', driver_info).group(1)
                logger.info(f'Ascend driver version: {driver_version}')
            else:
                logger.warning(f'Cannot find driver info file under default path, please check whether the driver is '
                               f'installed properly. Please run infer to https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit')

            if not is_cann_info_file_exist or not is_driver_info_file_exist:
                raise RuntimeError

        # version unmatched
        except RuntimeError:
            end = time.perf_counter()
            recommend_version_dict = self.get_recommend_versions(mf_version, self.version_mapping)
            self._print_unmatched_logs(
                end,
                recommend_version_dict['mf_version'],
                recommend_version_dict['ms_version'],
                recommend_version_dict['cann_version'],
                recommend_version_dict['driver_version']
            )

        else:
            end = time.perf_counter()
            recommend_version_dict = self.get_recommend_versions(mf_version, self.version_mapping)

            # version matched
            if (recommend_version_dict['is_mf_version_in_record']
                    and (ms_version == recommend_version_dict['ms_version']
                         or ms_version in self.version_mapping['mf'][mf_version]['support'])
                    and cann_version == self.version_mapping['ms'][ms_version]['prefer']
                    and driver_version == self.version_mapping['cann'][cann_version]['prefer']):
                self._print_matched_logs(end)

            # version unmatched
            else:
                self._print_unmatched_logs(
                    end,
                    recommend_version_dict['mf_version'],
                    recommend_version_dict['ms_version'],
                    recommend_version_dict['cann_version'],
                    recommend_version_dict['driver_version']
                )


def run_check():
    """
    Check whether the installed CANN, driver, MindSpore and MindFormers versions are matched.
    The structure of the VERSION_MAP.json is shown as below:
    {
        'mf': {
            'version1': {
                'prefer': 'prefered ms version',
                'support': [competible ms version list]
            },
        },
        'ms': {
            'version1': {
                'prefer' : 'prefered cann version',
                'support': [competible cann version list]
            },
        },
        'cann': {
            'version1': {
                'prefer' : 'prefered driver version',
                'support': [competible driver version list]
            },
        }
    }

    Examples:
        >>> from mindformers import run_check
        >>> run_check()
    """

    version_file = Path(f'{__file__}').parent / 'VERSION_MAP.json'
    if not version_file.is_file():
        raise RuntimeError('Cannot find VERSION_MAP.json or the found one is not a file')

    with open(version_file) as f:
        version_mapping = json.load(f)

    os.environ['MS_ALLOC_CONF'] = "enable_vmm:False"

    start = time.perf_counter()
    ms.set_context(mode=0)

    ms_checker = MSCheck(start, version_mapping=version_mapping)
    mf_checker = MFCheck(start, version_mapping=version_mapping)
    version_checker = VersionCheck(start, version_mapping=version_mapping)

    ms_checker.set_next(mf_checker)
    mf_checker.set_next(version_checker)

    ms_checker.check()
