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
import re
import sys
import time
import signal
import platform
import subprocess
from io import StringIO
from pathlib import Path
import numpy as np

import mindformers as mf
from mindformers import Trainer, TrainingArguments, AdamW
from mindformers.models.llama import LlamaForCausalLM, LlamaConfig
from mindformers.trainer.optimizer_grouped_parameters import get_optimizer_grouped_parameters
import mindspore as ms
from mindspore import JitConfig
from mindspore.dataset import GeneratorDataset
from .tools.logger import logger, get_logger, validate_level

VERSION_MAPPING = {
    '1.2.0': {
        'ms': '2.3.0',
        'cann': '8.0.RC2',
        'driver': '24.1.rc2',
    },
}


class BaseCheck:
    """The base check class, needs to be implemented"""

    def __init__(self, start, log_level='INFO'):
        """
        Init function

        Args:
            start (float): The start time of run_check
            log_level (str): The log level
        """
        self.start = start
        self.log_level = log_level.upper()
        validate_level('log_level', self.log_level)
        # pylint: disable=W0212
        self.ms_level = ms.log._get_logger().handlers[0].level
        self.mf_level = get_logger().handlers[0].level

    def _set_log_level(self, reset=False):
        """
        Set_log_level

        Args:
            reset (bool): If it is False, set the log level to specified level,
                          If it is True, set the log level back
        """
        if reset:
            ms_level = self.ms_level
            mf_level = self.mf_level
        else:
            ms_level = self.log_level
            mf_level = self.log_level

        # pylint: disable=W0212
        ms.log._get_logger().handlers[0].setLevel(ms_level)
        get_logger().handlers[0].setLevel(mf_level)

    def set_next(self, next_check):
        """
        Set next check

        Args:
            next_check : The instance of next check
        """
        self.next_check = next_check

    def check(self):
        pass

    def _next(self):
        self.next_check.check()

    def _error(self, **kwargs):
        pass

    def _success(self, **kwargs):
        pass


class MSCheck(BaseCheck):
    """Test MindSpore's run_check API"""

    def check(self):
        """MindSpore check"""
        logger.info('------------------------------Starting MindSpore Run Check------------------------------')
        self._set_log_level()
        buffer = StringIO()
        sys.stdout = buffer

        ms.run_check()
        result = buffer.getvalue()

        sys.stdout = sys.__stdout__
        self._set_log_level(reset=True)
        if re.search('failed', result):
            self._error(result=result)
            vc = VersionCheck(self.start, error_flag='MS')
            vc.check()
        else:
            self._success(result=result.split('\n')[1])
            self._next()

    def _error(self, **kwargs):
        logger.error('MindSpore failed!')
        logger.error(kwargs['result'])
        logger.info('The MindSpore is not installed correctly, please refer to https://www.mindspore.cn/install/')

    def _success(self, **kwargs):
        logger.info(kwargs['result'])


class MFCheck(BaseCheck):
    """Mindformers run test"""

    def __init__(self, start, log_level='INFO', batch_size=1, num_train_epochs=1, num_layers=2, seq_length=2,
                 step_num=1, vocab_size=32000):
        """
        init function.

        Args:
            start (float): The start time of run_check
            log_level (str): The log level
            batch_size (int): The batch size
            num_train_epochs (int): The train epochs
            num_layers (int): The number of layers
            seq_length (int): The sequence length
            step_num (int): The step number
            vocab_size (int): The vocab size
        """
        super().__init__(start, log_level)

        self.step_num = step_num
        self.vocab_size = vocab_size
        self.args = TrainingArguments(batch_size=batch_size, num_train_epochs=num_train_epochs, sink_mode=False,
                                      loss_scale_value=1024)
        self.model_config = LlamaConfig(num_layers=num_layers, seq_length=seq_length, use_flash_attention=True)

    def generator_train(self):
        """train dataset generator"""
        size = (self.step_num * self.args.batch_size, self.model_config.seq_length + 1,)
        input_ids = np.random.randint(low=0, high=self.vocab_size, size=size).astype(np.int32)
        for idx in range(len(input_ids)):
            yield input_ids[idx]

    def _train(self, jit_level='O1'):
        """
        Construct trainer and start training.

        Args:
            jit_level (str): The jit level, could be O0, O1 or O2
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
        """Run mindformers test"""
        logger.info('------------------------------Starting Pretrain Test------------------------------')
        self._set_log_level()
        try:
            if self.log_level in ['ERROR', 'CRITICAL']:
                buffer = StringIO()
                sys.stdout = buffer

            self._train(jit_level='O1')
            self._train(jit_level='O2')

            sys.stdout = sys.__stdout__

        # pylint: disable=W0702
        except:
            self._error(error_flag='Pretrain')
            vc = VersionCheck(self.start, error_flag='Pretrain')
            vc.check()
            sys.exit(1)

        else:
            self._success(test='Pretrain')

        logger.info('------------------------------Starting Predict Test------------------------------')
        self._set_log_level()
        try:
            self.model.generate([1], do_sample=False, use_past=False)
            self.model.generate([2], do_sample=False, use_past=True)

        # pylint: disable=W0702
        except:
            self._error(error_flag='Predict')
            vc = VersionCheck(self.start, error_flag='Predict')
            vc.check()

        else:
            self._success(test='Pretrain')
            self._next()

    def _error(self, **kwargs):
        sys.stdout = sys.__stdout__
        self._set_log_level(reset=True)
        error_flag = kwargs['error_flag']
        logger.error(f'{error_flag} test failed!', exc_info=True)
        logger.info('If you need any help, please open an issue in the MindFormers repository: '
                    'https://gitee.com/mindspore/mindformers/issues')

    def _success(self, **kwargs):
        self._set_log_level(reset=True)
        test = kwargs['test']
        logger.info(f'{test} test passed!')


class VersionCheck(BaseCheck):
    """Version check"""

    def __init__(self, start, error_flag=None):
        super().__init__(start)
        self.error_flag = error_flag

    @staticmethod
    def recommend(mfv):
        """
        Check whether the given mindformers version is in record

        Args:
            mfv (str): The mindformers version needs to be searched

        Returns:
            tuple:
                - bool: Whether the mindformers version is in record
                - str: The latest mindformers version or the input if the mindformers version is in record
                - dict: The corresponding environment
        """
        matched_vs = VERSION_MAPPING.get(mfv, None)
        if matched_vs:
            return True, mfv, matched_vs
        latest = max(VERSION_MAPPING.keys(), key=lambda x: tuple(map(int, x.split('.'))))
        return False, latest, VERSION_MAPPING[latest]

    def _matched(self, end):
        """
        The installed versions are matched

        Args:
            end (float): The end time of run_check
        """
        if self.error_flag:
            logger.error(f'The run check failed in {end - self.start:.2f} seconds, '
                         f'please check the above info for more details')
        else:
            logger.info(
                f'All checks passed, used {end - self.start:.2f} seconds, the environment is correctly set up!')

    def _unmatched(self, end, mfv, mfv_matching):
        """
        The installed versions are unmatched

        Args:
            end (float): The end time of run_check
            mfv (str): The recommended mindformers version
            mfv_matching (dict): The recommended environment version
        """
        if self.error_flag:
            logger.error(f"The run check failed in {end - self.start:.2f} "
                         f"seconds, It's recommended to install cann=={mfv_matching['cann']} "
                         f"mindspore=={mfv_matching['ms']} mindformers=={mfv}")
        else:
            logger.warning(f'The installed software are unmatched '
                           f'but all checks passed in {end - self.start:.2f} seconds')
            logger.info(f"It's recommended to install cann=={mfv_matching['cann']} "
                        f"mindspore=={mfv_matching['ms']} mindformers=={mfv}")

    def check(self):
        """Check whether the software versions are matched"""
        logger.info('------------------------------Searching Environment Info------------------------------')
        try:
            mfv = mf.__version__
            msv = ms.__version__
            logger.info(f'MindFormers version: {mfv}')
            logger.info(f'MindSpore version: {msv}')

            # one of ['aarch64', 'x86_64']
            arch = platform.machine()

            cann_pth = os.getenv('LOCAL_ASCEND')
            if cann_pth:
                cann_pth = Path(cann_pth) / 'ascend-toolkit' / 'latest'
            else:
                cann_pth = Path('/usr/local/Ascend/ascend-toolkit/latest')

            cann_info_file = f'{str(cann_pth)}/{arch}-linux/ascend_toolkit_install.info'
            if not Path(cann_info_file).exists():
                raise RuntimeError('Cannot find cann info file')

            cann_info = subprocess.run(f'cat {cann_info_file}', shell=True, capture_output=True).stdout.decode(
                'utf-8')
            cann_version = re.search(r'=(\d[\d.\w]+)', cann_info).group(1)
            logger.info(f'Ascend-cann-toolkit version: {cann_version}')

            driver_info_file = f'{str(cann_pth.parent.parent)}/driver/version.info'
            if not Path(driver_info_file).exists():
                raise RuntimeError('Cannot find driver info file')

            driver_info = subprocess.run(f'cat {driver_info_file}', shell=True, capture_output=True).stdout.decode(
                'utf-8')
            driver_version = re.search(r'=(\d[\d.\w]+)', driver_info).group(1)
            logger.info(f'Ascend driver version: {driver_version}')

        # 版本不匹配
        except RuntimeError as e:
            end = time.perf_counter()
            logger.error(str(e))
            mfv_inrecord, mfv, mfv_matching = self.recommend(mfv)
            self._unmatched(end, mfv, mfv_matching)

        else:
            end = time.perf_counter()
            mfv_inrecord, mfv, mfv_matching = self.recommend(mfv)

            # 版本匹配
            if (mfv_inrecord
                    and msv == mfv_matching['ms']
                    and cann_version == mfv_matching['cann']
                    and driver_version == mfv_matching['driver']):
                self._matched(end)

            # 版本不匹配
            else:
                self._unmatched(end, mfv, mfv_matching)


def timeout_handler(signum, frame):
    """Process timeout signal"""
    raise TimeoutError('Execution timed out, exceed 1 min')


def run_check(log_level):
    """
    Check whether the environment is correctly set up
    MindSpore run_check -> pretrain -> predict -> version_check

    Args:
        log_level (str):  The log level
    """
    os.environ['MS_ALLOC_CONF'] = "enable_vmm:False"

    start = time.perf_counter()
    ms.set_context(mode=0)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(120)

    msrc = MSCheck(start, log_level)
    mfc = MFCheck(start, log_level=log_level)
    vc = VersionCheck(start)

    msrc.set_next(mfc)
    mfc.set_next(vc)

    msrc.check()
