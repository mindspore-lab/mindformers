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
import ctypes
import random

from fm.kmc.kmc_wrapper_api import KmcWrapper
from fm.kmc.msg_type import KmcConfig
from fm.kmc.kmc_get_os import get_kmc_lib_path
from fm.aicc_tools.ailog.log import service_logger, service_logger_without_std
from fm.utils.io_utils import wrap_local_working_directory, get_config_dir_setting
import fm.kmc.kmc_constants as kmc_constants


@ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)
def kmc_logger_callback(level, msg):
    if level < kmc_constants.LOG_LEVEL:
        service_logger_without_std.info(msg.decode('utf-8'), extra={"skip": True})


def get_module_lib_path():
    try:
        kmc_lib_name = get_kmc_lib_path()
    except ValueError as err:
        service_logger_without_std.error(
            'encryption module get lib path failed, catch err: %s', err)
        raise err

    lib_dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                kmc_constants.KMC_SO_PATH,
                                kmc_lib_name)
    return lib_dir_path


class Kmc:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(Kmc, '_first_init'):
            return
        Kmc._first_init = True
        self.domain_id = kmc_constants.DOMAIN_ID

        try:
            lib_dir_path = get_module_lib_path()
        except Exception as err:
            service_logger_without_std.error(
                'encryption module get path failed, catch err: %s', err)
            raise err

        try:
            self.kmc_wrapper = KmcWrapper(lib_dir_path)
        except Exception as err:
            service_logger_without_std.error(
                'load encryption module failed, catch err: %s', err)
            raise err

    def finalize(self):
        try:
            self.kmc_wrapper.finalize()
        except Exception as err:
            service_logger_without_std.error(
                'destruct encryption module failed, catch err: %s', err)

    def initialize(self):
        # set logger function callback
        self.kmc_wrapper.set_logger(
            kmc_logger_callback, kmc_constants.LOG_LEVEL)
        service_logger_without_std.info(
            'encryption module set log para success.')

        # set ksf file path
        primary_ksf = wrap_local_working_directory(file_name=kmc_constants.KMC_KSF_PRIMARY,
                                                   specific_path_config=get_config_dir_setting())
        standby_ksf = wrap_local_working_directory(file_name=kmc_constants.KMC_KSF_STANDBY,
                                                   specific_path_config=get_config_dir_setting())
        if os.path.islink(primary_ksf):
            os.remove(primary_ksf)
        if os.path.islink(standby_ksf):
            os.remove(standby_ksf)

        # initial sem key
        sem_key = kmc_constants.DEFAULT_SEM_KEY + \
                  random.randint(kmc_constants.MIN_HEX_SEM_KEY,
                                 kmc_constants.MAX_HEX_SEM_KEY)

        # initial config structure
        kmc_config = KmcConfig(primary_ksf,
                               standby_ksf,
                               sem_key
                               )
        ret = self.kmc_wrapper.set_config(kmc_config)
        if ret != kmc_constants.RET_SUCCESS:
            service_logger_without_std.error(
                'encryption module set config failed, ret: %s', str(ret))
            return kmc_constants.INITIALIZE_FAILED

        # active new master key
        ret = self.active_new_mk()
        if ret != kmc_constants.RET_SUCCESS:
            service_logger_without_std.error(
                'encryption module active master key failed, ret: %s', str(ret))
            return kmc_constants.ACTIVE_NEW_MK_FAILED

        # check and update master key
        try:
            self.kmc_wrapper.check_and_update_task(self.domain_id,
                                                   kmc_constants.DEFAULT_MAX_KEYLIFE_DAYS -
                                                   kmc_constants.LIFETIME_DAYS)
        except Exception as err:
            service_logger_without_std.error(
                'encryption module check and update MK task failed, catch err: %s', err)

        return kmc_constants.RET_SUCCESS

    # active new master key
    def active_new_mk(self):
        ret, max_key_id = self.kmc_wrapper.get_max_mk_id(self.domain_id)
        if ret != kmc_constants.RET_SUCCESS:
            service_logger_without_std.info(
                'encryption module get max MK ID failed')
            return kmc_constants.GET_MAX_MK_ID_FAILED
        if int(max_key_id.value) <= 0:
            if self.kmc_wrapper.active_new_mk(self.domain_id) != kmc_constants.RET_SUCCESS:
                service_logger_without_std.error('encryption module active MK failed')
                return kmc_constants.ACTIVE_NEW_MK_FAILED

        return kmc_constants.RET_SUCCESS

    # encrypt AK/SK API
    def encrypt(self, plain_text):
        if plain_text == "":
            service_logger_without_std.warning('cipher is null.')
            return kmc_constants.ENCRPT_FAILED, ''

        ret = kmc_constants.ENCRPT_FAILED
        cipher_text = ""

        if self.initialize() != kmc_constants.RET_SUCCESS:
            raise RuntimeError('encryption module initialization fail.')

        try:
            ret, cipher_text = self.kmc_wrapper.encrypt(
                self.domain_id, plain_text)
        except Exception as err:
            service_logger_without_std.error(
                'encryption module encrypt plaintext failed, catch err: %s', err)

        if ret != kmc_constants.RET_SUCCESS:
            service_logger.error('encryption module encrypt plaintext failed.')

        self.finalize()

        del plain_text
        return ret, cipher_text

    # decrypt AK/SK API
    def decrypt(self, cipher_text):
        if cipher_text == "":
            service_logger_without_std.warning('ciphertext is null.')
            return kmc_constants.DECRPT_FAILED, ''

        ret = kmc_constants.DECRPT_FAILED
        plain_text = ""

        if self.initialize() != kmc_constants.RET_SUCCESS:
            raise RuntimeError('encryption module initialization fail.')

        try:
            ret, plain_text = self.kmc_wrapper.decrypt(
                self.domain_id, cipher_text)
        except Exception as err:
            service_logger_without_std.error(
                'encryption module decrypt ciphertext failed, catch err: %s', err)

        if ret != kmc_constants.RET_SUCCESS:
            service_logger.error(
                'encryption module decrypt ciphertext failed.')

        self.finalize()

        del cipher_text
        return ret, plain_text
