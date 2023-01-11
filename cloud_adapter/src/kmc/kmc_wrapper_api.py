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
import ctypes
from ctypes.util import find_library
import os

import fm.src.kmc.kmc_util.kmc_constants as kmc_constants
from fm.src.kmc.kmc_util.msg_type import KmcConfig
from fm.src.aicc_tools.ailog.log import service_logger_without_std


class KmcWrapper:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, lib_url):
        if hasattr(KmcWrapper, '_first_init'):
            return
        KmcWrapper._first_init = True
        lib_url = os.path.realpath(lib_url)
        try:
            self.kmc_lib = ctypes.cdll.LoadLibrary(
                lib_url + '/' + kmc_constants.KMC_EXT_SO)
            self.libc_lib = ctypes.cdll.LoadLibrary(find_library("c"))
        except Exception as err:
            service_logger_without_std.error('load so failed, catch err: %s', err)
            raise err

    def finalize(self):
        try:
            self.kmc_lib.KeFinalize.restype = ctypes.c_int
            self.kmc_lib.KeFinalize.argtypes = []
            self.kmc_lib.KeFinalize()
        except Exception as err:
            service_logger_without_std.error('KeFinalize failed, catch err: %s', err)

    def set_logger(self, kmc_logger_cb, logger_level):
        try:
            self.kmc_lib.KeSetLoggerCallback(kmc_logger_cb)
        except Exception as err:
            service_logger_without_std.error(
                'set_logger_callback_failed, catch err: %s', err)

        try:
            self.kmc_lib.KeSetLoggerLevel(logger_level)
        except Exception as err:
            service_logger_without_std.error('set_logger failed, catch err: %s', err)

    def set_config(self, config):
        ret = kmc_constants.INITIALIZE_FAILED
        try:
            ret = self.kmc_lib.KeInitialize(ctypes.POINTER(KmcConfig)(config))
        except Exception as err:
            service_logger_without_std.error('KeGetMaxMkID failed, catch err: %s', err)

        return ret

    def get_max_mk_id(self, domain_id):
        max_key_id = ctypes.c_uint()
        ret = kmc_constants.GET_MAX_MK_ID_FAILED

        try:
            ret = self.kmc_lib.KeGetMaxMkID(ctypes.c_uint(domain_id),
                                            ctypes.pointer(max_key_id))
        except Exception as err:
            service_logger_without_std.error('KeGetMaxMkID failed, catch err: %s', err)

        return ret, max_key_id

    def active_new_mk(self, domain_id):
        ret = kmc_constants.ACTIVE_NEW_MK_FAILED
        try:
            ret = self.kmc_lib.KeActiveNewKey(ctypes.c_uint(domain_id))
        except Exception as err:
            service_logger_without_std.error('KeActiveNewKey failed, catch err: %s', err)

        return ret

    def check_and_update_task(self, domain_id, advance_days):
        refresh_mask_ret = kmc_constants.REFRESH_MASK_FAILED
        update_mk_ret = kmc_constants.UPDATE_MK_FAILED
        try:
            refresh_mask_ret = self.kmc_lib.KeRefreshMkMask()
        except Exception as err:
            service_logger_without_std.error(
                'refresh MK mask failed, catch error: %s', err)

        if refresh_mask_ret:
            service_logger_without_std.error(
                'refresh MK mask failed, ret: %d', refresh_mask_ret)
            return

        try:
            update_mk_ret = self.kmc_lib.KeCheckAndUpdateMk(ctypes.c_uint(domain_id),
                                                            ctypes.c_int(advance_days))
        except Exception as err:
            service_logger_without_std.error(
                'check_and_updat MK failed, catch err: %s', err)

        if update_mk_ret:
            service_logger_without_std.error(
                'check_and_updat MK failed, ret:%d', update_mk_ret)

    def encrypt(self, domain_id, content):
        plain_text_len = len(content)
        plain_text = content.encode('utf-8')
        plcipher_text_len = ctypes.c_int()
        cipher_text = ctypes.c_char_p()
        ret = kmc_constants.ENCRPT_FAILED
        free_ret = kmc_constants.FREE_MEMORY_FAILED

        try:
            ret = self.kmc_lib.KeEncryptByDomain(ctypes.c_uint(domain_id),
                                                 ctypes.c_char_p(plain_text),
                                                 ctypes.c_int(plain_text_len),
                                                 ctypes.pointer(cipher_text),
                                                 ctypes.pointer(plcipher_text_len))

        except Exception as err:
            service_logger_without_std.error('encrypt failed, catch err: %s', err)

        cipher_value = cipher_text.value.decode('utf-8', 'ignore')

        try:
            free_ret = self.libc_lib.free(cipher_text)
        except Exception as err:
            service_logger_without_std.error(
                'free ciphertext memory failed, catch err: %s', err)

        if free_ret:
            service_logger_without_std.error(
                'free ciphertext memory failed, ret: %d', free_ret)

        del content
        del plain_text
        del cipher_text
        return ret, cipher_value

    def decrypt(self, domain_id, content):
        cipher_text_len = len(content)
        cipher_text = content.encode('utf-8')
        plain_text_len = ctypes.c_int()
        plain_text = ctypes.c_char_p()
        ret = kmc_constants.DECRPT_FAILED
        free_ret = kmc_constants.FREE_MEMORY_FAILED

        try:
            ret = self.kmc_lib.KeDecryptByDomain(ctypes.c_uint(domain_id),
                                                 ctypes.c_char_p(cipher_text),
                                                 ctypes.c_int(
                                                     cipher_text_len),
                                                 ctypes.pointer(plain_text),
                                                 ctypes.pointer(plain_text_len))
        except Exception as err:
            service_logger_without_std.error(
                'decrypt ciphertext failed, catch err: %s', err)

        plain_value = plain_text.value.decode('utf-8', 'ignore')

        try:
            free_ret = self.libc_lib.free(plain_text)
        except Exception as err:
            service_logger_without_std.error(
                'free plaintext memory failed, catch err: %s', err)

        if free_ret:
            service_logger_without_std.error(
                'free plaintext memory failed, ret: %d', free_ret)

        del content
        del cipher_text
        del plain_text
        return ret, plain_value
