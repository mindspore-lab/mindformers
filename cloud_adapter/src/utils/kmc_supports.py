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
from fm.src.aicc_tools.ailog.log import service_logger
from fm.src.kmc import kmc_constants, Kmc


def kmc_status_ok(status):
    return status == kmc_constants.RET_SUCCESS


def decrypt_with_kmc(cipher_content):
    """
    使用kmc解密
    """
    kmc = Kmc()

    decrypt_ret, plain_content = kmc.decrypt(cipher_content)

    if not kmc_status_ok(decrypt_ret):
        service_logger.error('error occurred in cert decryption')
        raise PermissionError

    return plain_content


def encrypt_with_kmc(plain_content):
    """
    使用kmc加密
    """
    kmc = Kmc()

    encrypt_ret, cipher_content = kmc.encrypt(str(plain_content))

    if not kmc_status_ok(encrypt_ret):
        service_logger.error('error occurred in cert encryption')
        raise PermissionError

    return cipher_content
