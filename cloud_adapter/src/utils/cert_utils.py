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
from fm.src.utils import encrypt_with_kmc, decrypt_with_kmc


def encrypt_cert(cert_input, cert_output):
    """
    加密认证凭据并刷新存量凭据
    """
    # 对认证信息明文进行加密组装
    cert_dict = pack_cert_plain(cert_input)

    update_info = {'scenario': cert_dict}

    cert_output.update(update_info)
    del cert_dict
    del update_info


def decrypt_cert(cert_output_cipher, flatten_to_list=False):
    """
    认证凭据解密, 支持凭据平铺/字典映射模式
    """
    # 逐层解析加密认证信息
    cert_info = unpack_cert_cipher(cert_output_cipher)

    cert_type = cert_info.get('type')
    cert_ak = decrypt_with_kmc(cert_info.get('ak'))
    cert_sk = decrypt_with_kmc(cert_info.get('sk'))
    cert_endpoint = cert_info.get('endpoint')
    del cert_info

    if flatten_to_list:
        cert_output_plain = {'scenario': [cert_type, cert_ak, cert_sk, cert_endpoint]}
    else:
        cert_output_plain = {
            'scenario': {'type': cert_type, 'ak': cert_ak, 'sk': cert_sk, 'endpoint': cert_endpoint}}

    del cert_ak
    del cert_sk

    return cert_output_plain


def pack_cert_plain(cert_input):
    """
    对认证信息明文进行加密组装
    """
    cert_type = str(cert_input[0])
    cert_ak_plain = str(cert_input[1])
    cert_sk_plain = str(cert_input[2])
    cert_endpoint = str(cert_input[3])
    del cert_input

    cert_ak_cipher = encrypt_with_kmc(cert_ak_plain)
    cert_sk_cipher = encrypt_with_kmc(cert_sk_plain)
    del cert_ak_plain
    del cert_sk_plain

    cert_dict = {
        'type': cert_type,
        'ak': cert_ak_cipher,
        'sk': cert_sk_cipher,
        'endpoint': cert_endpoint
    }

    del cert_ak_cipher
    del cert_sk_cipher

    return cert_dict


def unpack_cert_cipher(cert_output_cipher):
    """
    逐层解析加密认证信息
    """
    if 'scenario' not in cert_output_cipher:
        service_logger.error('scenario is missing in .fmrc file, check the file.')
        raise RuntimeError

    cert_info = cert_output_cipher.get('scenario')
    del cert_output_cipher

    if cert_info is None:
        service_logger.error('nothing in scenario, check the file.')
        raise RuntimeError

    cert_info_keys = list(cert_info.keys())

    if not cert_info_keys:
        service_logger.error('registry info in .fmrc file is empty, check the file.')
        raise RuntimeError

    for cert_info_value in cert_info.values():
        if cert_info_value is None:
            service_logger.error('find none value in .fmrc file, check the file.')
            raise RuntimeError
    del cert_info_value

    return cert_info
