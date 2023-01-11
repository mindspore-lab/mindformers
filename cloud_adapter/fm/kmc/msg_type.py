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
import fm.kmc.kmc_constants as kmc_constants


class KmcConfig(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ('primaryKeyStoreFile', ctypes.c_char * kmc_constants.SEC_PATH_MAX),
        ('standbyKeyStoreFile', ctypes.c_char * kmc_constants.SEC_PATH_MAX),
        ('domainCount', ctypes.c_int),
        ('role', ctypes.c_int),
        ('procLockPerm', ctypes.c_int),
        ('sdpAlgId', ctypes.c_int),
        ('hmacAlgId', ctypes.c_int),
        ('semKey', ctypes.c_int),
        ('innerSymmAlgId', ctypes.c_int),
        ('innerHashAlgId', ctypes.c_int),
        ('innerHmacAlgId', ctypes.c_int),
        ('innerKdfAlgId', ctypes.c_int),
        ('workKeyIter', ctypes.c_int),
        ('rootKeyIter', ctypes.c_int)

    ]

    def __init__(self, primaryKeyStoreFile, standbyKeyStoreFile, semKey, *args, **kwargs):
        super(KmcConfig, self).__init__(*args, **kwargs)
        self.primaryKeyStoreFile = primaryKeyStoreFile.encode('utf-8')
        self.standbyKeyStoreFile = standbyKeyStoreFile.encode('utf-8')
        self.domainCount = kmc_constants.DOMAIN_COUNT
        self.role = kmc_constants.KMC_ROLE_MASTER
        self.procLockPerm = kmc_constants.PROC_LOCK_PERM
        self.sdpAlgId = kmc_constants.SDP_ALG_ID
        self.hmacAlgId = kmc_constants.HMAC_AIG_ID
        self.semKey = semKey
        self.innerSymmAlgId = kmc_constants.INNER_SYMM_AI_GID
        self.innerHashAlgId = kmc_constants.INNER_HASH_AI_GID
        self.innerHmacAlgId = kmc_constants.INNER_HMAC_AIG_ID
        self.innerKdfAlgId = kmc_constants.INNER_KDF_AI_GID
        self.workKeyIter = kmc_constants.WORK_KEY_ITER
        self.rootKeyIter = kmc_constants.ROOT_KEY_ITER
