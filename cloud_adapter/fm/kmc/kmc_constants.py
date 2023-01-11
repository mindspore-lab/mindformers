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

# kmc config paramitem
KMC_SO_PATH = 'kmc_lib'
KMC_EXT_SO = 'libkmcext.so'
KMC_SO = 'libkmc.so'
KMC_SDP_SO = 'libsdp.so'
KMC_CRYPTO_SO = 'libcrypto.so'
KMC_SECUREC_SO = 'libsshsecurec.so'
KMC_KSF_PATH = '../'
KMC_KSF_PRIMARY = 'primary'
KMC_KSF_STANDBY = 'standby'
DOMAIN_ID = 0
DOMAIN_COUNT = 8
KMC_ROLE_MASTER = 1
KMC_ROLE_AGENT = 0
PROC_LOCK_PERM = 0o660
SDP_ALG_ID = 8
HMAC_AIG_ID = 2052
DEFAULT_SEM_KEY = 0x20160000
MIN_HEX_SEM_KEY = 0x1111
MAX_HEX_SEM_KEY = 0x9999
INNER_SYMM_AI_GID = 0
INNER_HASH_AI_GID = 0
INNER_HMAC_AIG_ID = 0
INNER_KDF_AI_GID = 0
WORK_KEY_ITER = 10000
ROOT_KEY_ITER = 10000

SEC_PATH_MAX = 4096
PATH_SEPERATOR = '/'

# ret enum Item
RET_SUCCESS = 0
RET_FAILED = 1
INITIALIZE_FAILED = 2
GET_MAX_MK_ID_FAILED = 3
ACTIVE_NEW_MK_FAILED = 4
ENCRPT_FAILED = 5
DECRPT_FAILED = 6
FINALIZE_FAILED = 7
REFRESH_MASK_FAILED = 8
UPDATE_MK_FAILED = 9
FREE_MEMORY_FAILED = 10

# check master key  paramitem
INTERVAL_TIMER = 3600
LIFETIME_DAYS = 10
DEFAULT_MAX_KEYLIFE_DAYS = 180

# log paramitem
LOG_LEVEL = 3

# plain text length limit
TEXT_LENGTH = 512

UNAME_MATCHINE_INDEX = 4