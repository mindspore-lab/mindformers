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
from fm.utils import constants
from fm.utils.kmc_supports import kmc_status_ok, encrypt_with_kmc, decrypt_with_kmc
from fm.utils.cert_utils import decrypt_cert, encrypt_cert
from fm.utils.io_utils import wrap_local_working_directory, is_link, read_file_with_link_check, \
    write_file_with_link_check, get_config_dir_setting, get_ca_dir_setting
from fm.utils.default_scenario_utils import get_default_scenario_from_local, set_default_scenario_to_local, \
    refresh_default_scenario_cache
from fm.utils.exception_utils import log_exception_details
from fm.utils.obs_tool import new_obs_client, check_obs_path, set_ssl_verify, obs_connection_check, \
    extract_ak_sk_endpoint_token_from_cert
from fm.utils.local_cache_utils import get_cached_cert
from fm.utils.obs_register import set_obs_path
from fm.utils.yaml_property_check import is_legal_device_num, is_legal_ip_format, is_legal_node_num
