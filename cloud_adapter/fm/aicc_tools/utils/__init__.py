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
from fm.aicc_tools.utils.cloud_adapter import Obs2Local, mox_adapter, obs_register
from fm.aicc_tools.utils.utils import get_rank_info, get_num_nodes_devices, Const
from fm.aicc_tools.utils.validator import check_obs_url, check_in_modelarts

__all__ = ['Obs2Local', 'mox_adapter', 'obs_register', 'get_rank_info',
           'get_num_nodes_devices', 'check_obs_url', 'check_in_modelarts', 'Const'
           ]
