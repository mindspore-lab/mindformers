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
"""MindFormers Tools."""
from .cloud_adapter import *
from .register import *
from .logger import logger, StreamRedirector, AiLogFastStreamRedirect2File
from .utils import PARALLEL_MODE, MODE, DEBUG_INFO_PATH, \
    check_in_modelarts, str2bool, count_params, get_output_root_path, \
    get_output_subpath, set_output_path, set_strategy_save_path, check_shared_disk
from .generic import add_model_info_to_auto_map
from .hub import *

__all__ = ['logger']
__all__.extend(cloud_adapter.__all__)
__all__.extend(register.__all__)
