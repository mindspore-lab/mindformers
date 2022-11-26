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
from .utils import str2bool, count_params
from .cloud_adapter import CFTS, PARALLEL_MODE, MODE, DEBUG_INFO_PATH,\
    check_in_modelarts, cloud_monitor
from .register import MindFormerRegister, MindFormerModuleType,\
    MindFormerConfig, ActionDict
from .logger import logger, StreamRedirector, AiLogFastStreamRedirect2File
