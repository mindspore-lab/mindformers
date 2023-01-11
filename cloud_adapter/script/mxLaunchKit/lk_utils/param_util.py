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
"""
功能: common parameter utils of launcher
"""

import os

from lk_utils import is_valid_path
from common_consts import BASE_PATH_PARAMS


def prepare_common_path_parameters(args, model_task_name, current_node_name):
    arg_dict = vars(args)

    def check_common_path_parameters(param_name, is_folder=False):
        value = arg_dict.get(param_name)
        if not value:
            raise ValueError(f"Parameter {param_name} is required.")
        if not is_valid_path(value, is_folder=is_folder):
            raise ValueError(f"Value of parameter {param_name} is not valid.")

    for param in BASE_PATH_PARAMS:
        # param: (param_name, is_folder)
        check_common_path_parameters(param_name=param[0], is_folder=param[1])
    # create final output_path
    final_output_path = os.path.join(args.output_path, model_task_name, current_node_name + os.sep)
    os.makedirs(final_output_path, exist_ok=True)
    return ["--data_path", args.data_path, "--output_path", final_output_path]
