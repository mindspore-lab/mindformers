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
import os

from fm.aicc_tools.ailog.log import service_logger_without_std, service_logger
from fm.utils import wrap_local_working_directory, constants, read_file_with_link_check, \
    write_file_with_link_check, get_config_dir_setting
from fm.utils.io_utils import DEFAULT_FLAGS, DEFAULT_MODES


def get_default_scenario_from_local():
    default_scenario_file_local_path = wrap_local_working_directory(file_name=constants.DEFAULT_SCENARIO_LOCAL_PATH,
                                                                    specific_path_config=get_config_dir_setting())

    if not os.path.exists(default_scenario_file_local_path):
        service_logger.error('no default scenario local cache, use config command or param to initialize.')
        raise RuntimeError
    else:
        default_scenario = read_file_with_link_check(default_scenario_file_local_path,
                                                     DEFAULT_FLAGS, DEFAULT_MODES).get('default_scenario')

        return default_scenario


def set_default_scenario_to_local(default_scenario):
    default_scenario_file_local_path = wrap_local_working_directory(file_name=constants.DEFAULT_SCENARIO_LOCAL_PATH,
                                                                    specific_path_config=get_config_dir_setting())

    if os.path.exists(default_scenario_file_local_path):
        os.remove(default_scenario_file_local_path)

    write_info = {'default_scenario': str(default_scenario)}

    write_file_with_link_check(default_scenario_file_local_path, write_info, DEFAULT_FLAGS, DEFAULT_MODES)

    service_logger_without_std.info('set default scenario: %s', default_scenario)


def refresh_default_scenario_cache(scenario, override=False):
    if scenario is None:
        try:
            scenario = get_default_scenario_from_local()
        except Exception as ex:
            service_logger.error('scenario is necessary, please specify scenario in param.')
            raise ex

    if override:
        set_default_scenario_to_local(scenario)

    return scenario
