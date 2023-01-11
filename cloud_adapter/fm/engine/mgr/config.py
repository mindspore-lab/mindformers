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

import fm.aicc_tools as ac
from fm.engine.options import ScenarioOption, CachedAppConfigOption
from fm.utils import constants, wrap_local_working_directory, write_file_with_link_check, \
    read_file_with_link_check, refresh_default_scenario_cache, get_config_dir_setting
from fm.engine.utils import prepare_configs, register
from fm.utils.io_utils import DEFAULT_FLAGS, DEFAULT_MODES
from fm.aicc_tools.ailog.log import service_logger


def config_options():
    options = [ScenarioOption(), CachedAppConfigOption()]

    def decorator(f):
        if not hasattr(f, '__click_params__'):
            f.__click_params__ = []
        f.__click_params__ += options
        return f

    return decorator


@ac.aicc_monitor
def config_process(*args, **kwargs):
    scenario = kwargs.get('scenario')
    app_config = kwargs.get('app_config')

    if scenario is None and app_config is None:
        service_logger.warning('param scenario and app_config are not found, load cache from local.')

    scenario = refresh_default_scenario_cache(scenario, override=True)

    if app_config is not None:
        refresh_app_config_cache(app_config, scenario, DEFAULT_FLAGS, DEFAULT_MODES)

    # get ak, sk, endpoint, obs_path to upload log
    kwargs = prepare_configs(kwargs, DEFAULT_FLAGS, DEFAULT_MODES)

    register(kwargs)

    service_logger.info('set config info success.')

    return True


def refresh_app_config_cache(app_config, scenario, flags, modes):
    app_config_local_path = wrap_local_working_directory(file_name=constants.APP_CONFIG_LOCAL_PATH,
                                                         specific_path_config=get_config_dir_setting())

    if os.path.exists(app_config_local_path):
        try:
            app_config_output = read_file_with_link_check(app_config_local_path, flags, modes)
        except Exception as ex:
            service_logger.error('content in app_config yaml is not legal yaml-like format, '
                                 'check the file and rerun config command to initialize.')
            raise ex
    else:
        app_config_output = {'scenario': {scenario: {}}}

    update_info = {scenario: app_config}
    app_config_output.get('scenario').update(update_info)

    if os.path.isfile(app_config_local_path):
        os.remove(app_config_local_path)

    write_file_with_link_check(app_config_local_path, app_config_output, flags, modes)
