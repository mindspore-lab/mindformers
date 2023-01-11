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
import shutil

from fm.engine.options import EnableCaOption, CertPathOption
from fm.utils import constants, wrap_local_working_directory, write_file_with_link_check, \
    get_ca_dir_setting
from fm.aicc_tools.ailog.log import service_logger
from fm.utils.io_utils import DEFAULT_FLAGS, DEFAULT_MODES


def enable_verify_options():
    options = [EnableCaOption(), CertPathOption()]

    def decorator(f):
        if not hasattr(f, '__click_params__'):
            f.__click_params__ = []
        f.__click_params__ += options
        return f

    return decorator


def cert_verify(*args, **kwargs):
    """
    extract enable(bool), path(str) from kwargs,store in yaml
    :param args:
    :param kwargs:
    :return:
    """
    enable = kwargs.get('enable')
    path = kwargs.get('path')
    ca_config_path = wrap_local_working_directory(file_name=constants.VERIFY_CONFIG_LOCAL_PATH,
                                                  specific_path_config=get_ca_dir_setting())
    ca_file_path = wrap_local_working_directory(file_name=constants.CA_FILE_LOCAL_PATH,
                                                specific_path_config=get_ca_dir_setting())
    if enable and path is None and not os.path.exists(ca_file_path):
        service_logger.error('ca file path can not be None')
        raise ValueError
    content = {'enable': enable}
    if os.path.isfile(ca_config_path):
        os.remove(ca_config_path)

    if enable and path is not None:
        if os.path.exists(ca_file_path) and os.path.isfile(ca_file_path):
            os.remove(ca_file_path)
        shutil.copy(path, ca_file_path)
    write_file_with_link_check(ca_config_path, content, DEFAULT_FLAGS, DEFAULT_MODES)
    service_logger.info(f"set cert verify {'enabled' if enable else 'disabled'}")
    return True
