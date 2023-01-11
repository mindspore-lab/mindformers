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
from fm.aicc_tools.ailog.log import service_logger


def is_legal_node_num(node_num):
    """
    check the compliance of node_num
    """

    if not node_num and node_num != 0:
        service_logger.error('parameter node_num is empty, check the setting.')
        return False

    try:
        num_value = int(node_num)
    except ValueError:
        service_logger.error('node_num should be an integer, other types are not allowed')
        return False

    if num_value > 0:
        return True
    else:
        service_logger.error('node_num should be larger than 0')
        return False


def is_legal_device_num(device_num):
    """
    check the compliance of device_num
    """

    if device_num is None:
        service_logger.error('parameter device_num is empty, check the setting.')
        return False

    try:
        device_num = int(device_num)
    except ValueError:
        service_logger.error('device_num should be an integer, other types are not allowed')
        return False

    if device_num <= 0:
        service_logger.error('device_num should be larger than 0')
        return False

    valid_device_num = [1, 2, 4, 8]
    if device_num in valid_device_num:
        return True
    else:
        service_logger.error(
            'device_num should be set within 1/2/4/8 when app_config.scenario.modelarts.pool_id is none')
        return False


def is_legal_ip_format(content_name, content_value):
    """
    check the compliance of ip format
    """

    if content_value is None:
        service_logger.error('parameter %s is empty, check the setting.', content_name)
        return False

    if ':' not in content_value:
        service_logger.error('character : is required')
        return False

    if '/' not in content_value:
        service_logger.error('character / is required')
        return False

    if content_value.count(':') > 1 or content_value.count('/') > 1:
        service_logger.error('character : or / should only appear one time')
        return False

    ip_part = content_value.strip(':/')
    if '.' not in ip_part:
        service_logger.error('character . is required')
        return False

    if ':' in ip_part:
        service_logger.error('port is not required')
        return False

    ip_segments = ip_part.split('.')
    if len(ip_segments) != 4:
        service_logger.error('4 subnet segments are required in ip address, for example: a.b.c.d:/')
        return False

    for i in range(4):
        try:
            ip_segments[i] = int(ip_segments[i])
        except ValueError:
            service_logger.error('subnet should be an integer')
            return False

        if ip_segments[i] < 0 or ip_segments[i] > 255:
            service_logger.error('subnet should be an integer between 0 and 255')
            return False

    if ip_segments[0] == 0:
        service_logger.error('first subnet should not be set as 0')
        return False

    return True
