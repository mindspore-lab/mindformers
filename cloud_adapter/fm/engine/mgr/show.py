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
import datetime
from datetime import datetime

from tabulate import tabulate
from fm.engine.options import ScenarioOption, AppConfigOption, JobIdOption, ModelIdOption, ServiceIdOption, \
    InstanceNumOption, InstanceTypeOption, DisplayOption

from fm.aicc_tools.ailog.log import service_logger

JOB_NAME_MAX_LEN = 64
RECORD_SIZE = 10
SUPPORT_SCENARIO = ["ma"]
MA_TAB_HEADS = {
    'job': ['job_id', 'job_name', 'status', 'create_time', 'duration', 'user'],
    'model': ['model_id', 'model_name', 'model_version', 'status', 'create_time', 'user'],
    'service': ['service_id', 'service_name', 'status', 'create_time', 'user']
}


def show_options():
    options = [ScenarioOption(), AppConfigOption(), JobIdOption(), ModelIdOption(), ServiceIdOption(),
               InstanceNumOption(), InstanceTypeOption(), DisplayOption()]

    def decorator(f):
        if not hasattr(f, '__click_params__'):
            f.__click_params__ = []
        f.__click_params__ += options
        return f

    return decorator


def is_support_display(scenario):
    if scenario not in SUPPORT_SCENARIO:
        service_logger.info(f"scenario {scenario} does not support info display")
        return False
    return True


def ma_output_list_format(output, instance_type):
    for dic in output:
        ma_output_single_format(dic, instance_type)


def ma_output_single_format(dic, instance_type):
    if instance_type == 'job':
        dic['job_id'] = dic.get('metadata').get('id')
        dic['job_name'] = dic.get('metadata').get('name')
        dic['duration'] = ms_to_hour(dic.get('status').get('duration'))
        dic['status'] = dic.get('status').get('phase')
        dic['create_time'] = datetime.fromtimestamp(int(dic.get('metadata').get('create_time') / 1000))
        dic['user'] = {'name': dic.get('metadata').get('user_name')}
    if instance_type == 'model':
        dic['model_id'] = dic.get('model_id')
        dic['model_name'] = dic.get('model_name')
        dic['model_version'] = dic.get('model_version')
        dic['status'] = dic.get('model_status')
        dic['create_time'] = datetime.fromtimestamp(int(dic.get('create_at') / 1000))
        dic['user'] = {'name': dic.get('owner')}
    if instance_type == 'service':
        dic['service_id'] = dic.get('service_id')
        dic['service_name'] = dic.get('service_name')
        dic['status'] = dic.get('status')
        dic['create_time'] = datetime.fromtimestamp(int(dic.get('publish_at') / 1000))
        dic['user'] = {'name': dic.get('owner')}


def list_show_process(output, scenario, instance_type, kwargs):
    if not is_support_display(scenario):
        return
    if kwargs.get('instance_id') is not None:
        if scenario == 'ma':
            ma_output_single_format(output, instance_type)
            service_logger.info(
                "showing job\n" + tabulate(
                    [[output.get(k) for k in MA_TAB_HEADS[instance_type]]],
                    headers=MA_TAB_HEADS[instance_type], tablefmt='simple'))
    else:
        if scenario == 'ma':
            ma_output_list_format(output, instance_type)
            cut_log(MA_TAB_HEADS[instance_type], output)


def cut_log(tab_head, output):
    """cut the log records , RECORD_SIZE jobs per record"""
    temp_output = [output[i:i + RECORD_SIZE] for i in range(0, len(output), RECORD_SIZE)]
    for record in temp_output:
        tab_data = []
        for it in record:
            tab_data += [[it[k] if k != 'user' else it[k]['name'] for k in tab_head]]
        service_logger.info("showing jobs\n" + tabulate(tab_data, headers=tab_head, tablefmt='simple'))


def ms_to_hour(millis):
    seconds = (millis / 1000) % 60
    mins = (millis / 60000) % 60
    hours = millis / 3600000
    return "%02d:%02d:%02d" % (hours, mins, seconds)


def restrict_name_len(job_name):
    if job_name is not None and len(job_name) > JOB_NAME_MAX_LEN:
        job_name = job_name[:JOB_NAME_MAX_LEN]
    return job_name
