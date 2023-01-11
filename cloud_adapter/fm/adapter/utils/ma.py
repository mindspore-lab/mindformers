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
"""功能: 适配模块ModelArts场景工具函数 """
import os
from importlib import import_module

from fm.aicc_tools.utils import check_in_modelarts
from fm.utils.obs_tool import is_startswith_obs_prefix, OBS_PREFIX
from fm.adapter.utils.common import get_unique_name, select_with_priority
from fm.adapter.constants.ma import DEFAULT_DEVICE_NUM, DEFAULT_NODE_NUM, MA_JOB_DIR, \
    DEFAULT_CACHE_DATA_PATH, DEFAULT_CACHE_OUTPUT_PATH, DEFAULT_CACHE_CKPT_PATH, DEFAULT_CACHE_PRETRAINED_MODEL_PATH, \
    TRAIN_INSTANCE_TYPE_SPEC, INSTANCE_TYPE_SHARE


class MACommonParams(object):
    """
        ModelArts平台使用场景公共参数
    """
    def __init__(self, app_config, node_num, device_num):
        self.device_num = select_with_priority(device_num, app_config.get('device_num'), DEFAULT_DEVICE_NUM)
        self.node_num = select_with_priority(node_num, app_config.get('node_num'), DEFAULT_NODE_NUM)
        if self.node_num > 1 and self.device_num != 8:
            raise ValueError("device_num should be 8 when node_num is greater than 1.")


class MATrainParams(object):
    """
        ModelArts平台使用场景训练参数
    """
    def __init__(self,
                 app_config,
                 job_name,
                 model_name,
                 task_type,
                 data_path,
                 output_path,
                 ckpt_path,
                 model_config_path,
                 pretrained_model_path,
                 node_num,
                 device_num,
                 device_type,
                 backend):
        self.train_module = import_module('modelarts.train_params')
        self.ma_common_params = MACommonParams(app_config, node_num, device_num)

        self.app_config = app_config
        self.job_name = job_name
        self.model_name = model_name
        self.task_type = task_type
        self.data_path, self.code_with_data, self.use_local_data_path = data_path, False, False
        self.output_path = output_path
        self.ckpt_path, self.code_with_ckpt = ckpt_path, False
        self.log_path = app_config.get('log_path')
        self.pool_id = None
        self.instance_type = None
        self.model_config_path = model_config_path
        self.pretrained_model_path, self.code_with_pm = pretrained_model_path, False
        self.device_type = device_type
        self.backend = backend

        self._prepare_params_with_priority(app_config)
        self.hyper_params = self._construct_hyper_params()
        self.inputs = self._construct_inputs()
        self.outputs = self._construct_outputs()
        self.volumes = self._construct_volumes(app_config)

    def _prepare_params_with_priority(self, app_config):
        self.job_name = select_with_priority(self.job_name, None, get_unique_name(prefix="job"))
        self.code_path = replace_s3_prefix(app_config.get('code_url'))
        self.boot_file_path = replace_s3_prefix(app_config.get('boot_file_path'))
        self.training_files = self.train_module.TrainingFiles(code_dir=self.code_path, boot_file=self.boot_file_path)
        self.data_path = self._handle_param("data_path")
        self.use_local_data_path = not is_startswith_obs_prefix(self.data_path)
        self.code_with_data = self.data_path and self.data_path.startswith(self.code_path)
        self.ckpt_path = self._handle_param("ckpt_path")
        self.code_with_ckpt = self.ckpt_path and self.ckpt_path.startswith(self.code_path)
        self.output_path = self._handle_param("output_path")
        self.log_path = self._handle_param("log_path")
        self.model_config_path = self._handle_param("model_config_path")
        self.pretrained_model_path = self._handle_param("pretrained_model_path")
        self.code_with_pm = self.pretrained_model_path and self.pretrained_model_path.startswith(self.code_path)
        self.pool_id = str(app_config.get('pool_id')) if str(app_config.get('pool_id')) != "None" else None
        self.instance_type = TRAIN_INSTANCE_TYPE_SPEC.get(self.ma_common_params.device_num) if self.pool_id else \
            INSTANCE_TYPE_SHARE.get(self.ma_common_params.device_num)

        if self.output_path is None or self.data_path is None:
            raise ValueError('Parameters data_path and output path cannot be None.')

    def _handle_param(self, name):
        return replace_s3_prefix(select_with_priority(self.__getattribute__(name), self.app_config.get(name)))

    def _construct_hyper_params(self):
        hyper_params = [{'name': 'task_type', 'value': str(self.task_type)}]
        if self.model_name:
            hyper_params.append({'name': 'model_name', 'value': str(self.model_name)})
        code_path = os.path.join(MA_JOB_DIR, os.path.basename(self.code_path.strip("/")) + "/")
        hyper_params.append({'name': 'code_path', 'value': code_path})
        boot_file_path = os.path.join(code_path, self.boot_file_path.replace(self.code_path, ''))
        hyper_params.append({'name': 'boot_file_path', 'value': boot_file_path})
        data_path = DEFAULT_CACHE_DATA_PATH
        if self.use_local_data_path:
            data_path = str(self.data_path)
            hyper_params.append({'name': 'use_sfs', 'value': str(True)})
        elif self.code_with_data:
            data_path = os.path.join(code_path, self.data_path.replace(self.code_path, ''))
        hyper_params.append({'name': 'data_path', 'value': data_path})
        hyper_params.append({'name': 'output_path', 'value': DEFAULT_CACHE_OUTPUT_PATH})
        if self.model_config_path:
            hyper_params.append({'name': 'model_config_path', 'value': self.model_config_path})
        if self.pretrained_model_path:
            pretrained_model_path = DEFAULT_CACHE_PRETRAINED_MODEL_PATH
            if self.code_with_pm:
                pretrained_model_path = os.path.join(code_path, self.pretrained_model_path.replace(self.code_path, ''))
            hyper_params.append({'name': 'pretrained_model_path', 'value': pretrained_model_path})
        if self.ckpt_path:
            ckpt_path = DEFAULT_CACHE_CKPT_PATH
            if self.code_with_ckpt:
                ckpt_path = os.path.join(code_path, self.ckpt_path.replace(self.code_path, ''))
            hyper_params.append({'name': 'ckpt_path', 'value': ckpt_path})
        if self.log_path:
            hyper_params.append({'name': 'log_path', 'value': self.log_path})
        hyper_params.append({'name': 'device_type', 'value': self.device_type})
        hyper_params.append({'name': 'backend', 'value': self.backend})

        return hyper_params

    def _construct_inputs(self):
        inputs = []
        if not self.use_local_data_path and not self.code_with_data:
            inputs.append(self.train_module.InputData(obs_path=self.data_path,
                                                      local_path=DEFAULT_CACHE_DATA_PATH,
                                                      name="data_path"))
        if self.task_type == "finetune" and not self.code_with_pm:
            inputs.append(self.train_module.InputData(obs_path=self.pretrained_model_path,
                                                      local_path=DEFAULT_CACHE_PRETRAINED_MODEL_PATH,
                                                      name="pretrained_model_path"))
        if self.task_type != "finetune" and not self.code_with_ckpt:
            inputs.append(self.train_module.InputData(obs_path=self.ckpt_path,
                                                      local_path=DEFAULT_CACHE_CKPT_PATH,
                                                      name="ckpt_path"))

        return inputs

    def _construct_outputs(self):
        return [self.train_module.OutputData(obs_path=self.output_path,
                                             local_path=DEFAULT_CACHE_OUTPUT_PATH,
                                             name='output_path')]

    def _construct_volumes(self, app_config):
        volumes = []
        if self.use_local_data_path:
            local_path = app_config.get('nas_mount_path')
            nfs_server_path = app_config.get('nas_share_addr')
            if not local_path or not nfs_server_path:
                raise ValueError('Local data_path is supported only when using SFS.')
            volumes.append({"nfs": {"local_path": local_path, "nfs_server_path": nfs_server_path, "read_only": False}})

        return volumes


class MAInferParams(object):
    """
        ModelArts平台使用场景推理参数
    """
    def __init__(self,
                 app_config,
                 node_num=None,
                 device_num=None,
                 model_name=None,
                 model_version=None,
                 model_path=None,
                 service_name=None,
                 model_id=None):
        if not app_config.get("deployment"):
            raise ValueError("deployment field is required in app_config "
                             "during model publication and service deployment")
        if not app_config.get('swr_endpoint'):
            raise ValueError("swr_endpoint field is required in app_config "
                             "during model publication and service deployment")
        deploy_config = app_config.get("deployment")
        self.ma_common_params = MACommonParams(deploy_config, node_num, device_num)

        self.model_name = select_with_priority(model_name, None, get_unique_name(prefix="model"))
        self.model_version = model_version
        self.model_path = remove_head(model_path) if model_path else model_path

        self.service_name = select_with_priority(service_name, None, get_unique_name(prefix="service"))
        self.model_id = model_id
        self.cluster_id = deploy_config.get('pool_id')
        self.specification = INSTANCE_TYPE_SHARE.get(self.ma_common_params.device_num)


def get_session(cert, app_config):
    """
    when training on modelArts get session from modelArts
    else get session from local
    """
    session_module = import_module('modelarts.session')

    session_module.Session.set_endpoint(
        iam_endpoint=app_config.get('iam_endpoint'),
        obs_endpoint=app_config.get('obs_endpoint'),
        modelarts_endpoint=app_config.get('modelarts_endpoint'),
        region_name=app_config.get('region_name'))

    if check_in_modelarts():
        session = session_module.Session(
            project_id=app_config.get('project_id'),
            region_name=app_config.get('region_name'))
    else:
        session = session_module.Session(
            access_key=cert[1],
            secret_key=cert[2],
            project_id=app_config.get('project_id'),
            region_name=app_config.get('region_name'))
        del cert

    return session


def remove_head(obs_path):
    """
    功能：删除obs路径头，s3://path  -> /path  or  obs://path  -> /path
    """
    no_head_path = None
    if obs_path.startswith(OBS_PREFIX[0]):
        no_head_path = obs_path[5:]
    elif obs_path.startswith(OBS_PREFIX[1]):
        no_head_path = obs_path[4:]

    return no_head_path


def replace_s3_prefix(obs_path):
    """
    功能：适配启动包，替换obs路径头，s3://path  -> obs://path
    """
    if obs_path and obs_path.startswith(OBS_PREFIX[1]):
        return obs_path.replace(OBS_PREFIX[1], OBS_PREFIX[0])
    return obs_path


def remove_endpoint_protocol_prefix(endpoint):
    """
    功能：删除配置在 app_config 中云端服务 endpoint 的协议前缀，https://path  -> path
    """
    if endpoint and endpoint.startswith("https://"):
        return endpoint[8:]
    if endpoint and endpoint.startswith("http://"):
        return endpoint[7:]
    return endpoint
