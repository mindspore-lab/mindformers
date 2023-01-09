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
import click

from fm.src.engine.callback import app_callback, name_callback, obs_path_callback, node_num_callback, \
    device_num_callback, backend_callback, boolean_option_callback, \
    id_callback, instance_num_callback, instance_type_callback, obs_path_with_file_callback, data_path_callback, \
    model_version_callback, cached_app_callback, cert_path_callback, ckpt_path_callback


class ScenarioOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-sn', '--scenario'),
            type=click.Choice(['modelarts']),
            default=None,
            help='scenario for using',
            show_default=True)


class EnableCaOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-e', '--enable'),
            type=bool,
            default=True,
            help='enable ca verify or not',
            show_default=True)


class CertPathOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-p', '--path'),
            type=str,
            default=None,
            callback=cert_path_callback,
            help='the path of CA file',
            show_default=True)


class ModelNameOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-m', '--model_name'),
            type=str,
            default='',
            callback=name_callback,
            help='model name',
            show_default=True)


class AppConfigOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-a', '--app_config'),
            type=str,
            default=None,
            callback=app_callback,
            help='app config file address',
            show_default=True)


class CachedAppConfigOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-a', '--app_config'),
            type=str,
            default=None,
            callback=cached_app_callback,
            help='app config file address',
            show_default=True)


class DataPathOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-dp', '--data_path'),
            type=str,
            default=None,
            callback=data_path_callback,
            help='data path address',
            show_default=True)


class OutputPathOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-op', '--output_path'),
            type=str,
            default=None,
            callback=obs_path_callback,
            help='output path address',
            show_default=True)


class NodeNumOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-nn', '--node_num'),
            type=int,
            default=None,
            callback=node_num_callback,
            help='node num',
            show_default=True)


class DeviceNumOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-dn', '--device_num'),
            type=int,
            default=None,
            callback=device_num_callback,
            help='device num',
            show_default=True)


class DeviceTypeOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-d', '--device_type'),
            type=click.Choice(['npu']),
            default='npu',
            help='only support npu now',
            show_default=True)


class BackendOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-b', '--backend'),
            default='mindspore',
            callback=backend_callback,
            help='only support mindspore now',
            show_default=True)


class ResumeOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-r', '--resume'),
            type=bool,
            default=False,
            callback=boolean_option_callback,
            help='resume status, True/False',
            show_default=True)


class ModelConfigPathOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-c', '--model_config_path'),
            type=str,
            default=None,
            callback=obs_path_with_file_callback,
            help='model config file address',
            show_default=True)


class JobNameOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-jn', '--job_name'),
            type=str,
            default='',
            callback=name_callback,
            help='job name',
            show_default=True)


class PretrainedModelPathOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-pm', '--pretrained_model_path'),
            type=str,
            default=None,
            callback=obs_path_callback,
            help='pre-trained model path address',
            show_default=True)


class JobIdOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-j', '--job_id'),
            type=str,
            default=None,
            callback=id_callback,
            help='job id',
            show_default=True)


class InstanceNumOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-im', '--instance_num'),
            type=int,
            default=20,
            callback=instance_num_callback,
            help='instance number',
            show_default=True)


class InstanceTypeOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-it', '--instance_type'),
            type=str,
            default="job",
            callback=instance_type_callback,
            help='instance type',
            show_default=True)


class DisplayOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-dis', '--display'),
            type=bool,
            default=True,
            callback=boolean_option_callback,
            help='whether display on command line, True/False',
            show_default=True)


# 模型训练权重结果路径参数选项
class CkptFileOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-cp', '--ckpt_path'),
            type=str,
            default=None,
            callback=ckpt_path_callback,
            help='checkpoint file path',
            show_default=True)


# for publish
class ModelVersionOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-mv', '--model_version'),
            type=str,
            callback=model_version_callback,
            help='model version',
            show_default=True,
            required=True)


class ModelPathOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-mp', '--model_path'),
            type=str,
            callback=obs_path_callback,
            help='path of model and related scripts',
            show_default=True,
            required=True)


# for deploy
class ModelIdOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-mid', '--model_id'),
            type=str,
            callback=id_callback,
            help='model id',
            show_default=True)


class ServiceNameOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-sen', '--service_name'),
            type=str,
            callback=name_callback,
            help='service name',
            show_default=True)


class ServiceIdOption(click.core.Option):
    def __init__(self):
        super().__init__(
            param_decls=('-sid', '--service_id'),
            type=str,
            callback=id_callback,
            help='service id',
            show_default=True)
