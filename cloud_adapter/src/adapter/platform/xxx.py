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
功能: 适配模块，底层适配第三方xxx平台
版权信息: 华为技术有限公司, 版权所有(C) 2010-2022
"""
from fm.src.adapter.strategy import strategy_register, register_strategy, Strategy


@register_strategy('xxx', strategy_register)
class StrategyXXX(Strategy):
    """
        第三方平台拉起作业的具体实现
    """

    def __init__(self):
        super().__init__()

    def finetune(self,
                 model_name,
                 cert,
                 app_config,
                 model_config_path,
                 data_path,
                 output_path,
                 node_num=1,
                 device_num=8,
                 resume=False,
                 job_name=None,
                 pretrained_model_path=None,
                 backend='mindspore',
                 device_type='npu'):
        pass

    def evaluate(self,
                 cert,
                 app_config,
                 model_config_path,
                 data_path,
                 output_path,
                 ckpt_path,
                 node_num=1,
                 device_num=8,
                 resume=False,
                 job_name=None,
                 backend='mindspore',
                 device_type='npu'):
        pass

    def pretrain(self,
                 model_name,
                 cert,
                 app_config,
                 model_config_path,
                 data_path,
                 output_path,
                 node_num=1,
                 device_num=8,
                 resume=False,
                 job_name=None,
                 pretrained_model_path=None,
                 backend='mindspore',
                 device_type='npu'):
        pass

    def stop(self,
             cert,
             app_config,
             job_id):
        pass

    def delete(self,
               cert,
               app_config,
               instance_type,
               instance_id):
        pass

    def show(self,
             cert,
             app_config,
             instance_num,
             instance_id,
             instance_type,
             get_all):
        pass

    def query_duration(self,
                       cert,
                       app_config,
                       job_id):
        pass

    def get_status(self,
                   cert,
                   app_config,
                   instance_type,
                   instance_id):
        pass

    def test(self, *argv, **kwargs):
        pass
