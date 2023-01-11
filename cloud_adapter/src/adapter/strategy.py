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
"""适配模块，对外提供统一的接口调用，底层适配不同的AI平台或者本地场景"""

from abc import ABCMeta, abstractmethod

strategy_register = {}


def register_strategy(strategy_type, strategy_register_):
    def inner(cls):
        strategy_register_[strategy_type] = cls
        return cls

    return inner


class Strategy(metaclass=ABCMeta):
    """
        不同平台拉起作业的抽象类
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def finetune(self,
                 model_name,
                 cert,
                 app_config,
                 model_config_path,
                 data_path,
                 output_path,
                 node_num,
                 device_num,
                 resume=False,
                 job_name=None,
                 pretrained_model_path=None,
                 backend='mindspore',
                 device_type='npu'):
        """
        功能： 模型微调接口
        参数：
            model_name： 模型名称
            cert: 按照规则给的list, 顺序要严格按照固定规则给。如 [cert_type, ak, sk], [cert_type, token], [cert_type, user, password]
            app_config: 按照app_config.yaml解析后的key:value对象. {iam_endpoint:xxx, obs_endpoint:yyy, ....}
            model_config_path:直接透传obs给1包
            data_path: 数据集路径, modelarts为obs路径
            result_path: 训练结果路径, modelarts为obs路径
            node_num: 计算集群的节点数量
            device_num: 每个计算节点的卡数(1, 2, 4, 8), modelarts场景取值为：modelarts.kat1.xlarge,
            resume: 是否需要断点续训， 默认False
            task_id: 与resume为True时配合使
            task_path: 与resume为True时配合使用
            pretrained_model_path: 预训练模型的路径, modelarts为obs路径
            backend: mindspore/pytorch/tensorflow, 目前只支持mindspore
            device_type: GPU/NPU, 目前值支持NPU
        返回值：

        """
        pass

    @abstractmethod
    def evaluate(self,
                 cert,
                 app_config,
                 model_config_path,
                 data_path,
                 output_path,
                 node_num,
                 device_num,
                 ckpt_path,
                 resume=False,
                 job_name=None,
                 backend='mindspore',
                 device_type='npu'):
        pass

    def infer(self,
              cert,
              app_config,
              model_config_path,
              data_path,
              output_path,
              node_num,
              device_num,
              ckpt_path,
              resume=False,
              job_name=None,
              backend='mindspore',
              device_type='npu'):
        pass

    @abstractmethod
    def pretrain(self,
                 model_name,
                 cert,
                 app_config,
                 model_config_path,
                 data_path,
                 output_path,
                 node_num,
                 device_num,
                 resume=False,
                 job_name=None,
                 pretrained_model_path=None,
                 backend='mindspore',
                 device_type='npu'):
        pass

    @abstractmethod
    def stop(self,
             cert,
             app_config,
             job_id):
        pass

    @abstractmethod
    def delete(self,
               cert,
               app_config,
               instance_id,
               instance_type):
        pass

    @abstractmethod
    def show(self,
             cert,
             app_config,
             instance_num,
             instance_id,
             instance_type,
             get_all):
        pass

    @abstractmethod
    def query_duration(self,
                       cert,
                       app_config,
                       job_id):
        pass

    @abstractmethod
    def get_status(self,
                   cert,
                   app_config,
                   instance_id,
                   instance_type):
        pass

    @abstractmethod
    def test(self, *argv, **kwargs):
        pass
