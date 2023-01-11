# -*- coding: utf-8 -*-
"""
功能: 适配模块，底层适配本地场景（local）
版权信息: 华为技术有限公司, 版权所有(C) 2010-2022
"""
from fm.src.adapter.strategy import strategy_register, register_strategy, Strategy


@register_strategy('local', strategy_register)
class StrategyLocal(Strategy):
    """
        Local场景拉起作业的具体实现
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
               instance_id,
               instance_type):
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
                   instance_id,
                   instance_type):
        pass

    def test(self, *argv, **kwargs):
        pass
