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
功能: 适配模块，底层适配ModelArts平台
版权信息: 华为技术有限公司, 版权所有(C) 2010-2022
"""
import os
from importlib import import_module
from fm.src.adapter.strategy import Strategy, strategy_register, register_strategy
from fm.src.adapter.utils.ma import MATrainParams, MAInferParams, get_session, remove_endpoint_protocol_prefix
from fm.src.adapter.constants.common import SUPPORTED_TASK_TYPES
from fm.src.adapter.constants.ma import MA_LAUNCHER, JOB_STATUS
from fm.src.aicc_tools.ailog.log import service_logger


@register_strategy('modelarts', strategy_register)
class StrategyModelArts(Strategy):
    """
        ModelArts平台拉起实例的具体实现(新版训练作业、AI应用)
    """

    def __init__(self, ):
        super().__init__()
        self.max_jobs = 10000
        self.scenario = 'ma'
        self.estimator_module = import_module('modelarts.estimatorV2')
        self.model_module = import_module('modelarts.model')
        self.config_module = import_module('modelarts.config')

    def deploy_ma_training_task(self, cert, app_config, task_type, **kwargs):
        """
        功能： MA训练任务创建
        参数：
            cert: 按照规则给的list, 顺序要严格按照固定规则给。如 [cert_type, ak, sk], [cert_type, token], [cert_type, user, password]
            app_config: 按照app_config.yaml解析后的key:value对象. {iam_endpoint:xxx, obs_endpoint:yyy, ....}
            kwargs: 非必须入参
        返回值：job_id
        """
        server_status = self.jobs_check(cert, app_config)
        if not server_status:
            service_logger.error('Server is overloading, Please wait a moment and try it again.')
            raise RuntimeError('Server is overloading, Please wait a moment and try it again.')

        if task_type not in SUPPORTED_TASK_TYPES:
            service_logger.error('Task type is invalid.')
            raise RuntimeError('Task type is invalid.')

        session = get_session(cert, app_config)
        ma_train_params = MATrainParams(app_config,
                                        kwargs.get("job_name"),
                                        kwargs.get("model_name"),
                                        task_type,
                                        kwargs.get("data_path"),
                                        kwargs.get("output_path"),
                                        kwargs.get("ckpt_path"),
                                        kwargs.get("model_config_path"),
                                        kwargs.get("pretrained_model_path"),
                                        kwargs.get("node_num"),
                                        kwargs.get("device_num"),
                                        kwargs.get("device_type"),
                                        kwargs.get("backend"))
        user_command = f"python {MA_LAUNCHER}"
        estimator = self.estimator_module.Estimator(session=session,
                                                    training_files=ma_train_params.training_files,
                                                    outputs=ma_train_params.outputs,
                                                    parameters=ma_train_params.hyper_params,
                                                    user_image_url=app_config.get('user_image_url'),
                                                    user_command=user_command,
                                                    pool_id=ma_train_params.pool_id,
                                                    train_instance_type=ma_train_params.instance_type,
                                                    train_instance_count=ma_train_params.ma_common_params.node_num,
                                                    log_url=ma_train_params.log_path,
                                                    volumes=ma_train_params.volumes,
                                                    env_variables={'USER_ENV_VAR': 'customize environment variable'},
                                                    job_description='mxFoundationModel Launch Train Job')
        job_instance = estimator.fit(inputs=ma_train_params.inputs, job_name=ma_train_params.job_name)

        job_info = job_instance.get_job_info()
        job_id = job_info.get('metadata').get('id')
        service_logger.info('job id: %s', job_id)
        service_logger.info('job name: %s', ma_train_params.job_name)
        service_logger.info('launch task status: %s', job_info.get('status').get('phase'))
        return job_id

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
        return self.deploy_ma_training_task(cert,
                                            app_config,
                                            "finetune",
                                            model_name=model_name,
                                            model_config_path=model_config_path,
                                            data_path=data_path,
                                            output_path=output_path,
                                            node_num=node_num,
                                            device_num=device_num,
                                            resume=resume,
                                            job_name=job_name,
                                            pretrained_model_path=pretrained_model_path,
                                            backend=backend,
                                            device_type=device_type)

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
        model_name = None
        if ckpt_path is None and app_config.get('ckpt_path') is None:
            service_logger.error('param: ckpt_path is necessary.')
            raise ValueError

        return self.deploy_ma_training_task(cert,
                                            app_config,
                                            "evaluate",
                                            model_name=model_name,
                                            model_config_path=model_config_path,
                                            data_path=data_path,
                                            output_path=output_path,
                                            node_num=node_num,
                                            device_num=device_num,
                                            ckpt_path=ckpt_path,
                                            resume=False,
                                            job_name=job_name,
                                            pretrained_model_path=None,
                                            backend='mindspore',
                                            device_type='npu')

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
        model_name = None
        if ckpt_path is None and app_config.get('ckpt_path') is None:
            service_logger.error('param: ckpt_path is necessary.')
            raise ValueError
        return self.deploy_ma_training_task(cert,
                                            app_config,
                                            "infer",
                                            model_name=model_name,
                                            model_config_path=model_config_path,
                                            data_path=data_path,
                                            output_path=output_path,
                                            node_num=node_num,
                                            device_num=device_num,
                                            ckpt_path=ckpt_path,
                                            resume=False,
                                            job_name=job_name,
                                            pretrained_model_path=None,
                                            backend='mindspore',
                                            device_type='npu')

    def publish(self,
                cert,
                app_config,
                **kwargs):
        model_version, model_path = kwargs.get("model_version"), kwargs.get("model_path")
        if model_version is None:
            service_logger.error('param: model_version is necessary.')
            raise ValueError
        if model_path is None:
            service_logger.error('param: model_path is necessary.')
            raise ValueError

        session = get_session(cert, app_config)
        ma_infer_params = MAInferParams(app_config,
                                        model_name=kwargs.get("model_name"),
                                        model_version=model_version,
                                        model_path=model_path)
        model = self.model_module.Model(session,
                                        model_name=ma_infer_params.model_name,
                                        runtime=os.path.join(
                                            remove_endpoint_protocol_prefix(app_config.get('swr_endpoint')),
                                            app_config.get('user_image_url')),
                                        model_version=ma_infer_params.model_version,
                                        source_location=ma_infer_params.model_path,
                                        model_type="Custom",
                                        prebuild="true")
        service_logger.info(f'model name    : {ma_infer_params.model_name}')
        service_logger.info(f'model version : {ma_infer_params.model_version}')
        service_logger.info(f'model id      : {model.model_id}')

        model_status = model.get_model_info().get("model_status")
        if model_status != "published":
            raise RuntimeError(f"Failed to publish model ,the status is {str(model_status)}")
        return model.model_id

    def deploy(self,
               cert,
               app_config,
               **kwargs):
        model_id = kwargs.get("model_id")
        if model_id is None:
            service_logger.error('param: model_id is necessary.')
            raise ValueError

        session = get_session(cert, app_config)
        ma_infer_params = MAInferParams(app_config,
                                        node_num=kwargs.get("node_num"),
                                        device_num=kwargs.get("device_num"),
                                        service_name=kwargs.get("service_name"),
                                        model_id=model_id)
        configs = [self.config_module.model_config.ServiceConfig(
            ma_infer_params.model_id,
            weight="100",
            instance_count=ma_infer_params.ma_common_params.node_num,
            specification=ma_infer_params.specification)]
        model = self.model_module.Model(session, model_id=ma_infer_params.model_id)
        service_logger.info(f'node_num      : {ma_infer_params.ma_common_params.node_num}')
        service_logger.info(f'device_num    : {ma_infer_params.ma_common_params.device_num}')
        predictor = model.deploy_predictor(service_name=ma_infer_params.service_name,
                                           infer_type="real-time",
                                           configs=configs,
                                           cluster_id=ma_infer_params.cluster_id)

        service_logger.info(f'service name  : {ma_infer_params.service_name}')
        service_logger.info(f'service id    : {predictor.service_id}')
        return predictor.service_id, predictor.get_service_info()["access_address"]

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

    def stop(self,
             cert,
             app_config,
             job_id):
        """
            功能: 停止训练作业
        """
        session = get_session(cert, app_config)
        self.estimator_module.Estimator.control_job_by_id(session=session, job_id=job_id)

    def delete(self,
               cert,
               app_config,
               instance_id,
               instance_type):
        """
            功能: 删除MA实例
        """
        session = get_session(cert, app_config)
        if instance_type == "job":
            self.estimator_module.Estimator.delete_job_by_id(session=session, job_id=instance_id)
        if instance_type == "model":
            self.delete_model(session, instance_id)
        if instance_type == "service":
            predictor = self.model_module.Predictor(session, service_id=instance_id)
            predictor.delete_service()

    # 当前 MA SDK 的给出的模型删除接口不会在失败时抛出异常
    def delete_model(self, session, model_id):
        model = self.model_module.Model(session, model_id=model_id)
        resp = model.model_instance.delete_model(model_id=model_id)
        if len(resp['delete_success_list']) == 1:
            service_logger.info("Successfully delete the model %s ." % model_id)
        else:
            service_logger.error(resp['delete_failed_list'][0]['error_msg'])
            raise RuntimeError

    def get_job_info(self,
                     cert,
                     app_config,
                     job_id=None,
                     job_num=20,
                     get_all=False):
        """
            功能：查询单个或多个job信息
            参数：
                job_id：任务id
                cert: 认证信息
                app_config: 应用配置
                job_num: 查询的任务数
                get_all: 是否查询所有任务
            返回值：
                单个job信息或job信息列表
        """
        session = get_session(cert, app_config)
        if job_id:
            return self.estimator_module.Estimator(session=session, job_id=job_id).get_job_info()
        else:
            output_list = []
            if not get_all:
                per_page = min(job_num, 50)
                page_list = range(1 + job_num // 50)
                for page in page_list:
                    job_list_info = self.estimator_module.Estimator.get_job_list(session=session,
                                                                                 offset=per_page * page,
                                                                                 limit=per_page,
                                                                                 sort_by='create_time',
                                                                                 order="desc")
                    output_list += job_list_info['items']

            else:
                per_page = 50
                page = 0
                while True:
                    job_list_info = self.estimator_module.Estimator.get_job_list(session=session,
                                                                                 offset=per_page * page,
                                                                                 limit=per_page,
                                                                                 sort_by='create_time',
                                                                                 order="desc")
                    if job_list_info['items']:
                        output_list += job_list_info['items']
                        page += 1
                    else:
                        break
            return output_list

    def get_model_info(self,
                       cert,
                       app_config,
                       model_id=None,
                       model_num=20,
                       get_all=False):
        session = get_session(cert, app_config)
        if model_id:
            return self.model_module.Model(session=session, model_id=model_id).get_model_info()
        else:
            output_list = []
            if not get_all:
                per_page = min(model_num, 50)
                page_list = range(1 + model_num // 50)
                for page in page_list:
                    model_list_info = self.model_module.Model.get_model_list(session=session,
                                                                             offset=per_page * page,
                                                                             limit=per_page,
                                                                             sort_by='create_at',
                                                                             order="desc")
                    output_list += model_list_info['models']

            else:
                per_page = 50
                page = 0
                while True:
                    model_list_info = self.model_module.Model.get_model_list(session=session,
                                                                             offset=per_page * page,
                                                                             limit=per_page,
                                                                             sort_by='create_at',
                                                                             order="desc")
                    if model_list_info['models']:
                        output_list += model_list_info['models']
                        page += 1
                    else:
                        break
            return output_list

    def get_service_info(self,
                         cert,
                         app_config,
                         service_id=None,
                         service_num=20,
                         get_all=False):
        session = get_session(cert, app_config)
        if service_id:
            return self.model_module.Predictor(session=session, service_id=service_id).get_service_info()
        else:
            output_list = []
            if not get_all:
                per_page = min(service_num, 50)
                page_list = range(1 + service_num // 50)
                for page in page_list:
                    service_list_info = self.model_module.Predictor.get_service_list(session,
                                                                                     offset=per_page * page,
                                                                                     limit=per_page,
                                                                                     infer_type='real-time',
                                                                                     sort_by='publish_at',
                                                                                     order="desc")
                    output_list += service_list_info['services']

            else:
                per_page = 50
                page = 0
                while True:
                    service_list_info = self.model_module.Predictor.get_service_list(session,
                                                                                     offset=per_page * page,
                                                                                     limit=per_page,
                                                                                     infer_type='real-time',
                                                                                     sort_by='publish_at',
                                                                                     order="desc")
                    if service_list_info['services']:
                        output_list += service_list_info['services']
                        page += 1
                    else:
                        break
            return output_list

    def show(self,
             cert,
             app_config,
             instance_num=20,
             instance_id=None,
             instance_type="job",
             get_all=False):
        """
            功能：获取MA实例列表
            参数：
                cert: 认证信息
                app_config: 应用配置
            返回值：
                -1：接口调用失败
                其他值： job执行时长，单位为毫秒
        """
        if instance_type == "job":
            return self.get_job_info(cert, app_config, instance_id), self.scenario
        elif instance_type == "model":
            return self.get_model_info(cert, app_config, instance_id), self.scenario
        elif instance_type == "service":
            return self.get_service_info(cert, app_config, instance_id), self.scenario
        else:
            raise ValueError(f"Instance type {instance_type} is not valid.")

    def jobs_check(self, cert, app_config):
        job_lists, _ = self.show(cert, app_config, get_all=True)
        return len([j for j in job_lists if j.get('status').get('phase') not in JOB_STATUS]) < self.max_jobs

    def query_duration(self,
                       cert,
                       app_config,
                       job_id):
        """
            功能：查询job执行时长，便于资源拥有者设计计费模式
            参数：
                job_id： job状态为完成的任务id
                cert: 认证信息
                app_config: 应用配置
            返回值：
                -1：接口调用失败
                其他值： job执行时长，单位为毫秒
        """
        session = get_session(cert, app_config)
        job_info = self.estimator_module.Estimator(session=session, job_id=job_id).get_job_info()

        if not job_info:
            return -1

        return job_info['status']['duration']

    def get_status(self,
                   cert,
                   app_config,
                   instance_id,
                   instance_type):
        """
            功能：查询MA实例运行状态
            参数：
                cert: 认证信息
                app_config: 应用配置
                instance_id: ma实例id
                instance_type: ma实例类型
            返回值：
                -1: 调用接口失败
                其他: ma实例状态, 详细查看modelarts文档
        """
        if not instance_id:
            raise ValueError(f"Param {instance_type}_id is required")

        session = get_session(cert, app_config)
        status = None
        if instance_type == "job":
            job_info = self.estimator_module.Estimator(session=session, job_id=instance_id).get_job_info()
            if not job_info:
                return -1
            status = job_info['status']['phase']
        if instance_type == "model":
            model_info = self.model_module.Model(session=session, model_id=instance_id).get_model_info()
            if not model_info:
                return -1
            status = model_info['model_status']
        if instance_type == "service":
            service_info = self.model_module.Predictor(session=session, service_id=instance_id).get_service_info()
            if not service_info:
                return -1
            status = service_info['status']
        service_logger.info(f'get {instance_type} status: {status}')
        return status

    def test(self, *argv, **kwargs):
        pass
