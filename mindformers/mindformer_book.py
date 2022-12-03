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

'''
MindFormerBook class,
which contains the lists of models, pipelines, tasks, and default settings in MindFormer repository.
'''
import os
from collections import OrderedDict

from mindformers.tools import logger


def print_dict(input_dict):
    '''
    Print dict function for show support method of MindFormerBook or other BaseClasses.

    Args:
        input_dict (dict): the dict to be printed.
    '''
    if isinstance(input_dict, dict):
        for key, val in input_dict.items():
            if isinstance(val, dict):
                logger.info("%s:", key)
                print_dict(val)
            else:
                logger.info("   %s : %s", key, val)
        logger.info("-------------------------------------")
    else:
        raise TypeError(f"{type(input_dict)} is unspoorted by print_dict")


def print_path_or_list(input_path_or_list):
    '''
    Print path or list function for show support method of MindFormerBook or other BaseClasses.

    Args:
        input_path_or_list (str, list): the path or list to be printed.
    '''
    if isinstance(input_path_or_list, (str, list)):
        logger.info("   %s", input_path_or_list)
        logger.info("-------------------------------------")
    else:
        raise TypeError(f"{type(input_path_or_list)} is unsupported by print_path_or_list")


class MindFormerBook:
    '''
    MindFormerBook class,
    which contains the lists of models, pipelines, tasks, and default
    settings in MindFormer repository
    '''
    _PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _DEFAULT_CHECKPOINT_DOWNLOAD_FOLDER = os.path.join(_PROJECT_PATH, 'checkpoint_download')
    _DEFAULT_CHECKPOINT_SAVE_FOLDER = os.path.join(_PROJECT_PATH, 'checkpoint_save')

    _TRAINER_SUPPORT_TASKS_LIST = OrderedDict([
        ("masked_image_modeling", OrderedDict([
            ("mae_vit_base_p16", os.path.join(
                _PROJECT_PATH, "configs/mae/run_mae_vit_base_p16_224_800ep.yaml")),
            ("common", os.path.join(
                _PROJECT_PATH, "configs/mae/run_mae_vit_base_p16_224_800ep.yaml"))])
         )
    ])

    _PIPELINE_SUPPORT_TASK_LIST = OrderedDict([
        ('zero_shot_image_classification', [
            'clip_vit_b_32',
        ])
    ])

    _MODEL_SUPPORT_LIST = OrderedDict([
        ('clip', [
            'clip_vit_b_32',
        ]),
        ('mae', [
            'mae_vit_base_p16',
        ])
    ])

    _MODEL_CKPT_URL_LIST = OrderedDict([
        ('clip_vit_b_32',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/clip/clip_vit_b_32.ckpt'
          ])
    ])

    _MODEL_CONFIG_URL_LIST = OrderedDict([
        ('clip_vit_b_32',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/clip/clip_vit_b_32.yaml'
          ])
    ])

    @classmethod
    def show_trainer_support_task_list(cls):
        '''show_trainer_support_task_list function'''
        logger.info("_TRAINER_SUPPORT_TASKS_LIST of MindFormer is: ")
        print_dict(cls._TRAINER_SUPPORT_TASKS_LIST)

    @classmethod
    def get_trainer_support_task_list(cls):
        '''get_trainer_support_task_list function'''
        return cls._TRAINER_SUPPORT_TASKS_LIST

    @classmethod
    def show_pipeline_support_task_list(cls):
        '''show_pipeline_support_task_list function'''
        logger.info("_PIPELINE_SUPPORT_TASK_LIST of MindFormer is: ")
        print_dict(cls._PIPELINE_SUPPORT_TASK_LIST)

    @classmethod
    def get_pipeline_support_task_list(cls):
        '''get_pipeline_support_task_list function'''
        return cls._PIPELINE_SUPPORT_TASK_LIST

    @classmethod
    def show_model_config_url_list(cls):
        '''show_model_config_url_list function'''
        logger.info("_MODEL_CONFIG_URL_LIST of MindFormer is: ")
        print_dict(cls._MODEL_CONFIG_URL_LIST)

    @classmethod
    def get_model_config_url_list(cls):
        '''get_model_config_url_list function'''
        return cls._MODEL_CONFIG_URL_LIST

    @classmethod
    def show_project_path(cls):
        '''show_project_path function'''
        logger.info("_PROJECT_PATH of MindFormer is: ")
        print_path_or_list(cls._PROJECT_PATH)

    @classmethod
    def get_project_path(cls):
        '''get_project_path function'''
        return cls._PROJECT_PATH

    @classmethod
    def show_default_checkpoint_download_folder(cls):
        '''show_default_checkpoint_download_folder function'''
        logger.info("_DEFAULT_CHECKPOINT_DOWNLOAD_FOLDER of MindFormer is: ")
        print_path_or_list(cls._DEFAULT_CHECKPOINT_DOWNLOAD_FOLDER)

    @classmethod
    def get_default_checkpoint_download_folder(cls):
        '''get_default_checkpoint_download_folder function'''
        return cls._DEFAULT_CHECKPOINT_DOWNLOAD_FOLDER

    @classmethod
    def set_default_checkpoint_download_folder(cls, download_folder):
        '''set_default_checkpoint_download_folder function'''
        if not os.path.isdir(download_folder):
            raise TypeError(f"{download_folder} should be a directory.")
        cls._DEFAULT_CHECKPOINT_DOWNLOAD_FOLDER = download_folder

    @classmethod
    def get_default_checkpoint_save_folder(cls):
        '''get_default_checkpoint_save_folder function'''
        return cls._DEFAULT_CHECKPOINT_SAVE_FOLDER

    @classmethod
    def set_default_checkpoint_save_folder(cls, save_folder):
        '''set_default_checkpoint_save_folder function'''
        if not os.path.isdir(save_folder):
            raise TypeError(f"{save_folder} should be a directory.")
        cls._DEFAULT_CHECKPOINT_SAVE_FOLDER = save_folder

    @classmethod
    def show_default_checkpoint_save_folder(cls):
        '''show_default_checkpoint_save_folder function'''
        logger.info("_DEFAULT_CHECKPOINT_SAVE_FOLDER of MindFormer is: ")
        print_path_or_list(cls._DEFAULT_CHECKPOINT_SAVE_FOLDER)

    @classmethod
    def show_model_support_list(cls):
        '''show_model_support_list function'''
        logger.info("MODEL_SUPPORT_LIST of MindFormer is: ")
        print_dict(cls._MODEL_SUPPORT_LIST)

    @classmethod
    def get_model_support_list(cls):
        '''get_model_support_list function'''
        return cls._MODEL_SUPPORT_LIST

    @classmethod
    def get_model_name_support_list(cls):
        support_model_name = []
        for task_name in cls._TRAINER_SUPPORT_TASKS_LIST.keys():
            support_model_name.extend(cls._TRAINER_SUPPORT_TASKS_LIST.get(task_name).keys())
        return set(support_model_name)

    @classmethod
    def show_model_ckpt_url_list(cls):
        '''show_model_ckpt_url_list function'''
        logger.info("MODEL_CKPT_URL_LIST of MindFormer is: ")
        print_dict(cls._MODEL_CKPT_URL_LIST)

    @classmethod
    def get_model_ckpt_url_list(cls):
        '''get_model_ckpt_url_list function'''
        return cls._MODEL_CKPT_URL_LIST
