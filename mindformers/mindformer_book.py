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
MindFormerBook class,
which contains the lists of models, pipelines, tasks, and default settings in MindFormer repository.
"""
import os
import copy
from collections import OrderedDict

from mindformers.tools.logger import logger


def print_dict(input_dict):
    """
    Print dict function for show support method of MindFormerBook or other BaseClasses.

    Args:
        input_dict (dict): the dict to be printed.
    """
    if isinstance(input_dict, dict):
        for key, val in input_dict.items():
            if isinstance(val, dict):
                logger.info("%s:", key)
                print_dict(val)
            else:
                logger.info("   %s : %s", key, val)
    else:
        raise TypeError(f"{type(input_dict)} is unspoorted by print_dict")


def print_path_or_list(input_path_or_list):
    """
    Print path or list function for show support method of MindFormerBook or other BaseClasses.

    Args:
        input_path_or_list (str, list): the path or list to be printed.
    """
    if isinstance(input_path_or_list, (str, list)):
        logger.info("   %s", input_path_or_list)
        logger.info("-------------------------------------")
    else:
        raise TypeError(f"{type(input_path_or_list)} is unsupported by print_path_or_list")


class MindFormerBook:
    """
    MindFormerBook class,
    which contains the lists of models, pipelines, tasks, and default
    settings in MindFormer repository
    When adding a new Model or assemble in this project, the following constants list and dict need adding.

    Examples:
        >>> from mindformers.mindformer_book import MindFormerBook
        >>>
        >>> # 1) Fill the following constant list and dict in this class
        >>> # 2) Overwrite the support_list when define a new Model or Pipeline.
        >>> @MindFormerRegister.register(MindFormerModuleType.MODELS)
        ... class ViTForImageClassification(PreTrainedModel):
        ...     _support_list = MindFormerBook.get_model_support_list()['vit']
        >>> # 3) Then you can use auto class and from pretrain to init an instance.
        >>> vit_model = AutoModel.from_pretrained('vit_base_p16')
    """
    _PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _DEFAULT_CHECKPOINT_DOWNLOAD_FOLDER = os.getenv("CHECKPOINT_DOWNLOAD_FOLDER",
                                                    os.path.join('.', 'checkpoint_download'))
    _DEFAULT_CHECKPOINT_SAVE_FOLDER = os.getenv("CHECKPOINT_SAVE_FOLDER", os.path.join('.', 'checkpoint_save'))

    _TRAINER_SUPPORT_TASKS_LIST = OrderedDict([
        ("general", OrderedDict([
            ("common", os.path.join(
                _PROJECT_PATH, "configs/general/run_general_task.yaml"))])
         ),
        ("text_generation", OrderedDict([
            ("glm4_9b", os.path.join(
                _PROJECT_PATH, "configs/glm4/predict_glm4_9b_chat.yaml")),
            ("common", os.path.join(
                _PROJECT_PATH, "configs/gpt2/run_gpt2.yaml"))])
         )
    ])

    _PIPELINE_SUPPORT_TASK_LIST = OrderedDict([
        ("text_generation", OrderedDict([
            ("glm4_9b", os.path.join(
                _PROJECT_PATH, "configs/glm4/predict_glm4_9b_chat.yaml")),
            ("common", os.path.join(
                _PROJECT_PATH, "configs/gpt2/run_gpt2.yaml"))
        ]))
    ])

    _CONFIG_SUPPORT_LIST = OrderedDict([
        ('glm4', [
            'glm4_9b',
        ]),
        ('deepseekv3', [
            'deepseekv3_671b',
        ])
    ])

    _MODEL_SUPPORT_LIST = OrderedDict([
        ('glm4', [
            'glm4_9b',
        ])
    ])

    _PROCESSOR_SUPPORT_LIST = OrderedDict([
        ('glm4', [
            'glm4_9b',
        ])
    ])

    _TOKENIZER_SUPPORT_LIST = OrderedDict([
        ('glm4', [
            'glm4_9b',
        ])
    ])

    _MODEL_CONFIG_TO_NAME = OrderedDict([
    ])

    _MODEL_CKPT_URL_LIST = OrderedDict([
    ])

    _MODEL_CONFIG_URL_LIST = OrderedDict([
    ])

    _PIPELINE_INPUT_DATA_LIST = OrderedDict([
    ])

    _TOKENIZER_SUPPORT_URL_LIST = OrderedDict([
    ])

    _TOKENIZER_NAME_TO_PROCESSOR = OrderedDict([
        ('ChatGLM4Tokenizer', 'GLMProcessor')
    ])

    @classmethod
    def show_trainer_support_model_list(cls, task=None):
        """show_trainer_support_model_list"""
        all_list = copy.deepcopy(cls._TRAINER_SUPPORT_TASKS_LIST)
        all_list.pop("general")
        for key, val in all_list.items():
            val.pop("common")
            temp_list = []
            for model_name, _ in val.items():
                temp_list.append(model_name)
            all_list[key] = temp_list

        if task:
            if task in all_list.keys():
                all_list = all_list[task]
                logger.info("Trainer support model list for %s task is: ", str(task))
                print_path_or_list(all_list)
            else:
                raise KeyError("unsupported task")
        else:
            logger.info("Trainer support model list of MindFormer is: ")
            print_dict(all_list)

    @classmethod
    def show_pipeline_support_model_list(cls, task=None):
        """show_pipeline_support_model_list"""
        all_list = copy.deepcopy(cls._PIPELINE_SUPPORT_TASK_LIST)
        for key, val in all_list.items():
            val.pop("common")
            temp_list = []
            for model_name, _ in val.items():
                temp_list.append(model_name)
            all_list[key] = temp_list

        if task:
            if task in all_list.keys():
                all_list = all_list[task]
                logger.info("Pipeline support model list for %s task is: ", str(task))
                print_path_or_list(all_list)
            else:
                raise KeyError("unsupported task")
        else:
            logger.info("Pipeline support model list of MindFormer is: ")
            print_dict(all_list)

    @classmethod
    def show_tokenizer_name_to_processor(cls):
        """show_tokenizer_name_to_processor function"""
        logger.info("_TRAINER_SUPPORT_TASKS_LIST of MindFormer is: ")
        print_dict(cls._TOKENIZER_NAME_TO_PROCESSOR)

    @classmethod
    def get_tokenizer_name_to_processor(cls):
        """get_tokenizer_name_to_processor function"""
        return cls._TOKENIZER_NAME_TO_PROCESSOR

    @classmethod
    def show_trainer_support_task_list(cls):
        """show_trainer_support_task_list function"""
        logger.info("_TRAINER_SUPPORT_TASKS_LIST of MindFormer is: ")
        print_dict(cls._TRAINER_SUPPORT_TASKS_LIST)

    @classmethod
    def show_pipeline_support_input_data_list(cls):
        """show_pipeline_support_input_data_list function"""
        logger.info("_PIPELINE_INPUT_DATA_LIST of MindFormer is: ")
        print_dict(cls._PIPELINE_INPUT_DATA_LIST)

    @classmethod
    def get_pipeline_support_input_data_list(cls):
        """get_pipeline_support_input_data_list function"""
        return cls._PIPELINE_INPUT_DATA_LIST

    @classmethod
    def get_trainer_support_task_list(cls):
        """get_trainer_support_task_list function"""
        return cls._TRAINER_SUPPORT_TASKS_LIST

    @classmethod
    def show_pipeline_support_task_list(cls):
        """show_pipeline_support_task_list function"""
        logger.info("_PIPELINE_SUPPORT_TASK_LIST of MindFormer is: ")
        print_dict(cls._PIPELINE_SUPPORT_TASK_LIST)

    @classmethod
    def get_pipeline_support_task_list(cls):
        """get_pipeline_support_task_list function"""
        return cls._PIPELINE_SUPPORT_TASK_LIST

    @classmethod
    def show_model_config_to_name(cls):
        """show_model_config_to_name function"""
        print_dict(cls._MODEL_CONFIG_TO_NAME)

    @classmethod
    def get_model_config_to_name(cls):
        """show_model_config_to_name function"""
        return cls._MODEL_CONFIG_TO_NAME

    @classmethod
    def set_model_config_to_name(cls, model_config, model_name):
        """
        set_model_config_to_name function
        Args:
            model_config (str): the name of model config
            model_name (str): the name of model
        """
        cls._MODEL_CONFIG_TO_NAME.update({
            model_config: model_name
        })

    @classmethod
    def show_model_config_url_list(cls):
        """show_model_config_url_list function"""
        logger.info("_MODEL_CONFIG_URL_LIST of MindFormer is: ")
        print_dict(cls._MODEL_CONFIG_URL_LIST)

    @classmethod
    def get_model_config_url_list(cls):
        """get_model_config_url_list function"""
        return cls._MODEL_CONFIG_URL_LIST

    @classmethod
    def show_project_path(cls):
        """show_project_path function"""
        logger.info("_PROJECT_PATH of MindFormer is: ")
        print_path_or_list(cls._PROJECT_PATH)

    @classmethod
    def get_project_path(cls):
        """get_project_path function"""
        return cls._PROJECT_PATH

    @classmethod
    def get_xihe_checkpoint_download_folder(cls):
        """get xihe's mindspore checkpoint download folder."""
        return os.path.join(cls._DEFAULT_CHECKPOINT_DOWNLOAD_FOLDER, 'mindspore')

    @classmethod
    def show_default_checkpoint_download_folder(cls):
        """show_default_checkpoint_download_folder function"""
        logger.info("_DEFAULT_CHECKPOINT_DOWNLOAD_FOLDER of MindFormer is: ")
        print_path_or_list(cls._DEFAULT_CHECKPOINT_DOWNLOAD_FOLDER)

    @classmethod
    def get_default_checkpoint_download_folder(cls):
        """get_default_checkpoint_download_folder function"""
        return cls._DEFAULT_CHECKPOINT_DOWNLOAD_FOLDER

    @classmethod
    def set_default_checkpoint_download_folder(cls, download_folder):
        """
        set_default_checkpoint_download_folder function
        Args:
            download_folder (str): the path of default checkpoint download folder
        """
        if not os.path.isdir(download_folder):
            raise TypeError(f"{download_folder} should be a directory.")
        cls._DEFAULT_CHECKPOINT_DOWNLOAD_FOLDER = download_folder

    @classmethod
    def get_default_checkpoint_save_folder(cls):
        """get_default_checkpoint_save_folder function"""
        return cls._DEFAULT_CHECKPOINT_SAVE_FOLDER

    @classmethod
    def set_default_checkpoint_save_folder(cls, save_folder):
        """
        set_default_checkpoint_save_folder function
        Args:
            save_folder (str): the path of default checkpoint save folder
        """
        if not os.path.isdir(save_folder):
            raise TypeError(f"{save_folder} should be a directory.")
        cls._DEFAULT_CHECKPOINT_SAVE_FOLDER = save_folder

    @classmethod
    def show_default_checkpoint_save_folder(cls):
        """show_default_checkpoint_save_folder function"""
        logger.info("_DEFAULT_CHECKPOINT_SAVE_FOLDER of MindFormer is: ")
        print_path_or_list(cls._DEFAULT_CHECKPOINT_SAVE_FOLDER)

    @classmethod
    def show_config_support_list(cls):
        """show_config_support_list function"""
        logger.info("CONFIG_SUPPORT_LIST of MindFormer is: ")
        print_dict(cls._CONFIG_SUPPORT_LIST)

    @classmethod
    def get_config_support_list(cls):
        """get_config_support_list function"""
        return cls._CONFIG_SUPPORT_LIST

    @classmethod
    def show_model_support_list(cls):
        """show_model_support_list function"""
        logger.info("MODEL_SUPPORT_LIST of MindFormer is: ")
        print_dict(cls._MODEL_SUPPORT_LIST)

    @classmethod
    def get_model_support_list(cls):
        """get_model_support_list function"""
        return cls._MODEL_SUPPORT_LIST

    @classmethod
    def show_processor_support_list(cls):
        """show_processor_support_list function"""
        logger.info("PROCESSOR_SUPPORT_LIST of MindFormer is: ")
        print_dict(cls._PROCESSOR_SUPPORT_LIST)

    @classmethod
    def get_processor_support_list(cls):
        """get_processor_support_list function"""
        return cls._PROCESSOR_SUPPORT_LIST

    @classmethod
    def get_tokenizer_support_list(cls):
        """get_tokenizer_support_list function"""
        return cls._TOKENIZER_SUPPORT_LIST

    @classmethod
    def show_tokenizer_support_list(cls):
        """show_tokenizer_support_list function"""
        print_dict(cls._TOKENIZER_SUPPORT_LIST)

    @classmethod
    def get_tokenizer_url_support_list(cls):
        """get_tokenizer_url_support_list function"""
        return cls._TOKENIZER_SUPPORT_URL_LIST

    @classmethod
    def show_tokenizer_url_support_list(cls):
        """show_tokenizer_url_support_list function"""
        print_dict(cls._TOKENIZER_SUPPORT_URL_LIST)

    @classmethod
    def get_model_name_support_list(cls):
        """get_model_name_support_list"""
        support_model_name = []
        for task_name in cls._TRAINER_SUPPORT_TASKS_LIST.keys():
            support_model_name.extend(cls._TRAINER_SUPPORT_TASKS_LIST.get(task_name).keys())
        return set(support_model_name)

    @classmethod
    def get_model_name_support_list_for_task(cls, task_name):
        """get_model_name_support_list"""
        support_model_name = cls._TRAINER_SUPPORT_TASKS_LIST.get(task_name).keys()
        return set(support_model_name)

    @classmethod
    def show_model_ckpt_url_list(cls):
        """show_model_ckpt_url_list function"""
        logger.info("MODEL_CKPT_URL_LIST of MindFormer is: ")
        print_dict(cls._MODEL_CKPT_URL_LIST)

    @classmethod
    def get_model_ckpt_url_list(cls):
        """get_model_ckpt_url_list function"""
        return cls._MODEL_CKPT_URL_LIST

    @classmethod
    def get_downloadable_model_name_list(cls):
        """get downloadable model name list"""
        return set(cls._MODEL_CKPT_URL_LIST.keys())
