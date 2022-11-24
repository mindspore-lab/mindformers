
import os
from collections import OrderedDict

class XFormerBook:

    _PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
    _DEFAULT_CHECKPOINT_DOWNLOAD_FOLDER = os.path.join(_PROJECT_PATH, 'checkpoint_download')
    _DEFAULT_CHECKPOINT_SAVE_FOLDER = os.path.join(_PROJECT_PATH, 'checkpoint_save')

    _TRAINER_SUPPORT_TASKS_LIST = OrderedDict([
        ("masked_image_modeling", OrderedDict([
            ("mae_vit_base_p16", "./configs/mae/run_mae_vit_base_p16_224_800ep.yaml"),
            ("simmim_vit_base_p16", "./configs/simmim/run_simmim_vit_base_p16_224_800ep.yaml"),
            ("simmim_swin_base_w6_p4", "./configs/simmim/run_simmim_swin_base_w6_p4_192_800ep.yaml"),
            ("common", "./configs/mae/run_mae_vit_base_p16_224_800ep.yaml")])
         ),
        ("image_classification", OrderedDict([
            ("vit_base_p16", "./configs/vit/run_vit_base_p16_224_100ep.yaml"),
            ("common", "./configs/vit/run_vit_base_p16_224_100ep.yaml")])
         ),
        ("image_segmentation", OrderedDict([
            ("upernet_vit_base_p16", "./configs/upernet/run_upernet_vit_base_p16_224_100ep.yaml"),
            ("common", "./configs/upernet/run_upernet_vit_base_p16_224_100ep.yaml")])
        )
    ])

    _PIPELINE_SUPPORT_TASK_LIST = OrderedDict([
        ('zero_shot_image_classification', [
            'clip_vit_b_16',
            'clip_vit_b_32',
            'clip_vit_l_14',
            'clip_vit_l_14@336'
        ]),
    ])

    _MODEL_SUPPORT_LIST = OrderedDict([
        ('clip', [
            'clip_vit_b_16',    # 键"clip" 与 值 “clip_vit_b_16”的首单词保持一直
            'clip_vit_b_32',
            'clip_vit_l_14',
            'clip_vit_l_14@336'
        ]),
    ])

    _MODEL_CKPT_URL_LIST = OrderedDict([
        ('clip_vit_b_16', ['https://xxxxxxx']),
        ('clip_vit_b_32', ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/clip_vit_b_32.ckpt']),
        ('clip_vit_l_14', ['https://xxxxxx']),
        ('clip_vit_l_14@336', ['https://xxxxxxxx']),
    ])

    _MODEL_CONFIG_URL_LIST = OrderedDict([
        ('clip_vit_b_16', ['https://xxxxxx']),
        ('clip_vit_b_32', ['https://xxxxxxxx']),
        ('clip_vit_l_14', ['https://xxxxxxxx']),
        ('clip_vit_l_14@336', ['https://xxxxxxxx']),
    ])

    @classmethod
    def show_trainer_support_task_list(cls):
        print("_TRAINER_SUPPORT_TASKS_LIST of XFormer is: ")
        print(cls._TRAINER_SUPPORT_TASKS_LIST)

    @classmethod
    def get_trainer_support_task_list(cls):
        return cls._TRAINER_SUPPORT_TASKS_LIST

    @classmethod
    def show_pipeline_support_task_list(cls):
        print("_PIPELINE_SUPPORT_TASK_LIST of XFormer is: ")
        print(cls._PIPELINE_SUPPORT_TASK_LIST)

    @classmethod
    def get_pipeline_support_task_list(cls):
        return cls._PIPELINE_SUPPORT_TASK_LIST

    @classmethod
    def show_model_config_url_list(cls):
        print("_MODEL_CONFIG_URL_LIST of XFormer is: ")
        print(cls._MODEL_CONFIG_URL_LIST)

    @classmethod
    def get_model_config_url_list(cls):
        return cls._MODEL_CONFIG_URL_LIST

    @classmethod
    def show_project_path(cls):
        print("_PROJECT_PATH of XFormer is: ")
        print(cls._PROJECT_PATH)

    @classmethod
    def get_project_path(cls):
        return cls._PROJECT_PATH

    @classmethod
    def show_default_checkpoint_download_folder(cls):
        print("_DEFAULT_CHECKPOINT_DOWNLOAD_FOLDER of XFormer is: ")
        print(cls._DEFAULT_CHECKPOINT_DOWNLOAD_FOLDER)

    @classmethod
    def get_default_checkpoint_download_folder(cls):
        return cls._DEFAULT_CHECKPOINT_DOWNLOAD_FOLDER

    @classmethod
    def show_default_checkpoint_save_folder(cls):
        print("_DEFAULT_CHECKPOINT_SAVE_FOLDER of XFormer is: ")
        print(cls._DEFAULT_CHECKPOINT_SAVE_FOLDER)

    @classmethod
    def get_default_checkpoint_save_folder(cls):
        return cls._DEFAULT_CHECKPOINT_SAVE_FOLDER

    @classmethod
    def show_default_checkpoint_save_folder(cls):
        print("_DEFAULT_CHECKPOINT_SAVE_FOLDER of XFormer is: ")
        print(cls._DEFAULT_CHECKPOINT_SAVE_FOLDER)

    @classmethod
    def show_model_support_list(cls):
        print("MODEL_SUPPORT_LIST of XFormer is: ")
        for key, val in cls._MODEL_SUPPORT_LIST.items():
            print(key, ':', val)

    @classmethod
    def get_model_support_list(cls):
        return cls._MODEL_SUPPORT_LIST

    @classmethod
    def show_model_ckpt_url_list(cls):
        print("MODEL_CKPT_URL_LIST of XFormer is: ")
        for key, val in cls._MODEL_CKPT_URL_LIST.items():
            print(key, ':', val)

    @classmethod
    def get_model_ckpt_url_list(cls):
        return cls._MODEL_CKPT_URL_LIST
