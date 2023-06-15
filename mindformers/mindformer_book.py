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

from mindformers.tools import logger


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
        ... class ViTForImageClassification(BaseModel):
        ...     _support_list = MindFormerBook.get_model_support_list()['vit']
        >>> # 3) Then you can use auto class and from pretrain to init an instance.
        >>> vit_model = AutoModel.from_pretrained('vit_base_p16')
    """
    _PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _DEFAULT_CHECKPOINT_DOWNLOAD_FOLDER = os.path.join('.', 'checkpoint_download')
    _DEFAULT_CHECKPOINT_SAVE_FOLDER = os.path.join('.', 'checkpoint_save')

    _TRAINER_SUPPORT_TASKS_LIST = OrderedDict([
        ("general", OrderedDict([
            ("common", os.path.join(
                _PROJECT_PATH, "configs/general/run_general_task.yaml"))])
         ),
        ("masked_image_modeling", OrderedDict([
            ("mae_vit_base_p16", os.path.join(
                _PROJECT_PATH, "configs/mae/run_mae_vit_base_p16_224_800ep.yaml")),
            ("common", os.path.join(
                _PROJECT_PATH, "configs/mae/run_mae_vit_base_p16_224_800ep.yaml"))])
         ),
        ("image_classification", OrderedDict([
            ("vit_base_p16", os.path.join(
                _PROJECT_PATH, "configs/vit/run_vit_base_p16_224_100ep.yaml")),
            ("swin_base_p4w7", os.path.join(
                _PROJECT_PATH, "configs/swin/run_swin_base_p4w7_224_100ep.yaml")),
            ("mindspore/vit_base_p16", os.path.join(
                _PROJECT_PATH, "configs/vit/run_vit_base_p16_224_100ep.yaml")),
            ("mindspore/swin_base_p4w7", os.path.join(
                _PROJECT_PATH, "configs/swin/run_swin_base_p4w7_224_100ep.yaml")),
            ("common", os.path.join(
                _PROJECT_PATH, "configs/vit/run_vit_base_p16_224_100ep.yaml"))])
         ),
        ("fill_mask", OrderedDict([
            ("bert_base_uncased", os.path.join(
                _PROJECT_PATH, "configs/bert/run_bert_base_uncased.yaml")),
            ("bert_tiny_uncased", os.path.join(
                _PROJECT_PATH, "configs/bert/run_bert_tiny_uncased.yaml")),
            ("common", os.path.join(
                _PROJECT_PATH, "configs/bert/run_bert_tiny_uncased.yaml"))])
         ),
        ("contrastive_language_image_pretrain", OrderedDict([
            ("clip_vit_b_32", os.path.join(
                _PROJECT_PATH, "configs/clip/run_clip_vit_b_32_pretrain_flickr8k.yaml")),
            ("mindspore/clip_vit_b_32", os.path.join(
                _PROJECT_PATH, "configs/clip/run_clip_vit_b_32_pretrain_flickr8k.yaml")),
            ("clip_vit_b_16", os.path.join(
                _PROJECT_PATH, "configs/clip/run_clip_vit_b_16_pretrain_flickr8k.yaml")),
            ("clip_vit_l_14", os.path.join(
                _PROJECT_PATH, "configs/clip/run_clip_vit_l_14_pretrain_flickr8k.yaml")),
            ("clip_vit_l_14@336", os.path.join(
                _PROJECT_PATH, "configs/clip/run_clip_vit_l_14@336_pretrain_flickr8k.yaml")),
            ("common", os.path.join(
                _PROJECT_PATH, "configs/clip/run_clip_vit_b_32_pretrain_flickr8k.yaml"))])
         ),
        ("zero_shot_image_classification", OrderedDict([
            ("clip_vit_b_32", os.path.join(
                _PROJECT_PATH, "configs/clip/run_clip_vit_b_32_zero_shot_image_classification_cifar100.yaml")),
            ("mindspore/clip_vit_b_32", os.path.join(
                _PROJECT_PATH, "configs/clip/run_clip_vit_b_32_zero_shot_image_classification_cifar100.yaml")),
            ("clip_vit_b_16", os.path.join(
                _PROJECT_PATH, "configs/clip/run_clip_vit_b_16_zero_shot_image_classification_cifar100.yaml")),
            ("clip_vit_l_14", os.path.join(
                _PROJECT_PATH, "configs/clip/run_clip_vit_l_14_zero_shot_image_classification_cifar100.yaml")),
            ("clip_vit_l_14@336", os.path.join(
                _PROJECT_PATH, "configs/clip/run_clip_vit_l_14@336_zero_shot_image_classification_cifar100.yaml")),
            ("common", os.path.join(
                _PROJECT_PATH, "configs/clip/run_clip_vit_b_32_zero_shot_image_classification_cifar100.yaml"))])
         ),
        ("translation", OrderedDict([
            ("t5_small", os.path.join(
                _PROJECT_PATH, "configs/t5/run_t5_small_on_wmt16.yaml")),
            ("t5_tiny", os.path.join(
                _PROJECT_PATH, "configs/t5/run_t5_tiny_on_wmt16.yaml")),
            ("common", os.path.join(
                _PROJECT_PATH, "configs/t5/run_t5_small_on_wmt16.yaml"))])
         ),
        ("text_classification", OrderedDict([
            ("txtcls_bert_base_uncased", os.path.join(
                _PROJECT_PATH, "configs/txtcls/run_txtcls_bert_base_uncased.yaml")),
            ("txtcls_bert_base_uncased_mnli", os.path.join(
                _PROJECT_PATH, "configs/txtcls/run_txtcls_bert_base_uncased_mnli.yaml")),
            ("mindspore/txtcls_bert_base_uncased_mnli", os.path.join(
                _PROJECT_PATH, "configs/txtcls/run_txtcls_bert_base_uncased_mnli.yaml")),
            ("common", os.path.join(
                _PROJECT_PATH, "configs/txtcls/run_txtcls_bert_base_uncased.yaml"))])
         ),
        ("token_classification", OrderedDict([
            ("tokcls_bert_base_chinese", os.path.join(
                _PROJECT_PATH, "configs/tokcls/run_tokcls_bert_base_chinese.yaml")),
            ("tokcls_bert_base_chinese_cluener", os.path.join(
                _PROJECT_PATH, "configs/tokcls/run_tokcls_bert_base_chinese_cluener.yaml")),
            ("common", os.path.join(
                _PROJECT_PATH, "configs/tokcls/run_tokcls_bert_base_chinese.yaml"))])
         ),
        ("question_answering", OrderedDict([
            ("qa_bert_base_uncased", os.path.join(
                _PROJECT_PATH, "configs/qa/run_qa_bert_base_uncased.yaml")),
            ("qa_bert_base_uncased_squad", os.path.join(
                _PROJECT_PATH, "configs/qa/run_qa_bert_base_uncased.yaml")),
            ("mindspore/qa_bert_base_uncased", os.path.join(
                _PROJECT_PATH, "configs/qa/run_qa_bert_base_uncased.yaml")),
            ("common", os.path.join(
                _PROJECT_PATH, "configs/qa/run_qa_bert_base_uncased.yaml"))])
         ),
        ("text_generation", OrderedDict([
            ("gpt2", os.path.join(
                _PROJECT_PATH, "configs/gpt2/run_gpt2.yaml")),
            ("gpt2_13b", os.path.join(
                _PROJECT_PATH, "configs/gpt2/run_gpt2_13b.yaml")),
            ("gpt2_52b", os.path.join(
                _PROJECT_PATH, "configs/gpt2/run_gpt2_52b.yaml")),
            ("llama_7b", os.path.join(
                _PROJECT_PATH, "configs/llama/run_llama_7b.yaml")),
            ("llama_13b", os.path.join(
                _PROJECT_PATH, "configs/llama/run_llama_13b.yaml")),
            ("llama_65b", os.path.join(
                _PROJECT_PATH, "configs/llama/run_llama_65b.yaml")),
            ("llama_7b_lora", os.path.join(
                _PROJECT_PATH, "configs/llama/run_llama_7b_lora.yaml")),
            ("pangualpha_2_6b", os.path.join(
                _PROJECT_PATH, "configs/pangualpha/run_pangualpha_2_6b.yaml")),
            ("pangualpha_13b", os.path.join(
                _PROJECT_PATH, "configs/pangualpha/run_pangualpha_13b.yaml")),
            ("glm_6b", os.path.join(
                _PROJECT_PATH, "configs/glm/run_glm_6b_finetune.yaml")),
            ("glm_6b_chat", os.path.join(
                _PROJECT_PATH, "configs/glm/run_glm_6b_infer.yaml")),
            ("glm_6b_lora", os.path.join(
                _PROJECT_PATH, "configs/glm/run_glm_6b_lora.yaml")),
            ("glm_6b_lora_chat", os.path.join(
                _PROJECT_PATH, "configs/glm/run_glm_6b_lora_infer.yaml")),
            ("bloom_560m", os.path.join(
                _PROJECT_PATH, "configs/bloom/run_bloom_560m.yaml")),
            ("bloom_7.1b", os.path.join(
                _PROJECT_PATH, "configs/bloom/run_bloom_7.1b.yaml")),
            ("bloom_65b", os.path.join(
                _PROJECT_PATH, "configs/bloom/run_bloom_65b.yaml")),
            ("bloom_176b", os.path.join(
                _PROJECT_PATH, "configs/bloom/run_bloom_176b.yaml")),
            ("common", os.path.join(
                _PROJECT_PATH, "configs/gpt2/run_gpt2.yaml"))])
         )
    ])

    _PIPELINE_SUPPORT_TASK_LIST = OrderedDict([
        ('zero_shot_image_classification', OrderedDict([
            ('clip_vit_b_16', os.path.join(
                _PROJECT_PATH, "configs/clip/run_clip_vit_b_16_zero_shot_image_classification_cifar100.yaml")),
            ('mindspore/clip_vit_b_32', os.path.join(
                _PROJECT_PATH, "configs/clip/run_clip_vit_b_32_zero_shot_image_classification_cifar100.yaml")),
            ('clip_vit_b_32', os.path.join(
                _PROJECT_PATH, "configs/clip/run_clip_vit_b_32_zero_shot_image_classification_cifar100.yaml")),
            ('clip_vit_l_14', os.path.join(
                _PROJECT_PATH, "configs/clip/run_clip_vit_l_14_zero_shot_image_classification_cifar100.yaml")),
            ('clip_vit_l_14@336', os.path.join(
                _PROJECT_PATH, "configs/clip/run_clip_vit_l_14@336_zero_shot_image_classification_cifar100.yaml")),
            ('common', os.path.join(
                _PROJECT_PATH, "configs/clip/run_clip_vit_b_32_zero_shot_image_classification_cifar100.yaml"))
        ])),
        ('masked_image_modeling', OrderedDict([
            ('mae_vit_base_p16', os.path.join(
                _PROJECT_PATH, "configs/mae/model_config/mae_vit_base_p16.yaml")),
            ('common', os.path.join(
                _PROJECT_PATH, "configs/mae/model_config/mae_vit_base_p16.yaml"))
        ])),
        ('image_classification', OrderedDict([
            ('vit_base_p16', os.path.join(
                _PROJECT_PATH, "configs/vit/run_vit_base_p16_224_100ep.yaml")),
            ("swin_base_p4w7", os.path.join(
                _PROJECT_PATH, "configs/swin/run_swin_base_p4w7_224_100ep.yaml")),
            ('mindspore/vit_base_p16', os.path.join(
                _PROJECT_PATH, "configs/vit/run_vit_base_p16_224_100ep.yaml")),
            ("mindspore/swin_base_p4w7", os.path.join(
                _PROJECT_PATH, "configs/swin/run_swin_base_p4w7_224_100ep.yaml")),
            ('common', os.path.join(
                _PROJECT_PATH, "configs/vit/run_vit_base_p16_224_100ep.yaml"))
        ])),
        ('translation', OrderedDict([
            ('t5_small', os.path.join(
                _PROJECT_PATH, "configs/t5/run_t5_small_on_wmt16.yaml")),
            ('common', os.path.join(
                _PROJECT_PATH, "configs/t5/run_t5_small_on_wmt16.yaml"))
        ])),
        ("fill_mask", OrderedDict([
            ("bert_base_uncased", os.path.join(
                _PROJECT_PATH, "configs/bert/run_bert_base_uncased.yaml")),
            ("bert_tiny_uncased", os.path.join(
                _PROJECT_PATH, "configs/bert/run_bert_tiny_uncased.yaml")),
            ("common", os.path.join(
                _PROJECT_PATH, "configs/bert/run_bert_tiny_uncased.yaml"))
        ])),
        ("text_classification", OrderedDict([
            ("txtcls_bert_base_uncased", os.path.join(
                _PROJECT_PATH, "configs/txtcls/run_txtcls_bert_base_uncased.yaml")),
            ("txtcls_bert_base_uncased_mnli", os.path.join(
                _PROJECT_PATH, "configs/txtcls/run_txtcls_bert_base_uncased_mnli.yaml")),
            ("mindspore/txtcls_bert_base_uncased_mnli", os.path.join(
                _PROJECT_PATH, "configs/txtcls/run_txtcls_bert_base_uncased_mnli.yaml")),
            ("common", os.path.join(
                _PROJECT_PATH, "configs/txtcls/run_txtcls_bert_base_uncased.yaml"))
        ])),
        ("token_classification", OrderedDict([
            ("tokcls_bert_base_chinese", os.path.join(
                _PROJECT_PATH, "configs/tokcls/run_tokcls_bert_base_chinese.yaml")),
            ("tokcls_bert_base_chinese_cluener", os.path.join(
                _PROJECT_PATH, "configs/tokcls/run_tokcls_bert_base_chinese_cluener.yaml")),
            ("common", os.path.join(
                _PROJECT_PATH, "configs/tokcls/run_tokcls_bert_base_chinese.yaml"))
        ])),
        ("question_answering", OrderedDict([
            ("qa_bert_base_uncased", os.path.join(
                _PROJECT_PATH, "configs/qa/run_qa_bert_base_uncased.yaml")),
            ("common", os.path.join(
                _PROJECT_PATH, "configs/qa/run_qa_bert_base_uncased.yaml"))
        ])),
        ("text_generation", OrderedDict([
            ("gpt2", os.path.join(
                _PROJECT_PATH, "configs/gpt2/run_gpt2.yaml")),
            ("llama_7b", os.path.join(
                _PROJECT_PATH, "configs/llama/run_llama_7b.yaml")),
            ("llama_13b", os.path.join(
                _PROJECT_PATH, "configs/llama/run_llama_13b.yaml")),
            ("llama_65b", os.path.join(
                _PROJECT_PATH, "configs/llama/run_llama_65b.yaml")),
            ("llama_7b_lora", os.path.join(
                _PROJECT_PATH, "configs/llama/run_llama_7b_lora.yaml")),
            ("pangualpha_2_6b", os.path.join(
                _PROJECT_PATH, "configs/pangualpha/run_pangualpha_2_6b.yaml")),
            ("pangualpha_13b", os.path.join(
                _PROJECT_PATH, "configs/pangualpha/run_pangualpha_13b.yaml")),
            ("glm_6b", os.path.join(
                _PROJECT_PATH, "configs/glm/run_glm_6b_finetune.yaml")),
            ("glm_6b_chat", os.path.join(
                _PROJECT_PATH, "configs/glm/run_glm_6b_infer.yaml")),
            ("glm_6b_lora", os.path.join(
                _PROJECT_PATH, "configs/glm/run_glm_6b_lora.yaml")),
            ("glm_6b_lora_chat", os.path.join(
                _PROJECT_PATH, "configs/glm/run_glm_6b_lora_infer.yaml")),
            ("bloom_560m", os.path.join(
                _PROJECT_PATH, "configs/bloom/run_bloom_560m.yaml")),
            ("bloom_7.1b", os.path.join(
                _PROJECT_PATH, "configs/bloom/run_bloom_7.1b.yaml")),
            ("bloom_65b", os.path.join(
                _PROJECT_PATH, "configs/bloom/run_bloom_65b.yaml")),
            ("bloom_176b", os.path.join(
                _PROJECT_PATH, "configs/bloom/run_bloom_176b.yaml")),
            ("common", os.path.join(
                _PROJECT_PATH, "configs/gpt2/run_gpt2.yaml"))
        ]))
    ])

    _CONFIG_SUPPORT_LIST = OrderedDict([
        ('clip', [
            'clip_vit_b_32',
            'clip_vit_b_16',
            'clip_vit_l_14',
            'clip_vit_l_14@336',
            'mindspore/clip_vit_b_32'
        ]),
        ('mae', [
            'mae_vit_base_p16',
        ]),
        ('vit', [
            'vit_base_p16', 'mindspore/vit_base_p16'
        ]),
        ('swin', [
            'swin_base_p4w7', 'mindspore/swin_base_p4w7'
        ]),
        ('bert', [
            'bert_base_uncased',
            'bert_tiny_uncased',
        ]),
        ('tokcls', OrderedDict([
            ('bert', ['tokcls_bert_base_chinese',
                      'tokcls_bert_base_chinese_cluener',
                      'mindspore/tokcls_bert_base_chinese_cluener'])
        ])),
        ('txtcls', OrderedDict([
            ('bert', ['txtcls_bert_base_uncased',
                      'txtcls_bert_base_uncased_mnli',
                      'mindspore/txtcls_bert_base_uncased_mnli'])
        ])),
        ('qa', OrderedDict([
            ('bert', ['qa_bert_base_uncased',
                      'qa_bert_base_uncased_squad'])
        ])),
        ('t5', [
            't5_small',
        ]),
        ('filip', [
            'filip_vit_l_14',
        ]),
        ('gpt2', [
            'gpt2'
        ]),
        ('llama', [
            'llama_7b',
            'llama_13b',
            'llama_65b',
            'llama_7b_lora'
        ]),
        ('pangualpha', [
            'pangualpha'
        ]),
        ('bloom', [
            'bloom_560m',
            'bloom_7.1b',
            'bloom_65b',
            'bloom_176b',
        ]),
        ('glm', [
            'glm_6b',
            'glm_6b_chat',
            'glm_6b_lora',
            'glm_6b_lora_chat'
        ])
    ])

    _MODEL_SUPPORT_LIST = OrderedDict([
        ('clip', [
            'clip_vit_b_32',
            'clip_vit_b_16',
            'clip_vit_l_14',
            'clip_vit_l_14@336',
            'mindspore/clip_vit_b_32'
        ]),
        ('mae', [
            'mae_vit_base_p16',
        ]),
        ('vit', [
            'vit_base_p16',
            'mindspore/vit_base_p16',
            'mae_vit_base_p16'
        ]),
        ('swin', [
            'swin_base_p4w7', 'mindspore/swin_base_p4w7'
        ]),
        ('bert', [
            'bert_base_uncased',
            'bert_tiny_uncased',
        ]),
        ('tokcls', OrderedDict([
            ('bert', ['tokcls_bert_base_chinese',
                      'tokcls_bert_base_chinese_cluener',
                      'mindspore/tokcls_bert_base_chinese_cluener'])
        ])),
        ('txtcls', OrderedDict([
            ('bert', ['txtcls_bert_base_uncased',
                      'txtcls_bert_base_uncased_mnli',
                      'mindspore/txtcls_bert_base_uncased_mnli'])
        ])),
        ('qa', OrderedDict([
            ('bert', ['qa_bert_base_uncased',
                      'qa_bert_base_uncased_squad'])
        ])),
        ('t5', [
            't5_small',
        ]),
        ('mt5', [
            't5_pegasus_base',
        ]),
        ('filip', [
            'filip_vit_l_14',
        ]),
        ('gpt2', [
            'gpt2'
        ]),
        ('llama', [
            'llama_7b',
            'llama_13b',
            'llama_65b',
            'llama_7b_lora'
        ]),
        ('pangualpha', [
            'pangualpha'
        ]),
        ('bloom', [
            'bloom_560m',
            'bloom_7.1b',
            'bloom_65b',
            'bloom_176b',
        ]),
        ('glm', [
            'glm_6b',
            'glm_6b_chat',
            'glm_6b_lora',
            'glm_6b_lora_chat'
        ])
    ])

    _PROCESSOR_SUPPORT_LIST = OrderedDict([
        ('clip', [
            'clip_vit_b_32',
            'clip_vit_b_16',
            'clip_vit_l_14',
            'clip_vit_l_14@336',
            'mindspore/clip_vit_b_32'
        ]),
        ('mae', [
            'mae_vit_base_p16',
        ]),
        ('vit', [
            'vit_base_p16',
            'mindspore/vit_base_p16',
        ]),
        ('swin', [
            'swin_base_p4w7',
            'mindspore/swin_base_p4w7',
        ]),
        ('bert', [
            'bert_base_uncased',
            'bert_tiny_uncased',
        ]),
        ('tokcls', OrderedDict([
            ('bert', ['tokcls_bert_base_chinese',
                      'tokcls_bert_base_chinese_cluener',
                      'mindspore/tokcls_bert_base_chinese_cluener'])
        ])),
        ('txtcls', OrderedDict([
            ('bert', ['txtcls_bert_base_uncased',
                      'txtcls_bert_base_uncased_mnli',
                      'mindspore/txtcls_bert_base_uncased_mnli'])
        ])),
        ('t5', [
            't5_small',
        ]),
        ('gpt2', [
            'gpt2',
        ]),
        ('llama', [
            'llama',
        ]),
        ('pangualpha', [
            'pangualpha'
        ]),
        ('glm', [
            'glm',
        ]),
        ('bloom', [
            'bloom_560m',
            'bloom_7.1b',
            'bloom_65b',
            'bloom_176b',
        ])
    ])

    _TOKENIZER_SUPPORT_LIST = OrderedDict([
        ('clip', [
            'clip_vit_b_32',
            'clip_vit_b_16',
            'clip_vit_l_14',
            'clip_vit_l_14@336',
            'mindspore/clip_vit_b_32'
        ]),
        ('bert', [
            'bert_base_uncased',
            'bert_tiny_uncased',
        ]),
        ('tokcls', OrderedDict([
            ('bert', ['tokcls_bert_base_chinese',
                      'tokcls_bert_base_chinese_cluener',
                      'mindspore/tokcls_bert_base_chinese_cluener'])
        ])),
        ('txtcls', OrderedDict([
            ('bert', ['txtcls_bert_base_uncased',
                      'txtcls_bert_base_uncased_mnli',
                      'mindspore/txtcls_bert_base_uncased_mnli'])
        ])),
        ('qa', OrderedDict([
            ('bert', ['qa_bert_base_uncased',
                      'qa_bert_base_uncased_squad'])
        ])),
        ('t5', [
            't5_small',
        ]),
        ('gpt2', [
            'gpt2',
        ]),
        ('llama', [
            'llama_7b',
            'llama_13b',
            'llama_65b',
            'llama_7b_lora'
        ]),
        ('pangualpha', [
            'pangualpha_2_6b',
            'pangualpha_13b'
        ]),
        ('glm', [
            'glm_6b',
        ]),
        ('bloom', [
            'bloom_560m',
            'bloom_7.1b',
            'bloom_65b',
            'bloom_176b',
        ])
    ])

    _MODEL_CONFIG_TO_NAME = OrderedDict([
    ])

    _MODEL_CKPT_URL_LIST = OrderedDict([
        ('clip_vit_b_32',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/clip/clip_vit_b_32.ckpt'
          ]),
        ('mindspore/clip_vit_b_32',
         ['https://xihe.mindspore.cn/api/v1/repo/model/MindSpore/clip_vit_b_32/file/clip_vit_b_32.ckpt']),
        ('clip_vit_b_16',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/MindFormers/clip/clip_vit_b_16.ckpt'
          ]),
        ('clip_vit_l_14',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/MindFormers/clip/clip_vit_l_14.ckpt'
          ]),
        ('clip_vit_l_14@336',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/MindFormers/clip/clip_vit_l_14%40336.ckpt'
          ]),
        ('mae_vit_base_p16',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/mae/mae_vit_base_p16.ckpt'
          ]),
        ('vit_base_p16',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/vit/vit_base_p16.ckpt'
          ]),
        ('mindspore/vit_base_p16',
         ['https://xihe.mindspore.cn/api/v1/repo/model/MindSpore/vit_base_p16/file/vit_base_p16.ckpt']),
        ('swin_base_p4w7',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/swin/swin_base_p4w7.ckpt'
          ]),
        ('mindspore/swin_base_p4w7',
         ['https://xihe.mindspore.cn/api/v1/repo/model/MindSpore/swin_base_p4w7/file/swin_base_p4w7.ckpt']),
        ('t5_small',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/t5/mindspore_model.ckpt'
          ]),
        ('bert_tiny_uncased',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/bert/bert_tiny/bert_tiny_uncased.ckpt'
          ]),
        ('bert_base_uncased',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/bert/bert_base/bert_base_uncased.ckpt'
          ]),
        ('tokcls_bert_base_chinese',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/bert/downstream_tasks/'
          'token_classification/tokcls_bert_base_chinese.ckpt'
          ]),
        ('tokcls_bert_base_chinese_cluener',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/bert/downstream_tasks/token_classification/'
          'tokcls_bert_base_chinese_cluener.ckpt'
          ]),
        ('mindspore/tokcls_bert_base_chinese_cluener',
         ['https://xihe.mindspore.cn/api/v1/repo/model/MindSpore/tokcls_bert_base_chineses_cluener/'
          'file/tokcls_bert_base_chinese_cluener.ckpt']),
        ('txtcls_bert_base_uncased',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/bert/downstream_tasks/txtcls/txtcls_bert_base_uncased.ckpt'
          ]),
        ('txtcls_bert_base_uncased_mnli',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/bert/downstream_tasks/txtcls/'
          'txtcls_bert_base_uncased_mnli.ckpt'
          ]),
        ('mindspore/txtcls_bert_base_uncased_mnli',
         ['https://xihe.mindspore.cn/'
          'api/v1/repo/model/MindSpore/txtcls_bert_base_uncased_mnli/file/txtcls_bert_base_uncased_mnli.ckpt'
          ]),
        ('qa_bert_base_uncased',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/bert/downstream_tasks/qa/qa_bert_base_uncased.ckpt'
          ]),
        ('qa_bert_base_uncased_squad',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/bert/downstream_tasks/qa/'
          'qa_bert_base_uncased_squad.ckpt'
          ]),
        ('gpt2',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/gpt2/gpt2.ckpt'
          ]),
        ('pangualpha_2_6b',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/pangualpha/pangualpha_2_6b.ckpt'
          ]),
        ('pangualpha_13b',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/pangualpha/pangualpha_13b.ckpt'
          ]),
        ('glm_6b',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/glm/glm_6b.ckpt'
          ]),
        ('glm_6b_chat',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/glm/glm_6b.ckpt'
          ]),
        ('glm_6b_lora',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/glm/glm_6b_lora.ckpt'
          ]),
        ('glm_6b_lora_chat',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/glm/glm_6b_lora.ckpt'
          ]),
        ('llama_7b',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/llama/open_llama_7b.ckpt'
          ]),
        ('llama_13b',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/llama/open_llama_13b.ckpt'
          ]),
        ('llama_7b_lora',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/llama/open_llama_7b_lora.ckpt'
          ]),
        ('bloom_560m',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/MindFormers/bloom/bloom_560m.ckpt'
          ]),
    ])

    _MODEL_CONFIG_URL_LIST = OrderedDict([
        ('clip_vit_b_32',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/clip/clip_vit_b_32.yaml'
          ]),
        ('clip_vit_b_16',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/MindFormers/clip/clip_vit_b_16.yaml'
          ]),
        ('clip_vit_l_14',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/MindFormers/clip/clip_vit_l_14.yaml'
          ]),
        ('clip_vit_l_14@336',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/MindFormers/clip/clip_vit_l_14%40336.yaml'
          ]),
        ('mae_vit_base_p16',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/mae/mae_vit_base_p16.yaml'
          ]),
        ('vit_base_p16',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/vit/vit_base_p16.yaml'
          ]),
        ('swin_base_p4w7',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/swin/swin_base_p4w7.yaml'
          ]),
        ('t5_small',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/t5/mindspore_model.yaml'
          ]),
        ('bert_tiny_uncased',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/bert/bert_tiny/bert_tiny_uncased.yaml'
          ]),
        ('bert_base_uncased',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/bert/bert_base/bert_base_uncased.yaml'
          ]),
        ('tokcls_bert_base_chinese',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/bert/downstream_tasks/token_classification/tokcls_bert_base_chinese.yaml'
          ]),
        ('tokcls_bert_base_chinese_cluener',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/bert/downstream_tasks/token_classification/'
          'tokcls_bert_base_chinese_cluener.yaml'
          ]),
        ('txtcls_bert_base_uncased',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/bert/downstream_tasks/txtcls/txtcls_bert_base_uncased.yaml'
          ]),
        ('txtcls_bert_base_uncased_mnli',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/bert/downstream_tasks/txtcls/'
          'txtcls_bert_base_uncased_mnli.yaml'
          ]),
        ('qa_bert_base_uncased',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/bert/downstream_tasks/qa/qa_bert_base_uncased.yaml'
          ]),
        ('qa_bert_base_uncased_squad',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/bert/downstream_tasks/qa/'
          'qa_bert_base_uncased_squad.yaml'
          ]),
        ('gpt2',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/gpt2/gpt2.yaml'
          ])
    ])

    _PIPELINE_INPUT_DATA_LIST = OrderedDict([
        ('zero_shot_image_classification',
         "https://ascend-repo-modelzoo.obs.cn-east-2."
         "myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
    ])

    _TOKENIZER_SUPPORT_URL_LIST = OrderedDict([
        ('clip_vit_b_32', [
            "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/"
            "XFormer_for_mindspore/clip/bpe_simple_vocab_16e6.txt.gz"
        ]),
        ('mindspore/clip_vit_b_32', [
            "https://xihe.mindspore.cn/api/v1/repo/model/MindSpore/clip_vit_b_32/file/bpe_simple_vocab_16e6.txt.gz"
        ]),
        ('clip_vit_b_16', [
            "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/"
            "XFormer_for_mindspore/clip/bpe_simple_vocab_16e6.txt.gz"
        ]),
        ('clip_vit_l_14', [
            "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/"
            "XFormer_for_mindspore/clip/bpe_simple_vocab_16e6.txt.gz"
        ]),
        ('clip_vit_l_14@336', [
            "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/"
            "XFormer_for_mindspore/clip/bpe_simple_vocab_16e6.txt.gz"
        ]),
        ('t5_small', [
            "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/"
            "XFormer_for_mindspore/t5/spiece.model"
        ]),
        ('bert_base_uncased', [
            "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/"
            "XFormer_for_mindspore/bert/vocab.txt"
        ]),
        ('bert_tiny_uncased', [
            "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/"
            "XFormer_for_mindspore/bert/vocab.txt"
        ]),
        ('tokcls_bert_base_chinese', [
            "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/"
            "XFormer_for_mindspore/bert/bert_base_chinese/vocab.txt"
        ]),
        ('tokcls_bert_base_chinese_cluener', [
            "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/"
            "XFormer_for_mindspore/bert/bert_base_chinese/vocab.txt"
        ]),
        ('mindspore/tokcls_bert_base_chinese_cluener', [
            "https://xihe.mindspore.cn/api/v1/repo/model/MindSpore/tokcls_bert_base_chineses_cluener/file/vocab.txt"
        ]),
        ('txtcls_bert_base_uncased', [
            "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/"
            "XFormer_for_mindspore/bert/bert_base_english/vocab.txt"
        ]),
        ('txtcls_bert_base_uncased_mnli', [
            "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/"
            "XFormer_for_mindspore/bert/bert_base_english/vocab.txt"
        ]),
        ('mindspore/txtcls_bert_base_uncased_mnli', [
            "https://xihe.mindspore.cn/api/v1/repo/model/MindSpore/txtcls_bert_base_uncased_mnli/file/vocab.txt"
        ]),
        ('qa_bert_base_uncased',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/'
          'XFormer_for_mindspore/bert/bert_base_english/vocab.txt'
          ]),
        ('qa_bert_base_uncased_squad',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/'
          'XFormer_for_mindspore/bert/bert_base_english/vocab.txt'
          ]),
        ('gpt2',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/gpt2/vocab.json',
          'https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/gpt2/merges.txt'
          ]),
        ('pangualpha_2_6b',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/pangualpha/vocab.model'
          ]),
        ('pangualpha_13b',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/pangualpha/vocab.model'
          ]),
        ('glm_6b',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/glm/ice_text.model'
          ]),
        ('llama_7b',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/llama/tokenizer.model'
          ]),
        ('llama_7b_lora',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/XFormer_for_mindspore/llama/tokenizer.model'
          ]),
        ('bloom_560m',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/MindFormers/bloom/tokenizer.json'
          ]),
        ('bloom_7.1b',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/MindFormers/bloom/tokenizer.json'
          ]),
        ('bloom_65b',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/MindFormers/bloom/tokenizer.json'
          ]),
        ('bloom_176b',
         ['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com'
          '/MindFormers/bloom/tokenizer.json'
          ]),
    ])

    _TOKENIZER_NAME_TO_PROCESSOR = OrderedDict([
        ('ChatGLMTokenizer', 'GLMProcessor'),
        ('CLIPTokenizer', 'CLIPProcessor'),
        ('BertTokenizer', 'BertProcessor'),
        ('T5Tokenizer', 'T5Processor'),
        ('LlamaTokenizer', 'LlamaProcessor'),
        ('GPT2Tokenizer', 'GPT2Processor'),
        ('BloomTokenizer', 'BloomProcessor')
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
    def show_model_ckpt_url_list(cls):
        """show_model_ckpt_url_list function"""
        logger.info("MODEL_CKPT_URL_LIST of MindFormer is: ")
        print_dict(cls._MODEL_CKPT_URL_LIST)

    @classmethod
    def get_model_ckpt_url_list(cls):
        """get_model_ckpt_url_list function"""
        return cls._MODEL_CKPT_URL_LIST
