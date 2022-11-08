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
pipeline for quick inference task.
"""
from collections import OrderedDict
from mindtransformer.tasks.question_answering import QATask, QATaskConfig
from mindtransformer.tasks.text_classification import TextClassificationTask, TextClassificationConfig
from mindtransformer.tasks.language_modeling import LMTask, LMTaskConfig

TASK_CONFIG_MAPPING = OrderedDict(
    [
        ('question_answering', QATaskConfig),
        ('text_classification', TextClassificationConfig),
        ('language_modeling', LMTaskConfig),
    ]
)

TASK_MAPPING = OrderedDict(
    [
        ('question_answering', QATask),
        ('text_classification', TextClassificationTask),
        ('language_modeling', LMTask),
    ]
)


def pipeline(task_name):
    """
    pipeline task
    """
    config_class = None
    if task_name in TASK_CONFIG_MAPPING.keys():
        config_class = TASK_CONFIG_MAPPING[task_name]
    if config_class is None:
        print("Invalid task name ", task_name)

    task_class = None
    if task_name in TASK_MAPPING.keys():
        task_class = TASK_MAPPING[task_name]
    if task_class is None:
        print("Invalid task name ", task_name)

    config = config_class()
    task = task_class(config)
    return task
