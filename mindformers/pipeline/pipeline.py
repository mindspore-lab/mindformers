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
pipeline
'''
from mindformers.mindformer_book import MindFormerBook
from mindformers.pipeline import build_pipeline


def pipeline(
        task,
        model=None,
        tokenizer=None,
        feature_extractor=None,
        **kwargs
):
    '''pipeline for downstream tasks'''
    if task not in MindFormerBook.get_pipeline_support_task_list().keys():
        raise KeyError(f"{task} is not supported by pipeline. please select"
                       f" a task from {MindFormerBook.get_pipeline_support_task_list().keys()}.")

    if model is None:
        if tokenizer is not None:
            raise KeyError("tokenizer is given without model specified.")
        if feature_extractor is not None:
            raise KeyError("feature_extractor is given without model specified.")

        model = MindFormerBook.get_pipeline_support_task_list()[task][0]

    if isinstance(model, str) and\
            model not in MindFormerBook.get_pipeline_support_task_list()[task]:
        raise KeyError(f"{model} is not supported by {task}.")

    pipeline_name = ''.join([item.capitalize() for item in task.split("_")])+"Pipeline"
    task_pipeline = build_pipeline(class_name=pipeline_name,
                                   model=model,
                                   feature_extractor=feature_extractor,
                                   tokenizer=tokenizer,
                                   **kwargs)

    return task_pipeline
