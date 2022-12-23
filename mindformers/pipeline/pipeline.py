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
pipeline
"""
from typing import Optional, Union

from mindformers.models import build_model, build_tokenizer, build_feature_extractor, \
    BaseModel, BaseTokenizer, BaseFeatureExtractor
from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.register import MindFormerConfig
from .build_pipeline import build_pipeline

SUPPORT_PIPELINES = MindFormerBook().get_pipeline_support_task_list()
SUPPORT_MODEL_NAMES = MindFormerBook().get_model_name_support_list()


def pipeline(
        task: str = None,
        model: Optional[Union[str, BaseModel]] = None,
        tokenizer: Optional[BaseTokenizer] = None,
        feature_extractor: Optional[BaseFeatureExtractor] = None,
        **kwargs):
    """
    Pipeline for downstream tasks

    Args:
        task (str): the supported task could be selected from
         MindFormerBook.show_pipeline_support_task_list().
        model (str, BaseModel): the model used for task.
        tokenizer (BaseTokenizer): the tokenizer of the model.
        feature_extractor (BaseFeatureExtractor): the feature extractor of the model.

    Return:
        a task pipeline.
    """
    if task not in SUPPORT_PIPELINES.keys():
        raise KeyError(f"{task} is not supported by pipeline. please select"
                       f" a task from {SUPPORT_PIPELINES.keys()}.")

    if isinstance(model, str):
        if model not in SUPPORT_MODEL_NAMES:
            raise KeyError(
                f"model must be in {SUPPORT_MODEL_NAMES} when model's type is string, but get {model}.")
        model_name = model
        model = None
    else:
        model_name = "common"

    pipeline_config = MindFormerConfig(SUPPORT_PIPELINES.get(task).get(model_name))
    pipeline_type = MindFormerBook().PIPELINE_TASK_NAME_TO_PIPELINE.get(task)

    if model is None:
        model = build_model(pipeline_config.model)

    if feature_extractor is None:
        feature_extractor = build_feature_extractor(pipeline_config.processor.feature_extractor)

    if tokenizer is None:
        tokenizer = build_tokenizer(pipeline_config.processor.tokenizer)

    task_pipeline = build_pipeline(class_name=pipeline_type,
                                   model=model,
                                   feature_extractor=feature_extractor,
                                   tokenizer=tokenizer,
                                   **kwargs)

    return task_pipeline
