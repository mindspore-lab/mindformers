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
from enum import Enum
from typing import Optional, Union, Tuple
from mindspore import Model

from mindformers.models import build_model, build_tokenizer, build_processor, \
    BaseModel, BaseTokenizer, BaseImageProcessor, BaseAudioProcessor
from mindformers.mindformer_book import MindFormerBook
from mindformers.pet import get_pet_model, is_supported_pet_type
from mindformers.tools.register import MindFormerConfig
from .build_pipeline import build_pipeline


SUPPORT_PIPELINES = MindFormerBook().get_pipeline_support_task_list()


class Backend(Enum):
    MS = "ms"
    MS_LITE = "mslite"


def pipeline(
        task: str = None,
        model: Optional[Union[str, BaseModel, Model, Tuple[str, str]]] = None,
        tokenizer: Optional[BaseTokenizer] = None,
        image_processor: Optional[BaseImageProcessor] = None,
        audio_processor: Optional[BaseAudioProcessor] = None,
        backend: str = "ms",
        **kwargs):
    r"""Pipeline for downstream tasks

    Args:
        task (str):
            The supported task could be selected from MindFormerBook.show_pipeline_support_task_list().
        model (Optional[Union[str, BaseModel]]):
            The model used for task.
        tokenizer (Optional[BaseTokenizer]):
            The tokenizer of the model.
        image_processor (Optional[BaseImageProcessor]):
            The image processor of the model.
        audio_processor (Optional[BaseAudioProcessor]):
            The audio processor of the model.
        backend(str):
            The inference backend. Default "ms", now support ["ms", "mslite"].
        **kwargs:
            Refers to the kwargs description of the corresponding task pipeline.

    Returns:
        A task pipeline.

    Raises:
        KeyError: If the task or model is not supported.

    Examples:
        >>> from mindformers import pipeline
        >>> from mindformers.tools.image_tools import load_image
        >>> classifier = pipeline("zero_shot_image_classification",
            candidate_labels=["sunflower", "tree", "dog", "cat", "toy"])
        >>> img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2."
            "myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
        >>> classifier(img)
        [[{'score': 0.99995565, 'label': 'sunflower'},
        {'score': 2.5318595e-05, 'label': 'toy'},
        {'score': 9.903885e-06, 'label': 'dog'},
        {'score': 6.75336e-06, 'label': 'tree'},
        {'score': 2.396818e-06, 'label': 'cat'}]]
    """
    if backend == Backend.MS_LITE.value:
        from mindformers.inference import get_mslite_pipeline
        task_pipeline = get_mslite_pipeline(task, model, tokenizer, image_processor, audio_processor, **kwargs)
    elif backend == Backend.MS.value:
        task_pipeline = get_ms_pipeline(task, model, tokenizer, image_processor, audio_processor, **kwargs)
    else:
        raise ValueError(f"The inference backend \"{backend}\" is not supported,"
                         f"please select a backend from [\"ms\", \"mslite\"]")
    return task_pipeline


def get_ms_pipeline(task, model, tokenizer, image_processor, audio_processor, **kwargs):
    """get mindspore infer pipeline."""
    if task not in SUPPORT_PIPELINES.keys():
        raise KeyError(f"{task} is not supported by pipeline. please select"
                       f" a task from {SUPPORT_PIPELINES.keys()}.")
    if isinstance(model, str):
        support_model_name = SUPPORT_PIPELINES[task].keys()
        if model not in support_model_name:
            raise KeyError(
                f"model must be in {support_model_name} when model's type is string, but get {model}.")
        model_name = model
        model = None
    else:
        model_name = "common"
    pipeline_config = MindFormerConfig(SUPPORT_PIPELINES.get(task).get(model_name))

    if model is None:
        batch_size = kwargs.get("batch_size", None)
        build_names = ["batch_size", "use_past", "seq_length"]
        build_args = {}
        for build_name in build_names:
            if build_name in kwargs:
                build_args[build_name] = kwargs.pop(build_name)
        ckpt_cfg = pipeline_config.model.model_config.checkpoint_name_or_path
        pet_config = pipeline_config.model.model_config.pet_config
        if pet_config and is_supported_pet_type(pet_config.pet_type):
            pipeline_config.model.model_config.checkpoint_name_or_path = None
        model = build_model(pipeline_config.model, default_args=build_args)
        if pet_config:
            model.config.checkpoint_name_or_path = ckpt_cfg
            model = get_pet_model(model, pet_config)
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
    if image_processor is None and hasattr(pipeline_config.processor, 'image_processor'):
        image_processor = build_processor(pipeline_config.processor.image_processor)
    if audio_processor is None and hasattr(pipeline_config.processor, 'audio_processor'):
        audio_processor = build_processor(pipeline_config.processor.audio_processor)
    if tokenizer is None:
        tokenizer = build_tokenizer(pipeline_config.processor.tokenizer, tokenizer_name=model_name)
    task_pipeline = build_pipeline(class_name=task,
                                   model=model,
                                   image_processor=image_processor,
                                   audio_processor=audio_processor,
                                   tokenizer=tokenizer,
                                   **kwargs)
    return task_pipeline
