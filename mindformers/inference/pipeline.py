# Copyright 2023 Huawei Technologies Co., Ltd
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
"""lite infer pipeline."""

import glob
import os.path
from typing import Optional, Union, Tuple

from mindformers.models import PreTrainedTokenizerBase, BaseImageProcessor, BaseAudioProcessor, \
    build_tokenizer, build_processor
from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.register import MindFormerConfig
from .infer_task import InferTask
from .infers.base_infer import BaseInfer
from .infer_config import InferConfig


SUPPORT_MODEL_NAMES = MindFormerBook().get_model_name_support_list()
SUPPORT_PIPELINES = MindFormerBook().get_pipeline_support_task_list()


def is_model_name(model):
    if not isinstance(model, str):
        return False
    if os.path.exists(model):
        return False
    return True


def is_tuple(model):
    return isinstance(model, tuple)


def is_path(model):
    return isinstance(model, str) and os.path.isfile(model)


def is_dir(model):
    return isinstance(model, str) and os.path.isdir(model)


# align with pipeline()
# pylint: disable=W0613
def get_infer_pipeline_from_dir(task, model, tokenizer, image_processor,
                                audio_processor, ge_config_path=None, **kwargs) -> BaseInfer:
    """
    Support user project contain certain files.
    (*.yaml/config.ini/tokenizer.model/graph1.mindir graph2.mindir)

    Args:
        task (str): The supported task could be selected from
            MindFormerBook.show_pipeline_support_task_list().
        model (str): The model dir.
        tokenizer (Optional[PreTrainedTokenizerBase]): The tokenizer of the model.
        image_processor (Optional[BaseImageProcessor]): The image processor of the model.
        audio_processor (Optional[BaseAudioProcessor]): The audio processor of the model.
        ge_config_path: ge config file path

    Returns:
        BaseInfer.
    """
    if not os.path.exists(model):
        raise FileNotFoundError(f"{model} is not found.")
    proj_dir = os.path.realpath(model)

    # check model run config
    run_config = glob.glob(proj_dir + "/*.yaml")
    if len(run_config) > 1:
        raise ValueError(f"{model} contains multi config files: {run_config}.")
    config = MindFormerConfig(run_config[0])
    if config.infer is None:
        raise ValueError(f"There is no infer config in {run_config[0]}.")
    if tokenizer is None:
        tokenizer = build_tokenizer(config.processor.tokenizer)
    if image_processor is None and hasattr(config.processor, 'image_processor'):
        image_processor = build_processor(config.processor.image_processor)

    # check ge config.ini
    if ge_config_path is None:
        ge_config_path = os.path.join(proj_dir, "config.ini")
    if not os.path.exists(ge_config_path):
        raise ValueError(f"There is ge config.ini file in {model}.")
    config.infer.ge_config_path = ge_config_path

    # graph.mindir
    graphs = glob.glob(proj_dir + "/*.mindir")
    if len(graphs) > 2:
        raise ValueError(f"MindIR files in {model} must equal to 1 or 2. But get {len(graphs)} files.")
    if len(graphs) == 2:
        if "prefill" in graphs[0]:
            config.infer.prefill_model_path = graphs[0]
            config.infer.increment_model_path = graphs[1]
        else:
            config.infer.prefill_model_path = graphs[1]
            config.infer.increment_model_path = graphs[0]
    else:
        config.infer.prefill_model_path = graphs[0]

    # set device id and rank id
    if "device_id" in kwargs:
        config.infer.device_id = kwargs["device_id"]
    if "rank_id" in kwargs:
        config.infer.rank_id = kwargs["rank_id"]

    # add model name
    config.infer.model_name = config.trainer.model_name

    infer_config = InferConfig(
        **config.infer
    )

    task_pipeline = InferTask.get_infer_task(task, infer_config, tokenizer=tokenizer, image_processor=image_processor)
    return task_pipeline


# align with pipeline()
# pylint: disable=W0613
def get_infer_pipeline_from_model_name(task, model, tokenizer, image_processor,
                                       audio_processor, ge_config_path, **kwargs) -> BaseInfer:
    """
    Support get infer pipeline from model name.

    Args:
        task (str): The supported task could be selected from
            MindFormerBook.show_pipeline_support_task_list().
        model (str): The model name.
        tokenizer (Optional[PreTrainedTokenizerBase]): The tokenizer of the model.
        image_processor (Optional[BaseImageProcessor]): The image processor of the model.
        audio_processor (Optional[BaseAudioProcessor]): The audio processor of the model.
        ge_config_path: ge config file path

    Returns:
        BaseInfer.
    """
    if model not in SUPPORT_MODEL_NAMES:
        raise KeyError(
            f"model must be in {SUPPORT_MODEL_NAMES} when model's type is string, but get {model}.")

    pipeline_config = MindFormerConfig(SUPPORT_PIPELINES.get(task).get(model))
    if pipeline_config.infer is None:
        raise ValueError(f"There is no infer config in {model} config.")

    if image_processor is None and hasattr(pipeline_config.processor, 'image_processor'):
        image_processor = build_processor(pipeline_config.processor.image_processor)
    if audio_processor is None and hasattr(pipeline_config.processor, 'audio_processor'):
        audio_processor = build_processor(pipeline_config.processor.audio_processor)
    if tokenizer is None:
        tokenizer = build_tokenizer(pipeline_config.processor.tokenizer)

    pipeline_config.infer.model_name = model

    # check ge config.ini
    if ge_config_path is not None:
        ge_config_path = os.path.realpath(ge_config_path)
        if not os.path.exists(ge_config_path):
            raise ValueError(f"GE config {ge_config_path} does not exist.")
        pipeline_config.infer.ge_config_path = ge_config_path

    # set device id and rank id
    if "device_id" in kwargs:
        pipeline_config.infer.device_id = kwargs["device_id"]
    if "rank_id" in kwargs:
        pipeline_config.infer.rank_id = kwargs["rank_id"]

    infer_config = InferConfig(
        **pipeline_config.infer
    )
    task_pipeline = InferTask.get_infer_task(task, infer_config, tokenizer=tokenizer, image_processor=image_processor)
    return task_pipeline


# align with pipeline()
# pylint: disable=W0613
def get_infer_pipeline_from_model_files(task, model, tokenizer, image_processor,
                                        audio_processor, ge_config_path, **kwargs) -> BaseInfer:
    """
    Support get infer pipeline from model files.

    Args:
        task (str): The supported task could be selected from
            MindFormerBook.show_pipeline_support_task_list().
        model (Union[str, Tuple[str, str]]): The model paths of tuple(model_path, model_path).
        tokenizer (Optional[PreTrainedTokenizerBase]): The tokenizer of the model.
        image_processor (Optional[BaseImageProcessor]): The image processor of the model.
        audio_processor (Optional[BaseAudioProcessor]): The audio processor of the model.
        ge_config_path: ge config file path

    Returns:
        BaseInfer.
    """
    if isinstance(model, str):
        full_model_path = model
        cache_model_path = ""
    else:
        full_model_path, cache_model_path = model
        full_model_path = os.path.realpath(full_model_path)
    if not os.path.exists(full_model_path):
        raise FileNotFoundError(f"{full_model_path} is not found.")
    if cache_model_path:
        cache_model_path = os.path.realpath(cache_model_path)
        if not os.path.exists(cache_model_path):
            raise FileNotFoundError(f"{cache_model_path} is not found.")
    config = InferConfig(
        prefill_model_path=full_model_path,
        increment_model_path=cache_model_path,
        ge_config_path=ge_config_path,
        **kwargs
    )
    task_pipeline = InferTask.get_infer_task(task, config, tokenizer=tokenizer, image_processor=image_processor)
    return task_pipeline


STRATEGY_MAP = {
    is_model_name: get_infer_pipeline_from_model_name,
    is_tuple: get_infer_pipeline_from_model_files,
    is_path: get_infer_pipeline_from_model_files,
    is_dir: get_infer_pipeline_from_dir,
}


def get_mslite_pipeline(task: str = None,
                        model: Optional[Union[str, BaseInfer, Tuple[str, str]]] = None,
                        tokenizer: Optional[PreTrainedTokenizerBase] = None,
                        image_processor: Optional[BaseImageProcessor] = None,
                        audio_processor: Optional[BaseAudioProcessor] = None,
                        ge_config_path: str = None,
                        **kwargs) -> BaseInfer:
    """
    Get mslite infer pipeline.

    Args:
        task (str): The supported task could be selected from
            MindFormerBook.show_pipeline_support_task_list().
        model (Union[str, Tuple[str, str]]): The model name.
        tokenizer (Optional[PreTrainedTokenizerBase]): The tokenizer of the model.
        image_processor (Optional[BaseImageProcessor]): The image processor of the model.
        audio_processor (Optional[BaseAudioProcessor]): The audio processor of the model.
        ge_config_path: ge config file path

    Returns:
        BaseInfer.
    """
    if not InferTask.check_task_valid(task):
        raise KeyError(f"{task} is not supported by pipeline. please select"
                       f" a task from {InferTask.support_list()}.")
    # init
    infer_pipeline = None
    for k, v in STRATEGY_MAP.items():
        if k(model):
            infer_pipeline = v(task, model, tokenizer, image_processor,
                               audio_processor, ge_config_path, **kwargs)
    if not infer_pipeline:
        raise ValueError(f"model must be a dir or a mindir file path or a tuple of file path, but get {model}.")

    return infer_pipeline
