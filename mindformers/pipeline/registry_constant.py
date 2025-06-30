# Copyright 2018 The HuggingFace Inc. team.
# Copyright 2023-2024 Huawei Technologies Co., Ltd
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

"""Constant Declaration of Pipeline Registry"""
from mindformers.models.auto.modeling_auto import AutoModelForCausalLM

from .pipeline_registry import PipelineRegistry
from .text_generation_pipeline import TextGenerationPipeline

TASK_ALIASES = {
    "image_classification": "image-classification",
    "masked_image_modeling": "masked-image-modeling",
    "text_generation": "text-generation",
    "zero_shot_image_classification": "zero-shot-image-classification",
}

# default model repo need to be filled.
SUPPORTED_TASKS = {
    "text-generation": {
        "impl": TextGenerationPipeline,
        "ms": (AutoModelForCausalLM,),
        "default": {"model": {"ms": ()}},
        "type": "text",
    },
}

NO_FEATURE_EXTRACTOR_TASKS = set()
NO_IMAGE_PROCESSOR_TASKS = set()
NO_TOKENIZER_TASKS = set()
# Those model configs are special, they are generic over their task, meaning
# any tokenizer/feature_extractor might be used for a given model, so we cannot
# use the statically defined TOKENIZER_MAPPING and FEATURE_EXTRACTOR_MAPPING to
# see if the model defines such objects or not.
# reserved multi-modal configs
MULTI_MODEL_CONFIGS = {}
for task, values in SUPPORTED_TASKS.items():
    if values.get("type") == "text":
        NO_FEATURE_EXTRACTOR_TASKS.add(task)
        NO_IMAGE_PROCESSOR_TASKS.add(task)
    elif values.get("type") in {"image", "video"}:
        NO_TOKENIZER_TASKS.add(task)
    elif values.get("type") in {"audio"}:
        NO_TOKENIZER_TASKS.add(task)
        NO_IMAGE_PROCESSOR_TASKS.add(task)
    elif values.get("type") != "multimodal":
        raise ValueError(f"SUPPORTED_TASK {task} contains invalid type {values.get('type')}")

PIPELINE_REGISTRY = PipelineRegistry(supported_tasks=SUPPORTED_TASKS, task_aliases=TASK_ALIASES)
