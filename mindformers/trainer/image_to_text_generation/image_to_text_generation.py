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
"""ImageToTextGeneration Trainer."""
from typing import Optional, Union

import numpy as np
from PIL.Image import Image

from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.dataset import GeneratorDataset

from mindformers.models import PreTrainedModel, PreTrainedTokenizerBase, BaseImageProcessor
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, \
    MindFormerModuleType, MindFormerConfig
from ...dataset.dataloader import build_dataset_loader
from ..config_args import ConfigArguments
from ..training_args import TrainingArguments
from ..base_trainer import BaseTrainer

__all__ = ['ImageToTextGenerationTrainer']


@MindFormerRegister.register(MindFormerModuleType.TRAINER)
class ImageToTextGenerationTrainer(BaseTrainer):
    """
    ImageToTextGenerationTrainer Task For Trainer.

    Args:
        model_name (str): The model name of Task-Trainer. Default: None

    Raises:
        NotImplementedError: If train method or evaluate method or predict method not implemented.

    Examples:
        >>> from mindformers.trainer import ImageToTextGenerationTrainer
        >>> trainer = ImageToTextGenerationTrainer(model_name="blip2_stage2_vit_g_llama_7b")
        >>> type(trainer)
        <class 'mindformers.trainer.image_to_text_generation.image_to_text_generation.ImageToTextGenerationTrainer'>
    """

    def __init__(self, model_name: str = None):
        super(ImageToTextGenerationTrainer, self).__init__("image_to_text_generation", model_name)

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "The ImageToTextGenerationTrainer task does not support train.")

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError(
            "The ImageToTextGenerationTrainer task does not support evaluate.")

    def predict(self,
                config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                input_data: Optional[Union[GeneratorDataset,
                                           Tensor, np.ndarray, Image, str, list]] = None,
                network: Optional[Union[Cell, PreTrainedModel]] = None,
                tokenizer: Optional[PreTrainedTokenizerBase] = None,
                image_processor: Optional[BaseImageProcessor] = None, **kwargs):
        """
        Predict task for ZeroShotImageToTextGenerationTrainer Trainer.
        This function is used to predict the network.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            input_data (Optional[Union[GeneratorDataset, Tensor, np.ndarray, Image, str, list]]):
                The dataset. It supports real dataset path or BaseDateset class or MindSpore Dataset class.
                Default: None.
            network (Optional[Union[Cell, PreTrainedModel]]):
                The network for trainer. It supports model name or PreTrainedModel or MindSpore Cell class.
                Default: None.
            tokenizer (Optional[PreTrainedTokenizerBase]):
                Used for text process. Default: None.
            image_processor (Optional[BaseImageProcessor]):
                Used for image process. Default: None.
        """
        config = self.set_config(config)

        logger.info(".........Build Input Data For Predict..........")
        if input_data is None and config.input_data is not None:
            input_data = config.input_data

        if input_data is None:
            input_data = build_dataset_loader(config.eval_dataset.data_loader)

        max_length = kwargs.pop("max_length", None)
        if max_length is None:
            if config.processor.tokenizer.max_length is not None:
                max_length = config.processor.tokenizer.max_length
            else:
                max_length = 32

        padding = kwargs.pop("padding", None)
        if padding is None:
            if config.processor.tokenizer.padding is not None:
                padding = config.processor.tokenizer.padding
            else:
                padding = "max_length"

        hypothesis_template = kwargs.pop("hypothesis_template", None)
        if hypothesis_template is None:
            if config.eval_dataset.data_loader.hypothesis_template is not None:
                hypothesis_template = config.eval_dataset.data_loader.hypothesis_template
            else:
                hypothesis_template = "{}"
        return self.predict_process(config=config,
                                    input_data=input_data,
                                    task='image_to_text_generation',
                                    network=network,
                                    tokenizer=tokenizer,
                                    image_processor=image_processor,
                                    max_length=max_length,
                                    padding=padding,
                                    hypothesis_template=hypothesis_template,
                                    **kwargs)

    def export(self, **kwargs):
        raise NotImplementedError(
            "The image to text generation task does not support export, please customize pipeline inference.")
