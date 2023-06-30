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
"""Masked Image Modeling Trainer."""
from typing import Optional, List, Union

import numpy as np
from PIL.Image import Image
from mindspore import Tensor
from mindspore.train import Callback
from mindspore.nn import TrainOneStepCell, Optimizer, Cell
from mindspore.dataset import GeneratorDataset

from mindformers.dataset import BaseDataset
from mindformers.models import BaseModel
from mindformers.tools.register import MindFormerRegister, \
    MindFormerModuleType, MindFormerConfig
from mindformers.trainer.config_args import ConfigArguments
from mindformers.trainer.training_args import TrainingArguments
from mindformers.trainer.base_trainer import BaseTrainer
from mindformers.models.base_processor import BaseImageProcessor
from mindformers.tools.logger import logger
from mindformers.tools.image_tools import load_image


@MindFormerRegister.register(MindFormerModuleType.TRAINER)
class MaskedImageModelingTrainer(BaseTrainer):
    r"""MaskedImageModeling Task For Trainer.
    Args:
        model_name (str): The model name of Task-Trainer. Default: None
    Examples:
        >>> import numpy as np
        >>> from mindspore.dataset import GeneratorDataset
        >>> from mindspore.nn import AdamWeightDecay, WarmUpLR, \
        ...      DynamicLossScaleUpdateCell, TrainOneStepWithLossScaleCell
        >>> from mindformers.trainer import GeneralTaskTrainer
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers.models import ViTMAEForPreTraining, ViTMAEConfig
        >>> class MyDataLoader:
        ...    def __init__(self):
        ...        self.image = [np.zeros((3, 224, 224), np.float32) for _ in range(64)]
        ...        self.mask = [np.zeros((196,), np.int32) for _ in range(64)]
        ...        self.ids_restore = [np.zeros((196,), np.int32) for _ in range(64)]
        ...        self.unmask_index = [np.zeros((49,), np.int32) for _ in range(64)]
        ...    def __getitem__(self, index):
        ...        return self.image[index], self.mask[index], self.ids_restore[index], self.unmask_index[index]
        ...    def __len__(self):
        ...        return len(self.image)
        >>> train_dataset = GeneratorDataset(source=MyDataLoader(),
        ...                                  column_names=["image", "mask", "ids_restore", "unmask_index"]).batch(2)
        >>> #1) use config to train
        >>> mae_trainer = MaskedImageModelingTrainer(model_name='mae_vit_base_p16')
        >>> mae_trainer.train(dataset=train_dataset)
        >>> #2) use instance function to train
        >>> mae_config = ViTMAEConfig(batch_size=2)
        >>> network_with_loss = ViTMAEForPreTraining(mae_config)
        >>> lr_schedule = WarmUpLR(learning_rate=0.001, warmup_steps=100)
        >>> optimizer = AdamWeightDecay(beta1=0.009, beta2=0.999,
        ...                             learning_rate=lr_schedule,
        ...                             params=network_with_loss.trainable_params())
        >>> loss_scale = DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)
        >>> wrapper = TrainOneStepWithLossScaleCell(network_with_loss, optimizer, scale_sense=loss_scale)
        >>> mae_trainer.train(wrapper=wrapper, optimizer=optimizer, dataset=train_dataset)
    Raises:
        NotImplementedError: If train method or evaluate method or predict method not implemented.
    """
    def __init__(self, model_name: str = None):
        super(MaskedImageModelingTrainer, self).__init__("masked_image_modeling", model_name)

    def train(self,
              config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
              network: Optional[Union[Cell, BaseModel]] = None,
              dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
              wrapper: Optional[TrainOneStepCell] = None,
              optimizer: Optional[Optimizer] = None,
              callbacks: Optional[Union[Callback, List[Callback]]] = None,
              **kwargs):
        r"""Train task for MaskedImageModeling Trainer.
        This function is used to train or fine-tune the network.

        The trainer interface is used to quickly start training for general task.
        It also allows users to customize the network, optimizer, dataset, wrapper, callback.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            network (Optional[Union[Cell, BaseModel]]): The network for trainer.
                It supports model name or BaseModel or MindSpore Cell class.
                Default: None.
            dataset (Optional[Union[BaseDataset, GeneratorDataset]]): The training dataset.
                It support real dataset path or BaseDateset class or MindSpore Dataset class.
                Default: None.
            optimizer (Optional[Optimizer]): The training network's optimizer. It support Optimizer class of MindSpore.
                Default: None.
            wrapper (Optional[TrainOneStepCell]): Wraps the `network` with the `optimizer`.
                It support TrainOneStepCell class of MindSpore.
                Default: None.
            callbacks (Optional[Union[Callback, List[Callback]]]): The training callback function.
                It support CallBack or CallBack List of MindSpore.
                Default: None.

        Raises:
            NotImplementedError: If wrapper not implemented.
        """
        self.training_process(
            config=config,
            network=network,
            callbacks=callbacks,
            dataset=dataset,
            wrapper=wrapper,
            optimizer=optimizer,
            **kwargs)

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError(
            "The MaskedImageModeling task does not support evaluate.")

    def predict(self,
                config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                input_data: Optional[Union[Tensor, np.ndarray, Image, str, list]] = None,
                network: Optional[Union[Cell, BaseModel]] = None,
                image_processor: Optional[BaseImageProcessor] = None, **kwargs):
        r"""Predict task for MaskedImageModeling Trainer.
                This function is used to predict the network.

                The trainer interface is used to quickly start training for general task.
                It also allows users to customize the network, tokenizer, image_processor, audio_processor.

                Args:
                    config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                        The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                        It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                        Default: None.
                    input_data (Optional[Union[Tensor, np.ndarray, Image, str, list]]): The predict data. Default: None.
                    network (Optional[Union[Cell, BaseModel]]): The network for trainer.
                        It supports model name or BaseModel or MindSpore Cell class.
                        Default: None.
                    image_processor (Optional[BaseImageProcessor]): The processor for image preprocessing.
                        It support BaseImageProcessor class.
                        Default: None.
                """
        logger.info(".........Build Input Data For Predict..........")
        if input_data is None:
            input_data = config.input_data
        if not isinstance(input_data, (Tensor, np.ndarray, Image, str, list)):
            raise ValueError("Input data's type must be one of "
                             "[str, ms.Tensor, np.ndarray, PIL.Image.Image, list]")
        batch_input_data = []
        if isinstance(input_data, str):
            batch_input_data.append(load_image(input_data))
        elif isinstance(input_data, list):
            for data_path in input_data:
                batch_input_data.append(load_image(data_path))
        else:
            batch_input_data = input_data

        return self.predict_process(config=config,
                                    input_data=batch_input_data,
                                    task='masked_image_modeling',
                                    network=network,
                                    image_processor=image_processor,
                                    **kwargs)
