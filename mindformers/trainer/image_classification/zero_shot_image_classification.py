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
"""Image Classification Trainer."""
from typing import Optional, List, Union

import numpy as np
from PIL.Image import Image

from mindspore.train.model import Model
from mindspore.train import Callback
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset

from mindformers.common.metric import build_metric
from mindformers.common.callback import build_callback
from mindformers.dataset import build_dataset, check_dataset_config, BaseDataset
from mindformers.models import build_model, build_tokenizer, build_processor, \
    BaseModel, BaseTokenizer, BaseImageProcessor
from mindformers.pipeline import pipeline
from mindformers.trainer.utils import check_model_config
from mindformers.tools.logger import logger
from mindformers.tools.utils import count_params
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from ...dataset.dataloader import build_dataset_loader
from ..config_args import ConfigArguments
from ..base_trainer import BaseTrainer


__all__ = ['ZeroShotImageClassificationTrainer']


@MindFormerRegister.register(MindFormerModuleType.TRAINER)
class ZeroShotImageClassificationTrainer(BaseTrainer):
    """Image Classification Trainer."""

    def __init__(self, model_name: str = None):
        super(ZeroShotImageClassificationTrainer, self).__init__(model_name)
        self.model_name = model_name
        self.kwargs = None

    def evaluate(self,
                 config: Optional[Union[dict, ConfigArguments]] = None,
                 network: Optional[Union[str, BaseModel]] = None,
                 dataset: Optional[Union[str, BaseDataset]] = None,
                 callbacks: Optional[Union[Callback, List[Callback]]] = None,
                 compute_metrics: Optional[Union[dict, set]] = None,
                 **kwargs):
        """evaluate for trainer."""
        self.kwargs = kwargs
        # build dataset
        logger.info(".........Build Dataset..........")
        check_dataset_config(config)
        if dataset is None:
            dataset = build_dataset(config.eval_dataset_task)
        logger.info("Create eval dataset finish, dataset size:%d", dataset.get_dataset_size())

        # build network
        logger.info(".........Build Net..........")
        check_model_config(config)
        if network is None:
            network = build_model(config.model)
        network.set_train(mode=False)
        logger.info("Network Parameters: %s M.", str(count_params(network)))

        logger.info(".........Build Metrics..........")
        if compute_metrics is None:
            compute_metrics = {'Top1 Accuracy': build_metric(config.metric)}

        if callbacks is None:
            callbacks = []
            if config.profile:
                callbacks.append(config.profile_cb)
            callbacks.extend(build_callback(config.eval_callbacks))

        logger.info(".........Starting Init Model..........")
        model = Model(network, metrics=compute_metrics, eval_network=network)

        logger.info(".........Starting Evaling Model..........")
        output = model.eval(dataset,
                            callbacks=callbacks,
                            dataset_sink_mode=config.runner_config.sink_mode)
        logger.info("Top1 Accuracy=%s", (str(output["Top1 Accuracy"]*100)+'%'))
        logger.info(".........Eval Over!.............")

    def predict(self,
                config: Optional[Union[dict, ConfigArguments]] = None,
                input_data: Optional[Union[GeneratorDataset,
                                           Tensor, np.ndarray, Image, str, list]] = None,
                network: Optional[Union[str, BaseModel]] = None,
                tokenizer: Optional[BaseTokenizer] = None,
                image_processor: Optional[BaseImageProcessor] = None, **kwargs):
        """predict for trainer."""
        self.kwargs = kwargs
        logger.info(".........Build Input Data For Predict..........")
        if input_data is None:
            input_data = build_dataset_loader(config.eval_dataset.data_loader)

        logger.info(".........Build Net..........")
        if network is None:
            network = build_model(config.model)

        if network is not None:
            logger.info("Network Parameters: %s M.", str(count_params(network)))

        if tokenizer is None:
            tokenizer = build_tokenizer(config.processor.tokenizer)

        if image_processor is None:
            image_processor = build_processor(config.processor.image_processor)

        candidate_labels = kwargs.pop("candidate_labels", None)
        if candidate_labels is None:
            if hasattr(input_data, "label_names"):
                candidate_labels = input_data.label_names
            else:
                candidate_labels = ["sunflower", "tree", "dog", "cat", "toy"]

        hypothesis_template = kwargs.pop("hypothesis_template", None)
        if hypothesis_template is None:
            if config.eval_dataset.data_loader.hypothesis_template is not None:
                hypothesis_template = config.eval_dataset.data_loader.hypothesis_template
            else:
                hypothesis_template = "{}"

        save_file = kwargs.pop("save_file", None)
        if save_file is None:
            if config.save_file is not None:
                save_file = config.save_file
            else:
                save_file = "results.txt"

        batch_size = kwargs.pop("batch_size", None)
        if batch_size is None:
            if config.eval_dataset.batch_size is not None:
                batch_size = config.eval_dataset.batch_size
            else:
                batch_size = 1

        top_k = kwargs.pop("top_k", None)
        if top_k is None and config.top_k is not None:
            top_k = config.top_k

        pipeline_task = pipeline(
            task='zero_shot_image_classification',
            model=network,
            tokenizer=tokenizer,
            image_processor=image_processor,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
            **kwargs
        )
        output_result = pipeline_task(input_data,
                                      batch_size=batch_size,
                                      top_k=top_k)

        with open(save_file, 'w') as file:
            file.write(str(output_result))
        file.close()

        logger.info("output result is saved at: %s", save_file)
        logger.info(".........Predict Over!.............")
        return output_result
