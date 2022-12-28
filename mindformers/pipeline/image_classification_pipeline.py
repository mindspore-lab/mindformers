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
"""Image Classification Pipeline API."""
from mindspore.ops import operations as P

from mindformers.auto_class import AutoProcessor, AutoModel
from mindformers.mindformer_book import MindFormerBook
from mindformers.models import BaseModel, BaseImageProcessor
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.dataset.labels import labels
from .base_pipeline import BasePipeline


@MindFormerRegister.register(MindFormerModuleType.PIPELINE, alias="image_classification")
class ImageClassificationForPipeline(BasePipeline):
    """
    Pipeline for image classification

    Args:
        model: a pretrained model (str or BaseModel) in _supproted_list.
        image_processor : a image_processor (None or BaseFeatureExtractor) for image processing
    """
    _support_list = MindFormerBook.get_pipeline_support_task_list()['image_classification'].keys()

    def __init__(self, model, image_processor=None, **kwargs):
        if isinstance(model, str):
            if model in self._support_list:
                if image_processor is None:
                    image_processor = AutoProcessor.from_pretrained(model).feature_extractor
                if not isinstance(image_processor, BaseImageProcessor):
                    raise TypeError(f"feature_extractor should be inherited from"
                                    f" BaseFeatureExtractor, but got {type(image_processor)}.")
                model = AutoModel.from_pretrained(model)
            else:
                raise ValueError(f"{model} is not supported by ZeroShotImageClassificationPipeline,"
                                 f"please selected from {self._support_list}.")

        if not isinstance(model, BaseModel):
            raise TypeError(f"model should be inherited from BaseModel, but got {type(BaseModel)}.")

        if image_processor is None:
            raise ValueError("ZeroShotImageClassificationPipeline"
                             " requires for a feature_extractor.")

        super().__init__(model.set_train(mode=False), image_processor=image_processor, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        """sanitize parameters for preprocess, forward, and postprocess."""
        preprocess_params = {}
        postprocess_params = {}

        pre_list = []
        for item in pre_list:
            if item in pipeline_parameters:
                preprocess_params[item] = pipeline_parameters.get(item)

        if "top_k" in pipeline_parameters:
            postprocess_params["top_k"] = pipeline_parameters.get("top_k")
        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs, **preprocess_params):
        """
        Preprocess of ZeroShotImageClassificationPipeline

        Args:
            inputs (url, PIL.Image, tensor, numpy): the image to be classified.

        Return:
            processed image.
        """
        if isinstance(inputs, dict):
            inputs = inputs['image']

        image_processed = self.image_processor(inputs)
        return {"image_processed": image_processed}

    def forward(self, model_inputs, **forward_params):
        """
        Forward process

        Args:
            model_inputs (dict): outputs of preprocess.

        Return:
            probs dict.
        """
        forward_params.pop("None", None)

        image_processed = model_inputs["image_processed"]

        logits_per_image, = self.model(image_processed)
        probs = P.Softmax()(logits_per_image).asnumpy()
        return {"probs": probs}

    def postprocess(self, model_outputs, **postprocess_params):
        """
        Postprocess

        Args:
            model_outputs (dict): outputs of forward process.
            top_k (int): return top_k probs of result

        Return:
            classification results.
        """
        top_k = postprocess_params.pop("top_k", None)
        scores = model_outputs['probs']

        outputs = []
        imagenet_labels = labels.get("imagenet")
        for score in scores:
            sorted_res = sorted(zip(score, imagenet_labels), key=lambda x: -x[0])
            if top_k is not None:
                sorted_res = sorted_res[:min(top_k, len(imagenet_labels))]
            outputs.append([{"score": score_item, "label": label}
                            for score_item, label in sorted_res])
        return outputs
