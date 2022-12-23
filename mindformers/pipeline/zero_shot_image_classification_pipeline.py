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

"""ZeroShotImageClassificationPipeline"""
from mindspore.ops import operations as P

from mindformers.auto_class import AutoProcessor, AutoModel
from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.image_tools import load_image
from mindformers.models import BaseModel, BaseFeatureExtractor, Tokenizer
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from .base_pipeline import BasePipeline


@MindFormerRegister.register(MindFormerModuleType.PIPELINE)
class ZeroShotImageClassificationPipeline(BasePipeline):
    """
    Pipeline for zero shot image classification

    Args:
        model: a pretrained model (str or BaseModel) in _supproted_list.
        tokenizer : a tokenizer (None or PretrainedTokenizer) for text processing
        feature_extractor : a feature_extractor (None or BaseFeatureExtractor) for image processing
    """
    _support_list = MindFormerBook.get_model_support_list()['clip']

    def __init__(self, model, tokenizer=None, feature_extractor=None, **kwargs):
        if isinstance(model, str):
            if model in self._support_list:
                if feature_extractor is None:
                    feature_extractor = AutoProcessor.from_pretrained(model).feature_extractor
                if not isinstance(feature_extractor, BaseFeatureExtractor):
                    raise TypeError(f"feature_extractor should be inherited from"
                                    f" BaseFeatureExtractor, but got {type(feature_extractor)}.")
                if tokenizer is None:
                    tokenizer = AutoProcessor.from_pretrained(model).tokenizer
                if not isinstance(tokenizer, Tokenizer):
                    raise TypeError(f"tokenizer should be inherited from"
                                    f" PretrainedTokenizer, but got {type(tokenizer)}.")
                model = AutoModel.from_pretrained(model)
            else:
                raise ValueError(f"{model} is not supported by ZeroShotImageClassificationPipeline,"
                                 f"please selected from {self._supprot_list}.")

        if not isinstance(model, BaseModel):
            raise TypeError(f"model should be inherited from BaseModel, but got {type(BaseModel)}.")

        if tokenizer is None:
            raise ValueError("ZeroShotImageClassificationPipeline"
                             " requires for a tokenizer.")
        if feature_extractor is None:
            raise ValueError("ZeroShotImageClassificationPipeline"
                             " requires for a feature_extractor.")

        super().__init__(model, tokenizer, feature_extractor, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        """sanitize parameters for preprocess, forward, and postprocess."""
        preprocess_params = {}
        postprocess_params = {}

        pre_list = ["candidate_labels", "hypothesis_template",
                    "max_length", "padding", "return_tensors"]
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
            candidate_labels (str, list): the candidate labels for classification.
            max_length (int): max length of tokenizer's output
            padding (False / "max_length"): padding for max_length
            return_tensors ("ms"): the type of returned tensors

        Return:
            processed image.
        """
        candidate_labels = preprocess_params.pop("candidate_labels", None)
        hypothesis_template = preprocess_params.pop("hypothesis_template",
                                                    "This is a photo of {}.")
        max_length = preprocess_params.pop("max_length", 77)
        padding = preprocess_params.pop("padding", "max_length")
        return_tensors = preprocess_params.pop("return_tensors", "ms")

        if candidate_labels is None:
            raise ValueError("candidate_labels are supposed for"
                             " ZeroShotImageClassificationPipeline, but got None.")
        if hypothesis_template is None:
            raise ValueError("hypothesis_template is supposed for"
                             " ZeroShotImageClassificationPipeline, but got None.")
        if isinstance(inputs, dict):
            inputs = inputs['image']

        image = load_image(inputs)
        image_processed = self.feature_extractor(image)
        sentences = [hypothesis_template.format(candidate_label)
                     for candidate_label in candidate_labels]
        input_ids = self.tokenizer(sentences, max_length=max_length, padding=padding,
                                   return_tensors=return_tensors)["input_ids"]
        return {"image_processed": image_processed,
                "input_ids": input_ids, "candidate_labels": candidate_labels}

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
        input_ids = model_inputs["input_ids"]

        logits_per_image, _ = self.model(image_processed, input_ids)
        probs = P.Softmax()(logits_per_image).asnumpy()
        return {"probs": probs, "candidate_labels": model_inputs["candidate_labels"]}

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

        labels = model_outputs['candidate_labels']
        scores = model_outputs['probs']

        outputs = []
        for score in scores:
            sorted_res = sorted(zip(score, labels), key=lambda x: -x[0])
            if top_k is not None:
                sorted_res = sorted_res[:min(top_k, len(labels))]
            outputs.append([{"score": score_item, "label": label}
                            for score_item, label in sorted_res])
        return outputs
