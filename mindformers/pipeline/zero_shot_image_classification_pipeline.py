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
from typing import Union, Optional
from mindspore.ops import operations as P

from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.image_tools import load_image
from mindformers.models import BaseModel, BaseImageProcessor, Tokenizer
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.auto_class import AutoProcessor, AutoModel
from .base_pipeline import BasePipeline
from ..models import BaseTokenizer


@MindFormerRegister.register(MindFormerModuleType.PIPELINE, alias="zero_shot_image_classification")
class ZeroShotImageClassificationPipeline(BasePipeline):
    r"""Pipeline For Zero Shot Image Classification

    Args:
        model (Union[str, BaseModel]): The model used to perform task,
            the input could be a supported model name, or a model instance
            inherited from BaseModel.
        tokenizer (Optional[BaseTokenizer]): A tokenizer for text processing.
        image_processor (Optional[BaseImageProcessor]): The image_processor of model,
            it could be None if the model do not need image_processor.

    Raises:
        TypeError: If input model, tokenizer, and image_processor's types are not corrected.
        ValueError: if the input model is not in support list.

    Examples:
        >>> from mindformers.tools.image_tools import load_image
        >>> from mindformers.pipeline import ZeroShotImageClassificationPipeline
        >>> classifier = ZeroShotImageClassificationPipeline(
        ...     model='clip_vit_b_32',
        ...     candidate_labels=["sunflower", "tree", "dog", "cat", "toy"],
        ...     hypothesis_template="This is a photo of {}."
        ...     )
        >>> img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2."
        ...                  "myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
        >>> classifier(img)
            [[{'score': 0.99995565, 'label': 'sunflower'},
            {'score': 2.5318595e-05, 'label': 'toy'},
            {'score': 9.903885e-06, 'label': 'dog'},
            {'score': 6.75336e-06, 'label': 'tree'},
            {'score': 2.396818e-06, 'label': 'cat'}]]
    """
    _support_list = MindFormerBook.get_pipeline_support_task_list()['zero_shot_image_classification'].keys()

    def __init__(self, model: Union[str, BaseModel],
                 tokenizer: Optional[BaseTokenizer] = None,
                 image_processor: Optional[BaseImageProcessor] = None,
                 **kwargs):
        if isinstance(model, str):
            if model in self._support_list:
                if image_processor is None:
                    image_processor = AutoProcessor.from_pretrained(model).image_processor
                if not isinstance(image_processor, BaseImageProcessor):
                    raise TypeError(f"image_processor should be inherited from"
                                    f" BaseImageProcessor, but got {type(image_processor)}.")
                if tokenizer is None:
                    tokenizer = AutoProcessor.from_pretrained(model).tokenizer
                if not isinstance(tokenizer, Tokenizer):
                    raise TypeError(f"tokenizer should be inherited from"
                                    f" PretrainedTokenizer, but got {type(tokenizer)}.")
                model = AutoModel.from_pretrained(model)
            else:
                raise ValueError(f"{model} is not supported by ZeroShotImageClassificationPipeline,"
                                 f"please selected from {self._support_list}.")

        if not isinstance(model, BaseModel):
            raise TypeError(f"model should be inherited from BaseModel, but got {type(model)}.")

        if tokenizer is None:
            raise ValueError("ZeroShotImageClassificationPipeline"
                             " requires for a tokenizer.")
        if image_processor is None:
            raise ValueError("ZeroShotImageClassificationPipeline"
                             " requires for a image_processor.")

        super().__init__(model, tokenizer, image_processor, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        r"""Sanitize Parameters

        Args:
            pipeline_parameters (Optional[dict]): The parameter dict to be parsed.
        """
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

    def preprocess(self, inputs: dict, **preprocess_params):
        r"""Preprocess of ZeroShotImageClassificationPipeline

        Args:
            inputs (Union[url, PIL.Image, tensor, numpy]): The image to be classified.
            candidate_labels (List[str]): The candidate labels for classification.
            hypothesis_template (Optional[str]): Prompt for text input.
            max_length (Optional[int]): Max length of tokenizer's output
            padding (Optional[Union[False, "max_length"]]): Padding for max_length
            return_tensors (Optional["ms"]): The type of returned tensors

        Return:
            Processed data.

        Raises:
            ValueError: If candidate_labels or hypothesis_template is None.
        """

        candidate_labels = preprocess_params.pop("candidate_labels", None)
        hypothesis_template = preprocess_params.pop("hypothesis_template",
                                                    "a picture of {}.")
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
        if isinstance(inputs, str):
            inputs = load_image(inputs)
        image_processed = self.image_processor(inputs)
        sentences = [hypothesis_template.format(candidate_label)
                     for candidate_label in candidate_labels]
        input_ids = self.tokenizer(sentences, max_length=max_length, padding=padding,
                                   return_tensors=return_tensors)["input_ids"]
        return {"image_processed": image_processed,
                "input_ids": input_ids, "candidate_labels": candidate_labels}

    def forward(self, model_inputs: dict, **forward_params):
        r"""Forward process

        Args:
            model_inputs (dict): Outputs of preprocess.

        Return:
            Probs dict.
        """
        forward_params.pop("None", None)

        image_processed = model_inputs["image_processed"]
        input_ids = model_inputs["input_ids"]

        logits_per_image, _ = self.model(image_processed, input_ids)
        probs = P.Softmax()(logits_per_image).asnumpy()
        return {"probs": probs, "candidate_labels": model_inputs["candidate_labels"]}

    def postprocess(self, model_outputs: dict, **postprocess_params):
        r"""Postprocess

        Args:
            model_outputs (dict): Outputs of forward process.
            top_k (int): Return top_k probs of result

        Return:
            Classification results.
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
