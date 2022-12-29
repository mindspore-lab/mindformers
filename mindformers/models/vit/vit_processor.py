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
VitProcessor
"""
import numpy as np
import PIL

import mindspore as ms
from mindspore.dataset.vision import CenterCrop, ToTensor, Normalize, Rescale

from mindformers.dataset import Resize
from mindformers.mindformer_book import MindFormerBook
from ..base_processor import BaseProcessor, BaseImageProcessor
from ...tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class VitImageProcessor(BaseImageProcessor):
    """
    VitImageProcessor.

    Args:
        image_resolution (int): the target size.
    """

    def __init__(self, image_resolution=224):
        super(VitImageProcessor, self).__init__(
            image_resolution=image_resolution
        )
        self.resize = Resize(256, interpolation='cubic')
        self.center_crop = CenterCrop(image_resolution)
        self.to_tensor = ToTensor()
        self.rescale = Rescale(1.0 / 255.0, 0.0)
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)

    def preprocess(self, images, **kwargs):
        """
        Preprocess required by base processor.

        Args:
            images (tensor, PIL.Image, numpy.array, list): a batch of images.

        Return:
            A 4-rank tensor for a batch of images.
        """
        if isinstance(images, PIL.Image.Image):
            images = np.array([images])

        elif isinstance(images, list):
            images = np.array(images)

        elif isinstance(images, np.ndarray):
            if len(images.shape) == 3:
                images = np.expand_dims(images, 0)
            images = images.transpose(0, 2, 3, 1)

        elif isinstance(images, ms.Tensor):
            if len(images.shape) == 3:
                images = np.expand_dims(images, 0)
            images = images.transpose(0, 2, 3, 1)

        elif not isinstance(images, ms.Tensor):
            raise ValueError("input type is not Tensor, numpy, Image, list of Image")

        res = []
        for image in images:
            image = self.resize(image)
            image = self.center_crop(image)
            image = self.to_tensor(image)
            image = self.normalize(image)
            res.append(image)
        return ms.Tensor(res)


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class VitProcessor(BaseProcessor):
    """
    Vit processor,
    consists of a feature extractor (BaseFeatureEXtractor) for image input,
    and a tokenizer (BaseTokenizer) for text input.
    """
    _support_list = MindFormerBook.get_model_support_list()['vit']

    def __init__(self, image_processor=None, return_tensors='ms'):
        super(VitProcessor, self).__init__(
            image_processor=image_processor,
            return_tensors=return_tensors
        )
