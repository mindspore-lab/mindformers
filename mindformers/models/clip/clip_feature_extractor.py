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
FeatureExtractors for Clip
"""
import numpy as np
import PIL

import mindspore as ms

from mindformers.mindformer_book import MindFormerBook
from mindformers.dataset import (
    BCHW2BHWC, BatchResize, BatchToTensor,
    BatchNormalize, BatchCenterCrop, BatchPILize
)
from ..base_feature_extractor import BaseImageFeatureExtractor, BaseFeatureExtractor
from ...tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.FEATURE_EXTRACTOR)
class ClipImageFeatureExtractor(BaseImageFeatureExtractor):
    """
    ClipImageProcessor.

    Args:
        image_resolution (int): the target size.
    """
    def __init__(self, image_resolution=224):
        super(ClipImageFeatureExtractor, self).__init__(
            image_resolution=image_resolution
        )

        self.bchw2bhwc = BCHW2BHWC()
        self.batch_pilizer = BatchPILize()
        self.batch_resizer = BatchResize(image_resolution)
        self.batch_crop = BatchCenterCrop(image_resolution)
        self.batch_totensor = BatchToTensor()
        self.batch_normalizer = BatchNormalize()

    def preprocess(self, images, **kwargs):
        """
        Preprocess required by base processor.

        Args:
            images (tensor, PIL.Image, numpy.array, list): a batch of images.

        Return:
            A 4-rank tensor for a batch of images.
        """
        if not self._bhwc_check(images):
            images = self.bchw2bhwc(images)
        images = self.batch_pilizer(images)
        images = self.batch_resizer(images)
        images = self.batch_crop(images)
        images = self.batch_totensor(images)
        images = self.batch_normalizer(images)

        kwargs.pop("other", None)
        if isinstance(images, list):
            return ms.Tensor(np.row_stack([np.expand_dims(item, axis=0) for item in images]))
        if len(images.shape) == 4:
            return ms.Tensor(images)
        return ms.Tensor(np.expand_dims(images, axis=0))

    def _bhwc_check(self, image_batch):
        """_bhwc_check"""
        if isinstance(image_batch, np.ndarray):
            if image_batch.shape[-1] == 3:
                return True
        if isinstance(image_batch, ms.Tensor):
            if image_batch.asnumpy().shape[-1] == 3:
                return True
        if isinstance(image_batch, (list, PIL.Image.Image)):
            return True
        return False


@MindFormerRegister.register(MindFormerModuleType.FEATURE_EXTRACTOR)
class ClipFeatureExtractor(BaseFeatureExtractor):
    """ClipFeatureExtractor"""
    _support_list = MindFormerBook.get_model_support_list()['clip']
    def __init__(self, image_feature_extractor):
        super(ClipFeatureExtractor, self).__init__(
            image_feature_extractor=image_feature_extractor
        )
