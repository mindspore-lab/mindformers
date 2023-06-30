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
SwinProcessor
"""
import numpy as np
from PIL import Image

from mindspore import Tensor
from mindspore.dataset.vision.transforms import CenterCrop, ToTensor, Normalize, Rescale

from mindformers.mindformer_book import MindFormerBook
from mindformers.dataset import Resize
from mindformers.dataset.base_dataset import BaseDataset
from mindformers.models.base_processor import BaseProcessor, BaseImageProcessor
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


__all__ = ['SwinProcessor', 'SwinImageProcessor']


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class SwinImageProcessor(BaseImageProcessor):
    """
    SwinImageProcessor.

    Args:
        image_resolution (int): the target size.
    """
    def __init__(self, size=224):
        super(SwinImageProcessor, self).__init__(image_resolution=size)
        self.resize = Resize(256, interpolation='cubic')
        self.center_crop = CenterCrop(size)
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
        images = self._format_inputs(images)

        res = []
        for image in images:
            image = self.resize(image)
            image = self.center_crop(image)
            image = self.to_tensor(image)
            image = self.normalize(image)
            res.append(image)
        return Tensor(res)

    def _format_inputs(self, inputs):
        """
        Transform image classification inputs into (bz, h, w, c) or (h, w, c) numpy array.

        Args:
             inputs (tensor, numpy.array, PIL.Image, list, BaseDataset):
             for numpy or tensor input, the channel could be (bz, c, h, w), (c, h, w) or (bz, h, w, c), (h, w, c);
             for list, the item could be PIL.Image, numpy.array, Tensor;
             for BaseDataset, return without any operations.

        Return:
             transformed images:
             for PIL.Image, numpy or tensor input, return a numpy array, the channel is (bz, h, w, c) or (h, w, c);
             for list, return a numpy array for each element;
             for BaseDataset, it is returned directly.
        """
        if not isinstance(inputs, (list, Image.Image, Tensor, np.ndarray, BaseDataset)):
            raise TypeError("input type is not Tensor, numpy, Image, list of Image or MindFormer BaseDataset")

        if isinstance(inputs, list):
            return [self._format_inputs(item) for item in inputs]

        if isinstance(inputs, Image.Image):
            inputs = np.array(inputs)

        if isinstance(inputs, Tensor):
            inputs = inputs.asnumpy()

        if isinstance(inputs, np.ndarray):
            if len(inputs.shape) == 3:
                inputs = np.expand_dims(inputs, 0)
                inputs = self._chw2hwc(inputs)
            elif len(inputs.shape) == 4:
                inputs = self._chw2hwc(inputs)
            else:
                raise ValueError(f"the rank of image_batch should be 3 or 4,"
                                 f" but got {len(inputs.shape)}")
        return inputs

    @staticmethod
    def _chw2hwc(inputs):
        if inputs.shape[-1] != 3:
            inputs = inputs.transpose(0, 2, 3, 1)
        return inputs


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class SwinProcessor(BaseProcessor):
    """
    Swin processor,
    consists of a feature extractor (BaseFeatureEXtractor) for image input.

    Examples:
        >>> from mindformers import MindFormerBook
        >>> from mindformers.models import SwinProcessor
        >>> yaml_path = os.path.join(MindFormerBook.get_project_path(), "configs",
        ...                          "swin", "run_swin_base_p4w7_224_100ep.yaml")
        >>> # build SwinProcessor from pretrained
        >>> pro_a = SwinProcessor.from_pretrained('swin_base_p4w7')
        >>> # build SwinProcessor from config
        >>> pro_b = SwinProcessor.from_pretrained(yaml_path)
    """
    _support_list = MindFormerBook.get_processor_support_list()['swin']

    def __init__(self, image_processor=None, return_tensors='ms'):
        super(SwinProcessor, self).__init__(
            image_processor=image_processor,
            return_tensors=return_tensors
        )
