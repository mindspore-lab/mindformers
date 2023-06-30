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
CLIPProcessor
"""
from typing import Optional, Union, List
import numpy as np
import PIL

import mindspore as ms

from mindformers.dataset import (
    BCHW2BHWC, BatchResize, BatchToTensor,
    BatchNormalize, BatchCenterCrop, BatchPILize
)
from mindformers.mindformer_book import MindFormerBook
from ..base_processor import BaseImageProcessor
from ..base_processor import BaseProcessor
from ...tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class CLIPImageProcessor(BaseImageProcessor):
    """
    CLIPImageProcessor.

    Args:
        image_resolution (int): The target size.

    Examples:
        >>> from mindformers import CLIPImageProcessor
        >>> from mindformers.tools.image_tools import load_image
        >>> processor = CLIPImageProcessor(image_resolution=256)
        >>> image = load_image("https://ascend-repo-modelzoo.obs.cn-east-2."
        ...                    "myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
        >>> processor(image)
            Tensor(shape=[1, 3, 256, 256], dtype=Float32, value=
            [[[[-1.52949083e+000, -1.52949083e+000, ... -1.48569560e+000, -1.50029397e+000],
            [-1.52949083e+000, -1.52949083e+000, ... -1.50029397e+000, -1.50029397e+000],
            [-1.51489246e+000, -1.51489246e+000, ... -1.48569560e+000, -1.48569560e+000],
            ...
            ...
            [8.66091192e-001, 8.80311251e-001, ... -1.36645925e+000, -1.45177972e+000],
            [8.09210956e-001, 8.23431015e-001, ... -1.29535890e+000, -1.43755960e+000],
            [7.09670484e-001, 7.94990897e-001, ... -1.26691878e+000, -1.42333949e+000]]]])
    """
    def __init__(self, image_resolution: Optional[int] = 224):
        super(CLIPImageProcessor, self).__init__(
            image_resolution=image_resolution)
        self.bchw2bhwc = BCHW2BHWC()
        self.batch_pilizer = BatchPILize()
        self.batch_resizer = BatchResize(image_resolution)
        self.batch_crop = BatchCenterCrop(image_resolution)
        self.batch_totensor = BatchToTensor()
        self.batch_normalizer = BatchNormalize()

    def preprocess(self, images: Union[ms.Tensor, PIL.Image.Image,
                                       np.ndarray, List[PIL.Image.Image]], **kwargs):
        r"""
        Preprocess Required By Base Processor.

        Args:
            images (ms.Tensor, PIL.Image, numpy.array, List[PIL.Image]): A batch of images.

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

    def _bhwc_check(self, image_batch: Union[ms.Tensor, PIL.Image.Image,
                                             np.ndarray, List[PIL.Image.Image]]):
        r"""Bhwc_check"""
        if isinstance(image_batch, np.ndarray):
            if image_batch.shape[-1] == 3:
                return True
        if isinstance(image_batch, ms.Tensor):
            if image_batch.asnumpy().shape[-1] == 3:
                return True
        if isinstance(image_batch, (list, PIL.Image.Image)):
            return True
        return False


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class CLIPProcessor(BaseProcessor):
    r"""CLIP Processor,
    consists of a feature extractor (BaseFeatureEXtractor) for image input,
    and a tokenizer (BaseTokenizer) for text input.

    Args:
        image_processor (BaseImageProcessor): Used for process image data.
        tokenizer (BaseTokenizer): Used for process text data.
        max_length (Optional[int]): The length of text tokens.
        padding (Optional[str]): The padding strategy of tokenizer, [None, "max_length"].
        return_tensors (Optional[str]): The type of returned tensors for tokenizer, [None, "ms"].

    Examples:
        >>> from mindformers import CLIPProcessor
        >>> from mindformers.tools.image_tools import load_image
        >>> image = load_image("https://ascend-repo-modelzoo.obs.cn-east-2."
        ...                    "myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
        >>> text = ["a boy", "a girl"]
        >>> CLIPProcessor.show_support_list()
            INFO - support list of CLIP Processor is:
            INFO -    ['clip_vit_b_32']
            INFO - -------------------------------------
        >>> processor = CLIPProcessor.from_pretrained('clip_vit_b_32')
        >>> processor(image, text)
            {'image': Tensor(shape=[1, 3, 224, 224], dtype=Float32, value=
            [[[[-1.52949083e+000, -1.52949083e+000,... -1.48569560e+000, -1.50029397e+000],
            [-1.52949083e+000, -1.52949083e+000, ... -1.50029397e+000, -1.50029397e+000],
            [-1.50029397e+000, -1.50029397e+000 ... -1.48569560e+000, -1.50029397e+000],
            ...
            [8.23431015e-001, 8.80311251e-001, ... -1.33801913e+000, -1.43755960e+000],
            [7.80770779e-001, 8.37651074e-001, ... -1.23847866e+000, -1.39489937e+000],
            [6.10130012e-001, 7.66550720e-001, ... -1.19581854e+000, -1.38067937e+000]]]]),
             'text': Tensor(shape=[2, 77], dtype=Int32, value=
            [[49406,   320,  1876 ...     0,     0,     0],
            [49406,   320,  1611 ...     0,     0,     0]])}
    """
    _support_list = MindFormerBook.get_processor_support_list()['clip']

    def __init__(self, image_processor, tokenizer,
                 max_length=77, padding='max_length', return_tensors='ms'):
        super(CLIPProcessor, self).__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_length=max_length,
            padding=padding,
            return_tensors=return_tensors)
