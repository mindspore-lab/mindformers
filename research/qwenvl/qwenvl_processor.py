# Copyright 2024 Huawei Technologies Co., Ltd
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
QwenVLProcessor
"""

import re
from typing import Optional, Union, List

import PIL
import PIL.Image
import numpy as np
import mindspore as ms
from mindspore.dataset import vision
from mindformers.dataset.transforms.vision_transforms import (
    BatchPILize,
    BatchResize,
    BatchToTensor,
    BatchNormalize
)
from mindformers.models.base_processor import BaseProcessor
from mindformers.models.image_processing_utils import BaseImageProcessor
from mindformers.tools.image_tools import load_image

from qwenvl_transform import QwenVLTransform


class BatchResizeV2(BatchResize):
    """
    Resize a batch of image to the given shape.

    Args:
         image_resolution (int): the target size.
         interpolation (str): interpolate method, default is 'cubic'.
    """

    def __init__(self, image_resolution, interpolation='cubic'):
        super().__init__(image_resolution, interpolation)
        self.resize = vision.Resize(image_resolution, self.interpolation)


class QwenVLImageProcessor(BaseImageProcessor):
    """
    QwenVLImageProcessor.

    Args:
        image_size (int): The target size.
    """

    def __init__(self,
                 image_size: Optional[int] = 224,
                 interpolation: Optional[str] = 'bicubic',
                 mean=None,
                 std=None,
                 is_hwc=False,
                 **kwargs):
        self.pilize = BatchPILize()
        super().__init__(**kwargs)
        if isinstance(image_size, int):
            image_size = (image_size,) * 2
        self.resize = BatchResizeV2(image_size, interpolation=interpolation)
        self.to_tensor = BatchToTensor()
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = BatchNormalize(mean, std, is_hwc)

    def preprocess(self, images: Union[ms.Tensor, PIL.Image.Image, np.ndarray, List[PIL.Image.Image]], **kwargs):
        r"""
        Preprocess Required By Base Processor.

        Args:
            images (ms.Tensor, PIL.Image, numpy.array, List[PIL.Image]): A batch of images.

        Return:
            A 4-rank tensor for a batch of images.
        """
        images = self.pilize(images)
        images = self.resize(images)
        images = self.to_tensor(images)
        images = self.normalize(images)

        kwargs.pop("other", None)
        if isinstance(images, list):
            return ms.Tensor(np.row_stack([np.expand_dims(item, axis=0) for item in images]))
        if len(images.shape) == 4:
            return ms.Tensor(images)
        return ms.Tensor(np.expand_dims(images, axis=0))

    @staticmethod
    def _bhwc_check(image_batch: Union[ms.Tensor, PIL.Image.Image, np.ndarray, List[PIL.Image.Image]]):
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


class QwenVLProcessor(BaseProcessor):
    r"""QwenVL Processor,
    consists of a feature extractor (BaseFeatureEXtractor) for image input,
    and a tokenizer for text input.

    Args:
        image_processor (BaseImageProcessor): Used for process image data.
        tokenizer: Used for process text data.
        max_length (Optional[int]): The length of text tokens.
        padding (Optional[str]): The padding strategy of tokenizer, [None, "max_length"].
        return_tensors (Optional[str]): The type of returned tensors for tokenizer, [None, "ms"].
    """

    def __init__(self, image_processor, tokenizer,
                 max_length=512,
                 image_padding_size=256,
                 prompt=None,
                 padding='max_length', return_tensors='ms', **kwargs):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_length=max_length,
            padding=padding,
            return_tensors=return_tensors, **kwargs)

        self.text_transform = QwenVLTransform(tokenizer,
                                              max_img_size=image_padding_size,
                                              max_length=max_length,
                                              prompt=prompt)

        self.padding_size = self.text_transform.max_img_size

    @staticmethod
    def process_text(text):
        """process text, including padding text and extracting image path"""
        start_tag_index = []
        end_tag_index = []
        for match in re.finditer(r'<img>', text):
            start_tag_index.append((match.start(), match.end()))

        for match in re.finditer(r'</img>', text):
            end_tag_index.append((match.start(), match.end()))

        if len(start_tag_index) != len(end_tag_index):
            raise ValueError("the text has unclosed image tag")

        replaced_text = []
        img_path = []
        last_end = 0
        for start_tag_index_item, end_tag_index_item in zip(start_tag_index, end_tag_index):
            start_tag_start_idx, start_tag_end_idx = start_tag_index_item
            end_tag_start_idx, end_tag_end_idx = end_tag_index_item

            if start_tag_end_idx > end_tag_start_idx:
                raise ValueError("the text has error image tag")

            replaced_text.append(text[last_end:start_tag_start_idx])
            img_path.append(text[start_tag_end_idx:end_tag_start_idx])
            last_end = end_tag_end_idx

        replaced_text.append(text[last_end:])

        img_padding = "<img></img>"
        padded_text = img_padding.join(replaced_text)
        return padded_text, img_path

    def process_query(self, query_ele_list, task):
        """parse query, tokenize and transform text, load images and generate image pos in text"""
        query_text = self.tokenizer.from_list_format(query_ele_list)
        padded_text, img_path_list = self.process_text(query_text)

        text_input_id, img_pos = self.text_transform({"task": task, task: padded_text}, template={task: "{}"})
        image_in_a_text = self.image_processor([load_image(img_path_item) for img_path_item in img_path_list])
        return text_input_id, image_in_a_text.asnumpy(), img_pos

    @staticmethod
    def padding_images(batch_images_list, batch_img_pos_list, max_img_len):
        """padding image and img_pos to max_img_len in a batch"""
        padded_batch_images = []
        padded_batch_img_pos = []

        for image, image_pos in zip(batch_images_list, batch_img_pos_list):
            image_size = image.shape[0]

            if image_size == max_img_len:
                padded_batch_images.append(image)
                padded_batch_img_pos.append(image_pos)
                continue

            repeat = [1] * image_size
            repeat[-1] = max_img_len - image_size + 1
            padded_batch_images.append(np.repeat(image, repeat, axis=0))
            padded_batch_img_pos.append(np.repeat(image_pos, repeat, axis=0))
        return padded_batch_images, padded_batch_img_pos

    def post_process(self, output_ids, queries):
        """post process the origin output ids, it converts <imgpad> token to origin image path"""
        output = []
        for output_ids_item, query in zip(output_ids, queries):
            output_item = self.tokenizer.post_process(output_ids_item, query)
            output.append(output_item)
        return output

    def __call__(self, image_input=None, text_input=None, task="caption"):
        """call function"""
        if isinstance(text_input, list) and text_input and isinstance(text_input[0], dict):
            text_input = [text_input]

        max_img_len = 0
        batch_text_ids = []
        batch_images_list = []
        batch_img_pos_list = []

        for text_input_item in text_input:
            text_in_a_query, image_in_a_query, img_pos_in_a_query = self.process_query(text_input_item, task)
            max_img_len = max(max_img_len, image_in_a_query.shape[0])
            batch_text_ids.append(text_in_a_query)
            batch_images_list.append(image_in_a_query)
            batch_img_pos_list.append(img_pos_in_a_query)

        padded_batch_image, padded_batch_img_pos = self.padding_images(batch_images_list,
                                                                       batch_img_pos_list,
                                                                       max_img_len)

        return {
            "input_ids": np.stack(batch_text_ids, axis=0),
            "image": ms.Tensor(np.stack(padded_batch_image, axis=0), ms.float32),
            "img_pos": ms.Tensor(np.stack(padded_batch_img_pos, axis=0), ms.int32)
        }
