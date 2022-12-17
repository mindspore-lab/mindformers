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
Test Module for testing forward, from_pretrained, and
save_pretrained functions of ClipProcessor and AutoProcessor

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_clip_model\\test_clip_processor.py
linux:  pytest ./tests/st/test_model/test_clip_model/test_clip_processor.py
"""
import os
import numpy as np
import pytest
import mindspore as ms

from mindformers import MindFormerBook, AutoProcessor, AutoModel
from mindformers.models import (
    ClipFeatureExtractor, ClipImageFeatureExtractor,
    ClipProcessor, ClipTokenizer
)
from mindformers.tools import logger


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_clip_processor():
    """
    Feature: ClipProcessor class
    Description: Test the forward, from_pretrained, and
    save_pretrained functions
    Expectation: ValueError
    """
    yaml_path = os.path.join(MindFormerBook.get_project_path(), "configs",
                             "clip", "model_config", "clip_vit_b_32.yaml")
    img_fe = ClipImageFeatureExtractor(image_resolution=224)
    feature_extractor = ClipFeatureExtractor(img_fe)
    tokenizer = ClipTokenizer()
    save_directory = os.path.join(MindFormerBook.get_default_checkpoint_save_folder(),
                                  'clip')

    ClipProcessor.show_support_list()
    support_list = ClipProcessor.get_support_list()
    logger.info(support_list)

    pro_a = ClipProcessor.from_pretrained('clip_vit_b_32')
    pro_b = ClipProcessor.from_pretrained(yaml_path)
    pro_c = ClipProcessor(feature_extractor, tokenizer)

    pro_d = AutoProcessor.from_pretrained('clip_vit_b_32')
    pro_e = AutoProcessor.from_pretrained(yaml_path)

    fake_image_np = np.random.random((3, 578, 213))
    fake_image_batch = np.random.random((5, 3, 578, 213))
    fake_text_np = "a boy"
    fake_text_batch = ["a boy", "a girl", "a women", "a men"]

    output_a_1 = pro_a(fake_image_np, fake_text_np)
    output_a_2 = pro_a(fake_image_np, fake_text_batch)
    output_a_3 = pro_a(fake_image_batch, fake_text_np)
    output_a_4 = pro_a(fake_image_batch, fake_text_batch)

    output_b_1 = pro_b(fake_image_np, fake_text_np)
    output_b_2 = pro_b(fake_image_np, fake_text_batch)
    output_b_3 = pro_b(fake_image_batch, fake_text_np)
    output_b_4 = pro_b(fake_image_batch, fake_text_batch)

    output_c_1 = pro_c(fake_image_np, fake_text_np)
    output_c_2 = pro_c(fake_image_np, fake_text_batch)
    output_c_3 = pro_c(fake_image_batch, fake_text_np)
    output_c_4 = pro_c(fake_image_batch, fake_text_batch)

    pro_d(fake_image_np, fake_text_np)
    pro_e(fake_image_np, fake_text_batch)

    pro_a.save_pretrained(save_directory, save_name='clip_vit_b_32')
    pro_c.save_pretrained(save_directory, save_name='clip_vit_b_32')
    pro_b.save_pretrained(save_directory, save_name='clip_vit_b_32')

    clip = AutoModel.from_pretrained("clip_vit_b_32")
    processed_data = pro_a(fake_image_batch, fake_text_batch)
    output = clip(**processed_data)

    assert output[0].shape == (5, 4)
    assert output[1].shape == (4, 5)

    assert output_a_1['image'].shape == (1, 3, 224, 224)
    assert output_a_2['image'].shape == (1, 3, 224, 224)
    assert output_a_3['image'].shape == (5, 3, 224, 224)
    assert output_a_4['image'].shape == (5, 3, 224, 224)
    assert output_a_1['text'].shape == (1, 77)
    assert output_a_2['text'].shape == (4, 77)
    assert output_a_3['text'].shape == (1, 77)
    assert output_a_4['text'].shape == (4, 77)
    assert isinstance(output_a_1['image'], ms.Tensor)
    assert isinstance(output_a_2['image'], ms.Tensor)
    assert isinstance(output_a_3['image'], ms.Tensor)
    assert isinstance(output_a_4['image'], ms.Tensor)
    assert isinstance(output_a_1['text'], ms.Tensor)
    assert isinstance(output_a_2['text'], ms.Tensor)
    assert isinstance(output_a_3['text'], ms.Tensor)
    assert isinstance(output_a_4['text'], ms.Tensor)

    assert output_b_1['image'].shape == (1, 3, 224, 224)
    assert output_b_2['image'].shape == (1, 3, 224, 224)
    assert output_b_3['image'].shape == (5, 3, 224, 224)
    assert output_b_4['image'].shape == (5, 3, 224, 224)
    assert output_b_1['text'].shape == (1, 77)
    assert output_b_2['text'].shape == (4, 77)
    assert output_b_3['text'].shape == (1, 77)
    assert output_b_4['text'].shape == (4, 77)
    assert isinstance(output_b_1['image'], ms.Tensor)
    assert isinstance(output_b_2['image'], ms.Tensor)
    assert isinstance(output_b_3['image'], ms.Tensor)
    assert isinstance(output_b_4['image'], ms.Tensor)
    assert isinstance(output_b_1['text'], ms.Tensor)
    assert isinstance(output_b_2['text'], ms.Tensor)
    assert isinstance(output_b_3['text'], ms.Tensor)
    assert isinstance(output_b_4['text'], ms.Tensor)

    assert output_c_1['image'].shape == (1, 3, 224, 224)
    assert output_c_2['image'].shape == (1, 3, 224, 224)
    assert output_c_3['image'].shape == (5, 3, 224, 224)
    assert output_c_4['image'].shape == (5, 3, 224, 224)
    assert output_c_1['text'].shape == (1, 77)
    assert output_c_2['text'].shape == (4, 77)
    assert output_c_3['text'].shape == (1, 77)
    assert output_c_4['text'].shape == (4, 77)
    assert isinstance(output_c_1['image'], ms.Tensor)
    assert isinstance(output_c_2['image'], ms.Tensor)
    assert isinstance(output_c_3['image'], ms.Tensor)
    assert isinstance(output_c_4['image'], ms.Tensor)
    assert isinstance(output_c_1['text'], ms.Tensor)
    assert isinstance(output_c_2['text'], ms.Tensor)
    assert isinstance(output_c_3['text'], ms.Tensor)
    assert isinstance(output_c_4['text'], ms.Tensor)
