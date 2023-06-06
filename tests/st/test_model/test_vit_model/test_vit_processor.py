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
save_pretrained functions of ViTProcessor and AutoProcessor

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_vit_model\\test_vit_processor.py
linux:  pytest ./tests/st/test_model/test_vit_model/test_vit_processor.py
"""
import os
import numpy as np
import mindspore as ms
import pytest

from mindformers import MindFormerBook, AutoProcessor, AutoModel
from mindformers.models import ViTImageProcessor, ViTProcessor
from mindformers.tools import logger


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_vit_processor():
    """
    Feature: ViTProcessor class
    Description: Test the forward, from_pretrained, and
    save_pretrained functions
    Expectation: ValueError
    """
    yaml_path = os.path.join(MindFormerBook.get_project_path(), "configs",
                             "vit", "run_vit_base_p16_224_100ep.yaml")
    img_processor = ViTImageProcessor(size=224)
    save_directory = os.path.join(MindFormerBook.get_default_checkpoint_save_folder(),
                                  'vit')

    ViTProcessor.show_support_list()
    support_list = ViTProcessor.get_support_list()
    logger.info(support_list)

    pro_a = ViTProcessor.from_pretrained('vit_base_p16')
    pro_b = ViTProcessor.from_pretrained(yaml_path)
    pro_c = ViTProcessor(img_processor)

    pro_d = AutoProcessor.from_pretrained('vit_base_p16')
    pro_e = AutoProcessor.from_pretrained(yaml_path)

    fake_image_np = np.uint8(np.random.random((3, 255, 255)))
    fake_image_batch = np.uint8(np.random.random((32, 3, 255, 255)))

    output_a_1 = pro_a(fake_image_np)
    output_a_2 = pro_a(fake_image_np)
    output_a_3 = pro_a(fake_image_batch)
    output_a_4 = pro_a(fake_image_batch)

    output_b_1 = pro_b(fake_image_np)
    output_b_2 = pro_b(fake_image_np)
    output_b_3 = pro_b(fake_image_batch)
    output_b_4 = pro_b(fake_image_batch)

    output_c_1 = pro_c(fake_image_np)
    output_c_2 = pro_c(fake_image_np)
    output_c_3 = pro_c(fake_image_batch)
    output_c_4 = pro_c(fake_image_batch)

    pro_d(fake_image_np)
    pro_e(fake_image_np)

    pro_a.save_pretrained(save_directory, save_name='vit_base_p16')
    pro_b.save_pretrained(save_directory, save_name='vit_base_p16')
    pro_c.save_pretrained(save_directory, save_name='vit_base_p16')

    vit = AutoModel.from_config(yaml_path).set_train(mode=False)
    processed_data = pro_c(fake_image_batch)
    output = vit(**processed_data)

    assert output[0].shape == (32, 1000)

    assert output_a_1['image'].shape == (1, 3, 224, 224)
    assert output_a_2['image'].shape == (1, 3, 224, 224)
    assert output_a_3['image'].shape == (32, 3, 224, 224)
    assert output_a_4['image'].shape == (32, 3, 224, 224)
    assert isinstance(output_a_1['image'], ms.Tensor)
    assert isinstance(output_a_2['image'], ms.Tensor)
    assert isinstance(output_a_3['image'], ms.Tensor)
    assert isinstance(output_a_4['image'], ms.Tensor)

    assert output_b_1['image'].shape == (1, 3, 224, 224)
    assert output_b_2['image'].shape == (1, 3, 224, 224)
    assert output_b_3['image'].shape == (32, 3, 224, 224)
    assert output_b_4['image'].shape == (32, 3, 224, 224)
    assert isinstance(output_b_1['image'], ms.Tensor)
    assert isinstance(output_b_2['image'], ms.Tensor)
    assert isinstance(output_b_3['image'], ms.Tensor)
    assert isinstance(output_b_4['image'], ms.Tensor)

    assert output_c_1['image'].shape == (1, 3, 224, 224)
    assert output_c_2['image'].shape == (1, 3, 224, 224)
    assert output_c_3['image'].shape == (32, 3, 224, 224)
    assert output_c_4['image'].shape == (32, 3, 224, 224)
    assert isinstance(output_c_1['image'], ms.Tensor)
    assert isinstance(output_c_2['image'], ms.Tensor)
    assert isinstance(output_c_3['image'], ms.Tensor)
    assert isinstance(output_c_4['image'], ms.Tensor)
