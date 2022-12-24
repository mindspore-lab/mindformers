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
Test Module for testing feature extractor function of
AutoFeatureExtractor and ClipFeatureExtractor

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_vit_model\\test_vit_feature_extractor.py
linux:  pytest ./tests/st/test_model/test_vit_model/test_vit_feature_extractor.py
"""
import os
import numpy as np
import pytest
from PIL import Image
import mindspore as ms
from mindformers.models import VitFeatureExtractor, VitImageFeatureExtractor
from mindformers import MindFormerBook, AutoFeatureExtractor
from mindformers.tools import logger


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_auto_feature_extractor():
    """
    Feature: AutoFeatureExtractor class
    Description: Test the from_pretrained functions
    Expectation: NotImplementedError, ValueError
    """
    yaml_path = os.path.join(MindFormerBook.get_project_path(), "configs",
                             "vit", "model_config", "vit_base_p16.yaml")

    AutoFeatureExtractor.show_support_list()
    support_list = AutoFeatureExtractor.get_support_list()
    logger.info(support_list)

    fe_b = AutoFeatureExtractor.from_pretrained(yaml_path)

    fake_image_np = np.uint8(np.random.random((3, 255, 255)))
    fake_image_batch = np.uint8(np.random.random((5, 3, 255, 255)))
    fake_image_pil = Image.fromarray(np.uint8(fake_image_np.transpose(1, 2, 0)))
    fake_image_pil_list = [fake_image_pil, fake_image_pil, fake_image_pil,
                           fake_image_pil, fake_image_pil]
    fake_image_tensor = ms.Tensor(fake_image_np)
    fake_image_tensor_batch = ms.Tensor(fake_image_batch)

    img_fe = VitImageFeatureExtractor(image_resolution=224)

    VitFeatureExtractor.show_support_list()
    support_list = AutoFeatureExtractor.get_support_list()
    logger.info(support_list)

    fe_d = VitFeatureExtractor.from_pretrained(yaml_path)
    fe_e = VitFeatureExtractor(image_feature_extractor=img_fe)

    res_b_1 = fe_b(fake_image_np)
    res_b_2 = fe_b(fake_image_batch)
    res_b_3 = fe_b(fake_image_pil)
    res_b_4 = fe_b(fake_image_pil_list)
    res_b_5 = fe_b(fake_image_tensor)
    res_b_6 = fe_b(fake_image_tensor_batch)

    assert isinstance(fe_d, VitFeatureExtractor)
    assert isinstance(fe_e, VitFeatureExtractor)

    assert res_b_1.shape == (1, 3, 224, 224)
    assert isinstance(res_b_1, ms.Tensor)
    assert res_b_2.shape == (5, 3, 224, 224)
    assert isinstance(res_b_2, ms.Tensor)
    assert res_b_3.shape == (1, 3, 224, 224)
    assert isinstance(res_b_3, ms.Tensor)
    assert res_b_4.shape == (5, 3, 224, 224)
    assert isinstance(res_b_4, ms.Tensor)
    assert res_b_5.shape == (1, 3, 224, 224)
    assert isinstance(res_b_5, ms.Tensor)
    assert res_b_6.shape == (5, 3, 224, 224)
    assert isinstance(res_b_6, ms.Tensor)

    save_directory = os.path.join(
        MindFormerBook.get_default_checkpoint_save_folder(), 'vit')
    fe_b.save_pretrained(save_directory, save_name='vit_base_p16')
