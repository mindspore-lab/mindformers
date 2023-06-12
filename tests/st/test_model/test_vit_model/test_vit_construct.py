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
windows:  pytest .\\tests\\st\\test_model\\test_vit_model\\test_vit_construct.py
linux:  pytest ./tests/st/test_model/test_vit_model/test_vit_construct.py
"""
import os
import numpy as np
# import pytest
import mindspore as ms
from mindformers.models import ViTForImageClassification
from mindformers import MindFormerBook, AutoModel
from mindformers.tools import logger


# @pytest.mark.level0
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.env_onecard
def test_construct():
    """
    Feature: AutoFeatureExtractor class
    Description: Test the from_pretrained functions
    Expectation: NotImplementedError, ValueError
    """
    yaml_path = os.path.join(MindFormerBook.get_project_path(), "configs",
                             "vit", "run_vit_base_p16_224_100ep.yaml")

    AutoModel.show_support_list()
    support_list = AutoModel.get_support_list()
    logger.info(support_list)
    net = AutoModel.from_config(yaml_path)

    fake_image_batch = np.random.random((32, 3, 224, 224))
    fake_image_tensor_batch = ms.Tensor(np.uint8(fake_image_batch), dtype=ms.float32)
    fake_image_target_batch = ms.Tensor(list(range(32)))
    assert isinstance(net, ViTForImageClassification)

    net.set_train(mode=False)
    res_1 = net(fake_image_tensor_batch, fake_image_target_batch)
    assert isinstance(res_1, tuple)
    assert isinstance(res_1[0], ms.Tensor)
    assert isinstance(res_1[1], ms.Tensor)
    assert res_1[0].shape == (32, 1000)
    assert res_1[1].shape == (32,)

    net.set_train(mode=True)
    fake_image_target_batch = ms.nn.OneHot(depth=1000)(fake_image_target_batch)
    res_2 = net(fake_image_tensor_batch, fake_image_target_batch)
    assert isinstance(res_2, ms.Tensor)
    assert res_2.shape == ()

    save_directory = os.path.join(
        MindFormerBook.get_default_checkpoint_save_folder(), 'vit')

    net.save_pretrained(save_directory, save_name='vit_base_p16')
