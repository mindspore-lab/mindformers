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
AutoFeatureExtractor and SwinFeatureExtractor

How to run this:
windows:  pytest .\\tests\\st\\test_model\\test_swin_model\\test_swin_construct.py
linux:  pytest ./tests/st/test_model/test_swin_model/test_swin_construct.py
"""
import os
# import pytest
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype

from mindformers.models import SwinForImageClassification
from mindformers import MindFormerBook, AutoConfig, AutoModel
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
    config = AutoConfig.from_pretrained('swin_base_p4w7')
    config.batch_size = 32
    config.embed_dim = 24
    config.depths = [1, 1, 2, 1]
    config.num_heads = [1, 2, 3, 4]

    AutoModel.show_support_list()
    support_list = AutoModel.get_support_list()
    logger.info(support_list)
    net = AutoModel.from_config(config)

    fake_image_batch = np.random.random((32, 3, 224, 224))
    fake_image_tensor_batch = Tensor(np.uint8(fake_image_batch), dtype=mstype.float32)
    fake_image_target_batch = Tensor(list(range(32)))
    assert isinstance(net, SwinForImageClassification)

    net.set_train(mode=False)
    res_1 = net(fake_image_tensor_batch, fake_image_target_batch)
    assert isinstance(res_1, tuple)
    assert isinstance(res_1[0], Tensor)
    assert isinstance(res_1[1], Tensor)
    assert res_1[0].shape == (32, 1000)
    assert res_1[1].shape == (32,)

    net.set_train(mode=True)
    fake_image_target_batch = nn.OneHot(depth=1000)(fake_image_target_batch)
    res_2 = net(fake_image_tensor_batch, fake_image_target_batch)
    assert isinstance(res_2, Tensor)
    assert res_2.shape == ()

    save_directory = os.path.join(
        MindFormerBook.get_default_checkpoint_save_folder(), 'swin')

    net.save_pretrained(save_directory, save_name='swin_base_p4w7')
