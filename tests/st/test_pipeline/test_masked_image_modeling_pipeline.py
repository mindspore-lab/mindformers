# Copyright 2023 Huawei Technologies Co., Ltd
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
Test Module for reconstruct function of
MaskedImageModelingPipeline

How to run this:
windows:
pytest .\\tests\\st\\test_pipeline\\test_masked_image_modeling_pipeline.py
linux:
pytest ./tests/st/test_pipeline/test_masked_image_modeling_pipeline.py
"""
import numpy as np
# import pytest

from mindformers.pipeline import MaskedImageModelingPipeline
from mindformers import ViTMAEImageProcessor

import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE)


# @pytest.mark.level0
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.env_onecard
def test_masked_image_modeling_pipeline():
    """
    Feature: MaskedImageModelingPipeline class
    Description: Test the pipeline functions
    Expectation: NotImplementedError, ValueError
    """
    processor = ViTMAEImageProcessor(size=224)
    reconstructor = MaskedImageModelingPipeline(
        model='mae_vit_base_p16',
        image_processor=processor
    )

    res_multi = reconstructor(np.uint8(np.random.random((5, 3, 255, 255))))
    print(res_multi)
    assert len(res_multi) == 5
