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
Test Module for classification function of
ImageClassificationPipeline

How to run this:
windows:
pytest .\\tests\\st\\test_pipeline\\test_image_classification_pipeline.py
linux:
pytest ./tests/st/test_pipeline/test_image_classification_pipeline.py

Note:
    pipeline also supports a dataset input
Example:
    import os
    from mindspore.dataset import Cifar100Dataset
    from mindformers import AutoProcessor, AutoModel
    from mindformers.pipeline import ImageClassificationPipeline

    dataset = Cifar100Dataset(dataset_dir = 'cifar-100-binary', shuffle=True, num_samples=20)
    with open(os.path.join('cifar-100-binary', 'fine_label_names.txt'), 'r') as f:
        class_name = f.read().splitlines()

    processor = AutoProcessor.from_pretrained("vit_base_p16")
    model = AutoModel.from_pretrained("vit_base_p16")
    classifier = ImageClassificationPipeline(
    model=model,
    image_processor=processor.image_processor
    )

    res_dataset = classifier(dataset, batch_size=2, top_k = 3)
"""
import numpy as np
import pytest

from mindformers.pipeline import ImageClassificationPipeline
from mindformers import VitImageProcessor


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_image_classification_pipeline():
    """
    Feature: ImageClassificationPipeline class
    Description: Test the pipeline functions
    Expectation: NotImplementedError, ValueError
    """
    processor = VitImageProcessor(image_resolution=224)
    classifier = ImageClassificationPipeline(
        model='vit_base_p16',
        image_processor=processor,
        top_k=5
    )

    res_multi = classifier(np.uint8(np.random.random((5, 3, 255, 255))))
    print(res_multi)
    assert len(res_multi) == 5
