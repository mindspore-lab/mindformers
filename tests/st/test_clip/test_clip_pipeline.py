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
Test Module for basic function of pipeline

How to run this:
windows:  pytest .\\tests\\st\\test_clip\\test_clip_pipeline.py
linux:  pytest ./tests/st/test_clip/test_clip_pipeline.py
"""
import os
import pytest
from mindformers import pipeline
from mindformers.tools.image_tools import load_image
from mindformers import AutoModel, AutoProcessor
from mindformers import MindFormerBook


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_pipeline():
    """
    Feature: pipline function
    Description: Test the pipeline functions
    Expectation: NotImplementedError, ValueError
    """
    classifier_a = pipeline("zero_shot_image_classification",
                            candidate_labels=["sunflower", "tree", "dog", "cat", "toy"])

    classifier_b = pipeline("zero_shot_image_classification",
                            model="clip_vit_b_32",
                            candidate_labels=["sunflower", "tree", "dog", "cat", "toy"])

    model = AutoModel.from_pretrained("clip_vit_b_32")
    processor = AutoProcessor.from_pretrained("clip_vit_b_32")
    classifier_c = pipeline("zero_shot_image_classification",
                            model=model,
                            feature_extractor=processor.feature_extractor,
                            tokenizer=processor.tokenizer,
                            candidate_labels=["sunflower", "tree", "dog", "cat", "toy"])

    image_path = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                              'clip', 'sunflower.jpg')
    if not os.path.exists(image_path):
        img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2."
                         "myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
    else:
        img = load_image(image_path)

    res_a = classifier_a(img)
    res_b = classifier_b(img)
    res_c = classifier_c(img)

    assert len(res_a) == 1
    assert len(res_b) == 1
    assert len(res_c) == 1
