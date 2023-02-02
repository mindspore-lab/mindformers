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
ZeroShotImageClassificationPipeline

How to run this:
windows:
pytest .\\tests\\st\\test_pipeline
\\test_zero_shot_image_classification_pipeline.py
linux:
pytest ./tests/st/test_pipeline
/test_zero_shot_image_classification_pipeline.py

Note:
    pipeline also supports a dataset input
Example:
    import os
    from mindspore.dataset import Cifar100Dataset
    from mindformers import AutoProcessor, AutoModel
    from mindformers.pipeline import ZeroShotImageClassificationPipeline

    dataset = Cifar100Dataset(dataset_dir = 'cifar-100-binary', shuffle=True, num_samples=20)
    with open(os.path.join('cifar-100-binary', 'fine_label_names.txt'), 'r') as f:
        class_name = f.read().splitlines()

    processor = AutoProcessor.from_pretrained("clip_vit_b_32")
    model = AutoModel.from_pretrained("clip_vit_b_32")
    classifier = ZeroShotImageClassificationPipeline(
    model=model,
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    candidate_labels = class_name,
    hypothesis_template="This is a photo of {}."
    )

    res_dataset = classifier(dataset, batch_size=2, top_k = 3)
"""
import os.path

import pytest

from mindformers.tools.image_tools import load_image
from mindformers.pipeline import ZeroShotImageClassificationPipeline
from mindformers import MindFormerBook


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_zsic_pipeline():
    """
    Feature: ZeroShotImageClassificationPipeline class
    Description: Test the pipeline functions
    Expectation: NotImplementedError, ValueError
    """
    classifier = ZeroShotImageClassificationPipeline(
        model='clip_vit_b_32',
        candidate_labels=["sunflower", "tree", "dog", "cat", "toy"],
        hypothesis_template="This is a photo of {}."
    )

    image_path = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                              'clip', 'sunflower.jpg')
    if not os.path.exists(image_path):
        img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2."
                         "myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
    else:
        img = load_image(image_path)
    res_single = classifier(img)
    res_multi = classifier([img, img, img])

    print(res_single)

    assert len(res_single) == 1
    assert len(res_multi) == 3
