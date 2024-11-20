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
"""test auto_augment"""
import os
import tempfile
from unittest.mock import patch
from PIL import Image
import numpy as np
import cv2
import pytest
from mindformers.dataset.transforms.auto_augment import (
    AugmentOp,
    pil_interp,
    auto_augment_policy,
    _select_rand_weights,
    rand_augment_transform,
    augment_and_mix_transform,
    auto_augment_transform,
    AugMixAugment,
    RandAugment,
    AutoAugment
)


tmp_dir = tempfile.TemporaryDirectory()
tmp_path = tmp_dir.name
mock_image = np.random.randint(256, size=(224, 224, 3))
img_path = os.path.join(tmp_path, "test.jpg")
cv2.imwrite(img_path, mock_image)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pil_interp():
    """
    Feature: auto_augment.pil_interp
    Description: test pil_interp function
    Expectation: success
    """
    assert pil_interp("bicubic") == 3
    assert pil_interp("hamming") == 5
    assert pil_interp("lanczos") == 1
    assert pil_interp("mock") == 2


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_auto_augment_policy():
    """
    Feature: auto_augment.auto_augment_policy
    Description: test auto_augment_policy function
    Expectation: success
    """
    assert len(auto_augment_policy("v0")) == len(auto_augment_policy("v0r")) == \
           len(auto_augment_policy("original")) == len(auto_augment_policy("originalr")) == 25
    with pytest.raises(ValueError):
        assert auto_augment_policy("mock")


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_select_rand_weights():
    """
    Feature: auto_augment._select_rand_weights
    Description: test _select_rand_weights function
    Expectation: success
    """
    res = _select_rand_weights()
    assert res.tolist() == [0.025, 0.005, 0.0, 0.3, 0.0, 0.005, 0.005, 0.025, 0.005, 0.005, 0.025, 0.2, 0.2, 0.1, 0.1]
    with pytest.raises(ValueError):
        assert _select_rand_weights(weight_idx=1)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_aug_mix_augment():
    """
    Feature: auto_augment.AugMixAugment
    Description: test AugMixAugment function
    Expectation: success
    """
    np.random.seed(0)
    res_ops = augment_and_mix_transform(config_str='augmix-m9-mstd0.5',
                                        hparams={'img_mean': (124, 116, 104), "interpolation": "cubic"})
    aug_mix_augment_basic = AugMixAugment(res_ops.ops, blended=False)
    aug_mix_augment_blended = AugMixAugment(res_ops.ops, blended=True)

    image = Image.open(img_path)
    res_basic = aug_mix_augment_basic(image)
    res_blended = aug_mix_augment_blended(image)
    assert res_basic.size == res_blended.size == (224, 224)
    assert res_basic.mode == res_blended.mode == "RGB"


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_augment_and_mix_transform():
    """
    Feature: auto_augment.augment_and_mix_transform
    Description: test augment_and_mix_transform function
    Expectation: success
    """
    with pytest.raises(ValueError):
        assert augment_and_mix_transform(config_str='rand-m9-mstd0.5', hparams={'img_mean': (124, 116, 104)})
    res = augment_and_mix_transform(config_str='augmix-m9-mstd0.5', hparams={'img_mean': (124, 116, 104)})
    assert len(res.ops) == 13
    assert res.alpha == 1.0
    assert res.depth == -1
    assert res.width == 3
    assert not res.blended


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_auto_augment_transform():
    """
    Feature: auto_augment.auto_augment_transform
    Description: test auto_augment_transform function
    Expectation: success
    """
    with pytest.raises(ValueError):
        assert auto_augment_transform(config_str='rand-m9r-mstd0.5', hparams={'img_mean': (124, 116, 104)})
    res = auto_augment_transform(config_str='v0-m-mstd0.5',
                                 hparams={'img_mean': (124, 116, 104), "interpolation": "cubic"})
    assert len(res.policy) == 25
    auto_augment = AutoAugment(policy=res.policy)
    res = auto_augment(Image.open(img_path))
    assert res.size == (224, 224)
    assert res.mode == "RGB"


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_rand_augment_transform():
    """
    Feature: auto_augment.rand_augment_transform
    Description: test rand_augment_transform function
    Expectation: success
    """
    res = rand_augment_transform(config_str='rand-m9-mstd0.5', hparams={'img_mean': (124, 116, 104)})
    assert len(res.ops) == 15
    assert res.num_layers == 2
    with pytest.raises(ValueError):
        assert rand_augment_transform(config_str='v0-m9-mstd0.5', hparams={'img_mean': (124, 116, 104)})
    res = rand_augment_transform(config_str='rand-m9-n9-w0-inc1-mstd0.5',
                                 hparams={'img_mean': (124, 116, 104), "interpolation": "cubic"})
    assert res.choice_weights.tolist() == \
           [0.025, 0.005, 0.0, 0.3, 0.0, 0.005, 0.005, 0.025, 0.005, 0.005, 0.025, 0.2, 0.2, 0.1, 0.1]
    assert len(res.ops) == 15
    assert res.num_layers == 9
    rand_augment = RandAugment(ops=res.ops)
    res = rand_augment(Image.open(img_path))
    assert res.size == (224, 224)
    assert res.mode == "RGB"
    with pytest.raises(ValueError):
        assert rand_augment_transform(config_str='rand-k9-mstd0.5', hparams={'img_mean': (124, 116, 104)})


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("PIL.__version__", "5.2.0")
def test_auto_augment():
    """
    Feature: auto_augment
    Description: test auto_augment when PIL.__version__ is 5.2.0
    Expectation: success
    """
    np.random.seed(0)
    mean = (0.485, 0.456, 0.406)
    image = Image.open(img_path)
    img_size_min = min(image.size)

    all_policy_use_op = [
        ['AutoContrast', 1, 10], ['Equalize', 1, 10], ['Invert', 1, 10], ['Rotate', 1, 10], ['Posterize', 1, 10],
        ['PosterizeIncreasing', 1, 10], ['PosterizeOriginal', 1, 10], ['Solarize', 1, 10],
        ['SolarizeIncreasing', 1, 10], ['SolarizeAdd', 1, 10], ['Color', 1, 10], ['ColorIncreasing', 1, 10],
        ['Contrast', 1, 10], ['ContrastIncreasing', 1, 10], ['Brightness', 1, 10], ['BrightnessIncreasing', 1, 10],
        ['Sharpness', 1, 10], ['SharpnessIncreasing', 1, 10], ['ShearX', 1, 10], ['ShearY', 1, 10],
        ['TranslateX', 1, 10], ['TranslateY', 1, 10], ['TranslateXRel', 1, 10], ['TranslateYRel', 1, 10]
    ]

    for op_name, p, m in all_policy_use_op:
        aug_op = AugmentOp(name=op_name, prob=p, magnitude=m,
                           hparams={'translate_const': int(img_size_min * 0.45),
                                    'img_mean': tuple([min(255, round(255 * x)) for x in mean]),
                                    "interpolation": "cubic",
                                    "magnitude_std": 0.1})
        image = aug_op(image)

    assert image.size == (224, 224)


@patch("mindformers.dataset.transforms.auto_augment._PIL_VER", (5, 1))
@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_auto_augment_version():
    """
    Feature: auto_augment
    Description: test auto_augment when _PIL_VER is (5, 1)
    Expectation: success
    """
    np.random.seed(0)
    mean = (0.485, 0.456, 0.406)
    image = Image.open(img_path)
    img_size_min = min(image.size)

    all_policy_use_op = [
        ['AutoContrast', 1, 10], ['Equalize', 1, 10], ['Invert', 1, 10], ['Rotate', 1, 10], ['Posterize', 1, 10],
        ['PosterizeIncreasing', 1, 10], ['PosterizeOriginal', 1, 10], ['Solarize', 1, 10],
        ['SolarizeIncreasing', 1, 10], ['SolarizeAdd', 1, 10], ['Color', 1, 10], ['ColorIncreasing', 1, 10],
        ['Contrast', 1, 10], ['ContrastIncreasing', 1, 10], ['Brightness', 1, 10], ['BrightnessIncreasing', 1, 10],
        ['Sharpness', 1, 10], ['SharpnessIncreasing', 1, 10], ['ShearX', 1, 10], ['ShearY', 1, 10],
        ['TranslateX', 1, 10], ['TranslateY', 1, 10], ['TranslateXRel', 1, 10], ['TranslateYRel', 1, 10]
    ]

    for op_name, p, m in all_policy_use_op:
        aug_op = AugmentOp(name=op_name, prob=p, magnitude=m,
                           hparams={'translate_const': int(img_size_min * 0.45),
                                    'img_mean': tuple([min(255, round(255 * x)) for x in mean]),
                                    "interpolation": "cubic"})
        image = aug_op(image)

    assert image.size == (224, 224)
