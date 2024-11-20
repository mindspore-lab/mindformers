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
"""test mixup"""
from unittest.mock import patch
import pytest
import torch
import numpy as np
from mindformers.tools.logger import logger
from mindformers.dataset.transforms.mixup import Mixup, mixup_target, rand_bbox_minmax


def mock_func(x, axis=0):
    """mock func for test case"""
    if isinstance(x, np.ndarray):
        return x
    x = x.numpy()
    return np.flip(x, axis=axis)


class MockMixUP(Mixup):
    """mock MixUp"""
    def __call__(self, x, target):
        """Mixup apply"""
        # the same to image, label
        if len(x) % 2 != 0:
            if len(x) > 1:
                x = x[:-1]
                logger.warning('Batch size is odd. When using mixup, batch size should be even.'
                               'The last data in batch has been dropped to use mixip.'
                               'you can set "drop_remainder" true in dataset config manually.')
            else:
                logger.warning('Batch size is 1.'
                               'If error occurs, please set "drop_remainder" true in dataset config.')
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing)
        if not isinstance(x, np.ndarray):
            x = x.numpy()
        return x.astype(np.float32), target.astype(np.float32)


# pylint: disable=W0212
@patch("numpy.bool", np.bool_)
@patch("numpy.flip", mock_func)
@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_mixup():
    """
    Feature: test mixup.Mixup
    Description: test Mixup function
    Expectation: success
    """
    data = np.random.randint(0, 255, size=(2, 100, 100, 3)).astype(np.float32)
    label = np.array([[0, 1]])
    # default
    mixup = Mixup()
    res = mixup(data, label)
    assert len(res) == 2
    assert res[0].shape == (2, 100, 100, 3)
    assert res[1].shape == (2, 1000)
    mixup = MockMixUP(mode='elem')
    res = mixup._mix_elem(torch.tensor(data))
    assert res.shape == (2, 1)
    mixup = MockMixUP(mode='pair')
    res = mixup._mix_pair(torch.tensor(data))
    assert res.shape == (2, 1)
    # cutmix_alpha=1.0
    mixup = Mixup(cutmix_alpha=1.0)
    res = mixup(data, label)
    assert len(res) == 2
    assert res[0].shape == (2, 100, 100, 3)
    assert res[1].shape == (2, 1000)
    mixup = MockMixUP(mode='elem', cutmix_alpha=1.0)
    res = mixup._mix_elem(torch.tensor(data))
    assert res.shape == (2, 1)
    mixup = MockMixUP(mode='pair', cutmix_alpha=1.0)
    res = mixup._mix_pair(torch.tensor(data))
    assert res.shape == (2, 1)
    # mixup_alpha=0,cutmix_alpha=1.0
    mixup = Mixup(mixup_alpha=0, cutmix_alpha=1.0)
    res = mixup(data, label)
    assert len(res) == 2
    assert res[0].shape == (2, 100, 100, 3)
    assert res[1].shape == (2, 1000)
    mixup = MockMixUP(mode='elem', mixup_alpha=0, cutmix_alpha=1.0)
    res = mixup._mix_elem(torch.tensor(data))
    assert res.shape == (2, 1)
    mixup = MockMixUP(mode='pair', mixup_alpha=0, cutmix_alpha=1.0)
    res = mixup._mix_pair(torch.tensor(data))
    assert res.shape == (2, 1)
    # cutmix_minmax=[1, 1]
    mixup = Mixup(cutmix_minmax=[0.1, 0.5], mixup_alpha=0, cutmix_alpha=1.0)
    res = mixup(data, label)
    assert len(res) == 2
    assert res[0].shape == (2, 100, 100, 3)
    assert res[1].shape == (2, 1000)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_illegal_cutmix_minmax():
    """
    Feature: test mixup.Mixup
    Description: test Mixup when cutmix_minmax is illegal
    Expectation: success
    """
    with pytest.raises(ValueError):
        assert Mixup(cutmix_minmax=[1])
    with pytest.raises(ValueError):
        assert rand_bbox_minmax("", [1])


@patch("numpy.bool", np.bool_)
@patch("numpy.flip", mock_func)
@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_illegal_mixup_alpha_cutmix_alpha():
    """
    Feature: test mixup.Mixup
    Description: test Mixup when cutmix_minmax and mixup_alpha are illegal
    Expectation: success
    """
    data = np.random.randint(0, 255, size=(2, 100, 100, 3)).astype(np.float32)
    label = np.array([[0, 1]])
    with pytest.raises(ValueError):
        mixup = Mixup(mixup_alpha=0, cutmix_alpha=0)
        assert mixup(data, label)
    with pytest.raises(ValueError):
        mixup = MockMixUP(mode='elem', mixup_alpha=0, cutmix_alpha=0)
        assert mixup._mix_elem(torch.tensor(data))
