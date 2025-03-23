# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test text_transforms"""
import os
import tempfile
import numpy as np
import mindspore as ms
import pytest
from PIL import Image
from mindformers.dataset.transforms.vision_transforms import (
    BCHW2BHWC,
    BatchResize,
    BatchCenterCrop,
    BatchToTensor,
    BatchNormalize,
    BatchPILize,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomCropDecodeResize,
    Resize
)

np.random.seed(0)
data = np.random.randint(0, 255, size=(2, 100, 100, 3)).astype(np.float16)
tmp_dir = tempfile.TemporaryDirectory()
path = tmp_dir.name
jpg_path = os.path.join(path, "test.jpg")
img = Image.new("RGB", (300, 300), (255, 255, 255))
img.save(jpg_path)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bchw2bhwc():
    """
    Feature: test vision_transforms.BCHW2BHWC
    Description: test BCHW2BHWC function
    Expectation: success
    """
    bchw2bhwc = BCHW2BHWC()
    tmp_data = data.transpose((0, 3, 1, 2))
    res = bchw2bhwc(ms.Tensor(tmp_data))
    assert (res == data).all()
    res = bchw2bhwc(ms.Tensor(tmp_data[0]))
    assert (res == data[0]).all()
    with pytest.raises(ValueError):
        assert bchw2bhwc(ms.Tensor(tmp_data[0][0]))
    with pytest.raises(TypeError):
        assert bchw2bhwc(0)
    res = bchw2bhwc([tmp_data[0]])
    assert (res == data[0]).all()
    res = bchw2bhwc(img)
    assert res.size == (300, 300)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batch_resize():
    """
    Feature: test vision_transforms.BatchResize
    Description: test BatchResize function
    Expectation: success
    """
    batch_resize = BatchResize(image_resolution=224)
    res = batch_resize(ms.Tensor(data))
    assert res.shape == (2, 224, 224, 3)
    assert res[0][0][0].tolist() == [821.0, 2044.0, -344.0]
    res = batch_resize(ms.Tensor(data[0]))
    assert res.shape == (224, 224, 3)
    assert res[0][0].tolist() == [821.0, 2044.0, -344.0]
    with pytest.raises(ValueError):
        assert batch_resize(ms.Tensor(data[0][0]))
    with pytest.raises(TypeError):
        assert batch_resize(0)
    res = batch_resize([data[0]])
    assert res[0].shape == (224, 224, 3)
    assert res[0][0][0].tolist() == [821.0, 2044.0, -344.0]
    res = batch_resize(img)
    assert res.shape == (224, 224, 3)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batch_center_crop():
    """
    Feature: test vision_transforms.BatchCenterCrop
    Description: test BatchCenterCrop function
    Expectation: success
    """
    batch_center_crop = BatchCenterCrop(image_resolution=224)
    res = batch_center_crop(ms.Tensor(data))
    assert res.shape == (2, 224, 224, 3)
    assert res[0][64][64].tolist() == [198.0, 9.0, 188.0]
    res = batch_center_crop(ms.Tensor(data[0]))
    assert res.shape == (224, 224, 3)
    assert res[64][64].tolist() == [198.0, 9.0, 188.0]
    with pytest.raises(ValueError):
        assert batch_center_crop(ms.Tensor(data[0][0]))
    with pytest.raises(TypeError):
        assert batch_center_crop(0)
    res = batch_center_crop([data[0]])
    assert res[0].shape == (224, 224, 3)
    res = batch_center_crop(img)
    assert res.size == (224, 224)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batch_to_tensor():
    """
    Feature: test vision_transforms.BatchToTensor
    Description: test BatchToTensor function
    Expectation: success
    """
    batch_to_tensor = BatchToTensor()
    res = batch_to_tensor(ms.Tensor(data))
    assert res.shape == (2, 3, 100, 100)
    assert abs(res[0][0][0][0] - 0.6745098) < 1e-5
    res = batch_to_tensor(ms.Tensor(data[0]))
    assert res.shape == (3, 100, 100)
    assert abs(res[0][0][0] - 0.6745098) < 1e-5
    with pytest.raises(ValueError):
        assert batch_to_tensor(ms.Tensor(data[0][0]))
    with pytest.raises(TypeError):
        assert batch_to_tensor(0)
    res = batch_to_tensor([data[0]])
    assert res[0].shape == (3, 100, 100)
    assert abs(res[0][0][0][0] - 0.6745098) < 1e-5
    res = batch_to_tensor(img)
    assert res.shape == (3, 300, 300)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batch_normalize():
    """
    Feature: test vision_transforms.BatchNormalize
    Description: test BatchNormalize function
    Expectation: success
    """
    batch_normalize = BatchNormalize()
    tmp_data = data.transpose((0, 3, 1, 2))
    res = batch_normalize(ms.Tensor(tmp_data))
    assert res.shape == (2, 3, 100, 100)
    res = batch_normalize(tmp_data[0])
    assert res.shape == (3, 100, 100)
    with pytest.raises(ValueError):
        assert batch_normalize(tmp_data[0][0])
    with pytest.raises(TypeError):
        assert batch_normalize(0)
    res = batch_normalize([tmp_data[0]])
    assert res[0].shape == (3, 100, 100)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batch_pilize():
    """
    Feature: test vision_transforms.BatchPILize
    Description: test BatchPILize function
    Expectation: success
    """
    batch_pilize = BatchPILize()
    tmp_data = data.transpose((0, 3, 1, 2))
    res = batch_pilize(img)
    assert res.size == (300, 300)
    res = batch_pilize([img])
    assert res[0].size == (300, 300)
    with pytest.raises(ValueError):
        assert batch_pilize(tmp_data[0][0])
    with pytest.raises(ValueError):
        assert batch_pilize(0)
    with pytest.raises(TypeError):
        assert batch_pilize([0])


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_random_crop_decode_resize():
    """
    Feature: test vision_transforms.RandomCropDecodeResize
    Description: test RandomCropDecodeResize function
    Expectation: success
    """
    tmp_data = np.fromfile(jpg_path, np.uint8)
    random_crop_decode_resize = RandomCropDecodeResize(size=(50, 75))
    res = random_crop_decode_resize(tmp_data)
    assert res.shape == (50, 75, 3)
    assert res[0][0].tolist() == [255, 255, 255]


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_random_resized_crop():
    """
    Feature: test vision_transforms.RandomResizedCrop
    Description: test RandomResizedCrop function
    Expectation: success
    """
    random_resized_crop = RandomResizedCrop(size=224)
    res = random_resized_crop(data)
    assert res.shape == (2, 224, 224, 3)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_resize():
    """
    Feature: test vision_transforms.Resize
    Description: test Resize function
    Expectation: success
    """
    resize = Resize(size=224)
    res = resize(data)
    assert res.shape == (2, 224, 224, 3)
    assert res[0][0][0].tolist() == [821.0, 2044.0, -344.0]


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_random_horizontal_flip():
    """
    Feature: test vision_transforms.RandomHorizontalFlip
    Description: test RandomHorizontalFlip function
    Expectation: success
    """
    random_horizontal_flip = RandomHorizontalFlip(prob=1.0)
    res = random_horizontal_flip(data)
    assert res.shape == (2, 100, 100, 3)
    assert res[0][0][0].tolist() == [182.0, 235.0, 165.0]
