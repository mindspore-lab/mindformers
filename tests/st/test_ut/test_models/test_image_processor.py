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
"""test image processor."""
import os
from pathlib import Path
import tempfile

from mindformers.models import (
    CLIPImageProcessor
)
from mindformers.tools.image_tools import load_image


TEST_IMAGE = load_image("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
DIFF_THRESHOLD = 0.1

CLIP_REMOTE_PATH = 'mindformersinfra/test_clip'

CLIP_OUTPUT_SHAPE = (1, 3, 224, 224)

CLIP_OUTPUT_SUM = -40270.586


def run_processor(
        name,
        processor,
        remote_path,
        output_shape,
        output_sum
):
    """run processor func"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # test init processor from repo
        cache_dir = Path(tmp_dir) / name
        save_dir = Path(tmp_dir) / 'save_dir'
        image_processor = processor.from_pretrained(remote_path, cache_dir=cache_dir)
        assert isinstance(image_processor, processor)

        # test save_pretrained
        image_processor.save_pretrained(save_directory=save_dir)
        assert os.path.exists(save_dir / 'preprocessor_config.json')

        # test init processor from local dir
        image_processor_local = processor.from_pretrained(save_dir)
        assert isinstance(image_processor_local, processor)

        # test preprocess image
        output = image_processor(TEST_IMAGE)
        assert output.shape == output_shape
        assert abs(output.sum() - output_sum) < DIFF_THRESHOLD


class TestImageProcessor:
    """A test class for testing image processor."""

    def test_clip_image_processor(self):
        """test clip image processor."""
        run_processor(
            name='clip',
            processor=CLIPImageProcessor,
            remote_path=CLIP_REMOTE_PATH,
            output_shape=CLIP_OUTPUT_SHAPE,
            output_sum=CLIP_OUTPUT_SUM
        )
