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
""" test image processor """
import os
from pathlib import Path
import unittest
import tempfile

from research.visualglm.visualglm_processor import VisualGLMImageProcessor
from mindformers.models import (
    Blip2ImageProcessor,
    CLIPImageProcessor,
    ViTMAEImageProcessor,
    SwinImageProcessor,
    SamImageProcessor,
    ViTImageProcessor
)
from mindformers.tools.image_tools import load_image


TEST_IMAGE = load_image("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
DIFF_THRESHOLD = 0.1

VISUALGLM_REMOTE_PATH = 'mindformersinfra/test_visualglm'
BLIP2_REMOTE_PATH = 'mindformersinfra/test_blip2'
CLIP_REMOTE_PATH = 'mindformersinfra/test_clip'
MAE_REMOTE_PATH = 'mindformersinfra/test_mae'
SWIN_REMOTE_PATH = 'mindformersinfra/test_swin'
SAM_REMOTE_PATH = 'mindformersinfra/test_sam'
VIT_REMOTE_PATH = 'mindformersinfra/test_vit'

VISUALGLM_OUTPUT_SHAPE = (1, 3, 224, 224)
BLIP2_OUTPUT_SHAPE = (1, 3, 224, 224)
CLIP_OUTPUT_SHAPE = (1, 3, 224, 224)
MAE_OUTPUT_SHAPE = [(1, 3, 224, 224), (1, 196), (1, 196), (1, 49)]
SWIN_OUTPUT_SHAPE = (1, 3, 224, 224)
SAM_OUTPUT_1_SHAPE = (1, 3, 1024, 1024)
VIT_OUTPUT_SHAPE = (1, 3, 224, 224)

VISUALGLM_OUTPUT_SUM = -54167.4
BLIP2_OUTPUT_SUM = -54167.402
CLIP_OUTPUT_SUM = -40270.586
MAE_OUTPUT_SUM = [-64566.43, 147, 19110, 4720]
SWIN_OUTPUT_SUM = -26959.375
SAM_OUTPUT_1_SUM = -1019078.3
SAM_OUTPUT_2 = (773, 1024)
VIT_OUTPUT_SUM = -26959.375


class ImageProcessorTest(unittest.TestCase):
    """image processor test case"""

    def test_visualglm(self):
        """test visualglm processor"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # test load from_repo
            cache_dir = Path(tmp_dir) / 'visualglm'
            save_dir = Path(tmp_dir) / 'save_dir'
            visualglm_image_processor = VisualGLMImageProcessor.from_pretrained(VISUALGLM_REMOTE_PATH,
                                                                                cache_dir=cache_dir)
            self.assertIsInstance(visualglm_image_processor, VisualGLMImageProcessor)

            # test save to dir
            visualglm_image_processor.save_pretrained(save_directory=save_dir)
            self.assertTrue(os.path.exists(save_dir / 'preprocessor_config.json'))

            # test load from local
            processor_from_local = VisualGLMImageProcessor.from_pretrained(save_dir)
            self.assertIsInstance(processor_from_local, VisualGLMImageProcessor)

            # test preprocess image
            output = visualglm_image_processor(TEST_IMAGE)
            self.assertTrue(output.shape == VISUALGLM_OUTPUT_SHAPE)
            self.assertLess(abs(output.sum() - VISUALGLM_OUTPUT_SUM), DIFF_THRESHOLD)

    def test_blip2(self):
        """test blip2 processor"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # test load from_repo
            cache_dir = Path(tmp_dir) / 'blip2'
            save_dir = Path(tmp_dir) / 'save_dir'
            blip2_image_processor = Blip2ImageProcessor.from_pretrained(BLIP2_REMOTE_PATH, cache_dir=cache_dir)
            self.assertIsInstance(blip2_image_processor, Blip2ImageProcessor)

            # test save to dir
            blip2_image_processor.save_pretrained(save_directory=save_dir)
            self.assertTrue(os.path.exists(save_dir / 'preprocessor_config.json'))

            # test load from local
            processor_from_local = Blip2ImageProcessor.from_pretrained(save_dir)
            self.assertIsInstance(processor_from_local, Blip2ImageProcessor)

            # test preprocess image
            output = blip2_image_processor(TEST_IMAGE)
            self.assertTrue(output.shape == BLIP2_OUTPUT_SHAPE)
            self.assertLess(abs(output.sum() - BLIP2_OUTPUT_SUM), DIFF_THRESHOLD)

    def test_clip(self):
        """test clip processor"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # test load from_repo
            cache_dir = Path(tmp_dir) / 'clip'
            save_dir = Path(tmp_dir) / 'save_dir'
            clip_image_processor = CLIPImageProcessor.from_pretrained(CLIP_REMOTE_PATH, cache_dir=cache_dir)
            self.assertIsInstance(clip_image_processor, CLIPImageProcessor)

            # test save to dir
            clip_image_processor.save_pretrained(save_directory=save_dir)
            self.assertTrue(os.path.exists(save_dir / 'preprocessor_config.json'))

            # test load from local
            processor_from_local = CLIPImageProcessor.from_pretrained(save_dir)
            self.assertIsInstance(processor_from_local, CLIPImageProcessor)

            # test preprocess image
            output = clip_image_processor(TEST_IMAGE)
            self.assertTrue(output.shape == CLIP_OUTPUT_SHAPE)
            self.assertLess(abs(output.sum() - CLIP_OUTPUT_SUM), DIFF_THRESHOLD)

    def test_mae(self):
        """test mae processor"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # test load from_repo
            cache_dir = Path(tmp_dir) / 'mae'
            save_dir = Path(tmp_dir) / 'save_dir'
            mae_image_processor = ViTMAEImageProcessor.from_pretrained(MAE_REMOTE_PATH, cache_dir=cache_dir)
            self.assertIsInstance(mae_image_processor, ViTMAEImageProcessor)

            # test save to dir
            mae_image_processor.save_pretrained(save_directory=save_dir)
            self.assertTrue(os.path.exists(save_dir / 'preprocessor_config.json'))

            # test load from local
            processor_from_local = ViTMAEImageProcessor.from_pretrained(save_dir)
            self.assertIsInstance(processor_from_local, ViTMAEImageProcessor)

            # test preprocess image
            output = mae_image_processor(TEST_IMAGE)
            for i in range(len(output)):
                self.assertTrue(output[i].shape == MAE_OUTPUT_SHAPE[i])
            for i in range(len(output) - 1):
                self.assertLess(abs(output[i].sum() - MAE_OUTPUT_SUM[i]), DIFF_THRESHOLD)

    def test_swin(self):
        """test swin processor"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # test load from_repo
            cache_dir = Path(tmp_dir) / 'swin'
            save_dir = Path(tmp_dir) / 'save_dir'
            swin_image_processor = SwinImageProcessor.from_pretrained(SWIN_REMOTE_PATH, cache_dir=cache_dir)
            self.assertIsInstance(swin_image_processor, SwinImageProcessor)

            # test save to dir
            swin_image_processor.save_pretrained(save_directory=save_dir)
            self.assertTrue(os.path.exists(save_dir / 'preprocessor_config.json'))

            # test load from local
            processor_from_local = SwinImageProcessor.from_pretrained(save_dir)
            self.assertIsInstance(processor_from_local, SwinImageProcessor)

            # test preprocess image
            output = swin_image_processor(TEST_IMAGE)
            self.assertTrue(output.shape == SWIN_OUTPUT_SHAPE)
            self.assertLess(abs(output.sum() - SWIN_OUTPUT_SUM), DIFF_THRESHOLD)

    def test_sam(self):
        """test sam processor"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # test load from_repo
            cache_dir = Path(tmp_dir) / 'sam'
            save_dir = Path(tmp_dir) / 'save_dir'
            sam_image_processor = SamImageProcessor.from_pretrained(SAM_REMOTE_PATH, cache_dir=cache_dir)
            self.assertIsInstance(sam_image_processor, SamImageProcessor)

            # test save to dir
            sam_image_processor.save_pretrained(save_directory=save_dir)
            self.assertTrue(os.path.exists(save_dir / 'preprocessor_config.json'))

            # test load from local
            processor_from_local = SamImageProcessor.from_pretrained(save_dir)
            self.assertIsInstance(processor_from_local, SamImageProcessor)

            # test preprocess image
            output = sam_image_processor(TEST_IMAGE)
            self.assertTrue(output[0].shape == SAM_OUTPUT_1_SHAPE)
            self.assertLess(abs(output[0].sum() - SAM_OUTPUT_1_SUM), DIFF_THRESHOLD)
            self.assertTrue(output[1] == SAM_OUTPUT_2)

    def test_vit(self):
        """test vit processor"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # test load from_repo
            cache_dir = Path(tmp_dir) / 'sam'
            save_dir = Path(tmp_dir) / 'save_dir'
            vit_image_processor = ViTImageProcessor.from_pretrained(VIT_REMOTE_PATH, cache_dir=cache_dir)
            self.assertIsInstance(vit_image_processor, ViTImageProcessor)

            # test save to dir
            vit_image_processor.save_pretrained(save_directory=save_dir)
            self.assertTrue(os.path.exists(save_dir / 'preprocessor_config.json'))

            # test load from local
            processor_from_local = ViTImageProcessor.from_pretrained(save_dir)
            self.assertIsInstance(processor_from_local, ViTImageProcessor)

            # test preprocess image
            output = vit_image_processor(TEST_IMAGE)
            self.assertTrue(output.shape == VIT_OUTPUT_SHAPE)
            self.assertLess(abs(output.sum() - VIT_OUTPUT_SUM), DIFF_THRESHOLD)
