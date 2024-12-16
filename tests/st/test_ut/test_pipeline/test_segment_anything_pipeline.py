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
"""
Test module for testing pipeline function.
How to run this:
pytest tests/st/test_pipeline/test_pipeline.py
"""
import os
import tempfile
import yaml
import numpy as np
import pytest
from PIL import Image

import mindspore as ms

from mindformers.models.build_config import build_model_config
from mindformers.models.sam import SamImageProcessor
from mindformers import SamModel, MindFormerConfig, ImageEncoderConfig
from mindformers.models.sam.sam_config import PromptEncoderConfig, MaskDecoderConfig
from mindformers.pipeline import SegmentAnythingPipeline


ms.set_context(mode=0)

temp_dir = tempfile.TemporaryDirectory()
path = temp_dir.name
image = Image.new('RGB', (1, 1), color='white')
image.save(os.path.join(path, 'blank.jpg'), 'JPEG')

def mock_yaml():
    """A mock yaml function for testing segment_anything_pipeline."""
    yaml_path = os.path.join("configs", "sam")
    os.makedirs(yaml_path, exist_ok=True)
    sam_yaml_path = os.path.join(yaml_path, "run_sam_vit-b.yaml")
    useless_names = ["_name_or_path", "tokenizer_class", "architectures", "is_encoder_decoder",
                     "is_sample_acceleration", "parallel_config", "moe_config"]
    image_encoder_config = ImageEncoderConfig(layer_norm_eps=1.e-12).to_dict()
    image_encoder_config["type"] = "ImageEncoderConfig"
    prompt_config_ = PromptEncoderConfig().to_dict()
    prompt_config_["type"] = "PromptEncoderConfig"
    decoder_config_ = MaskDecoderConfig(layer_norm_eps=1.e-12).to_dict()
    decoder_config_["type"] = "MaskDecoderConfig"
    image_encoder = {"arch": {"type": "SamImageEncoder"}, "model_config": image_encoder_config}
    prompt_config = {"arch": {"type": "SamPromptEncoder"}, "model_config": prompt_config_}
    decoder_config = {"arch": {"type": "SamMaskDecoder"}, "model_config": decoder_config_}
    sam_ori_config = dict()
    for name in useless_names:
        sam_ori_config.pop(name, None)
    sam_ori_config["image_encoder"] = image_encoder
    sam_ori_config["prompt_config"] = prompt_config
    sam_ori_config["decoder_config"] = decoder_config
    sam_ori_config["checkpoint_name_or_path"] = "sam_vit_b"
    sam_ori_config["num_layers"] = 1
    sam_ori_config["type"] = "SamConfig"
    sam_config = {"model": {"arch": {"type": "SamModel"}, "model_config": sam_ori_config},
                  "processor": {
                      "image_processor": {
                          "img_size": 1024,
                          "mean": [123.675, 116.28, 103.53],
                          "std": [58.395, 57.12, 57.375],
                          "type": "SamImageProcessor"},
                      "type": "SamProcessor"}}
    with open(sam_yaml_path, "w", encoding="utf-8") as w:
        yaml.dump(sam_config, w, default_flow_style=False)

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_segment_anything_pipeline():
    """
    Feature: segment_anything_pipeline interface.
    Description: Test basic function of segment_anything_pipeline api.
    Expectation: success
    """
    mock_yaml()
    sam_config = MindFormerConfig("configs/sam/run_sam_vit-b.yaml")
    model_config = build_model_config(sam_config.model.model_config)
    model = SamModel(model_config)
    image_processor = SamImageProcessor()
    sam_pipeline = SegmentAnythingPipeline(model=model, image_processor=image_processor)

    model_outputs = {"image": os.path.join(path, 'blank.jpg'),
                     "rles": [{"size": [1, 2], "counts": [1]}],
                     "boxes": np.random.random((4, 4)),
                     "iou_preds": np.random.random(1),
                     "points": np.random.random((2, 2)),
                     "stability_score": np.random.random(1),
                     "crop_boxes": np.random.random((2, 4)),
                     }
    output = sam_pipeline.postprocess(model_outputs, seg_image={"test": "test"})[0]
    assert isinstance(output, dict)
    assert False in output["segmentation"]
