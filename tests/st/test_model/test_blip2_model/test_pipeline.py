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
Test module for testing the blip2 interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_blip2_model/test_pipeline.py
"""
from PIL import Image
import mindspore as ms

from mindformers import pipeline, Pipeline

ms.set_context(mode=0, device_id=7)


class TestBlip2PipelineMethod:
    """A test class for testing pipeline."""
    def setup_method(self):
        """setup method."""
        self.test_llm_list = ['blip2_stage1_classification']

    def test_pipeline(self):
        """
        Feature: pipeline.
        Description: Test pipeline by input model type.
        Expectation: TypeError, ValueError, RuntimeError
        """
        for model_type in self.test_llm_list:
            classifier = pipeline("zero_shot_image_classification",
                                  model=model_type,
                                  candidate_labels=["sunflower", "tree", "dog", "cat", "toy"],
                                  hypothesis_template="A picture of {}.")

            image = Image.new('RGB', (512, 512))
            classifier(image)


class TestBlip2SecondStagePipelineMethod:
    """A test class for testing pipeline of ImageToTextGeneration."""
    def setup_method(self):
        """setup method."""
        self.test_llm_list = ['itt_blip2_stage2_vit_g_llama_7b']

    def test_pipeline(self):
        """
        Feature: pipeline.
        Description: Test pipeline by input model type.
        Expectation: TypeError, ValueError, RuntimeError
        """
        for model_type in self.test_llm_list:
            generator = pipeline("image_to_text_generation",
                                 model=model_type, max_length=32)
            assert isinstance(generator, Pipeline)
            image = Image.new('RGB', (448, 448))
            generator(image, hypothesis_template='a picture of ')
