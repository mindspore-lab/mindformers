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
Test module for testing the qwenvl interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_qwenvl_model/test_trainer.py
"""
import os
import sys
import pytest

MFPATH = os.path.abspath(__file__)
MFPATH = os.path.abspath(MFPATH + '/../../../../../')
sys.path.append(MFPATH)
sys.path.append(MFPATH + '/research/qwenvl')
os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
# pylint: disable=C0413
import mindformers as mf
from mindformers import Trainer, MindFormerConfig, MindFormerRegister, MindFormerModuleType
from mindformers import build_context
from mindformers.models import build_network

from research.qwenvl.qwenvl import QwenVL
from research.qwenvl.qwenvl_config import QwenVLConfig
from research.qwenvl.qwenvl_processor import QwenVLImageProcessor, QwenVLProcessor
from research.qwenvl.qwenvl_tokenizer import QwenVLTokenizer
from research.qwenvl.qwenvl_transform import QwenVLTransform
from research.qwenvl.qwen.optim import AdamWeightDecayX
from research.qwenvl.qwen.qwen_model import QwenForCausalLM
from research.qwenvl.qwen.qwen_config import QwenConfig


def register_modules():
    """register modules"""
    MindFormerRegister.register_cls(AdamWeightDecayX, MindFormerModuleType.OPTIMIZER)
    MindFormerRegister.register_cls(QwenVL, MindFormerModuleType.MODELS)
    MindFormerRegister.register_cls(QwenForCausalLM, MindFormerModuleType.MODELS)
    MindFormerRegister.register_cls(QwenConfig, MindFormerModuleType.CONFIG)
    MindFormerRegister.register_cls(QwenVLConfig, MindFormerModuleType.CONFIG)
    MindFormerRegister.register_cls(QwenVLTokenizer, MindFormerModuleType.TOKENIZER)
    MindFormerRegister.register_cls(QwenVLTransform, MindFormerModuleType.TRANSFORMS)
    MindFormerRegister.register_cls(QwenVLProcessor, MindFormerModuleType.PROCESSOR)
    MindFormerRegister.register_cls(QwenVLImageProcessor, MindFormerModuleType.PROCESSOR)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestQwenVLTrainerMethod:
    """A test class for testing pipeline."""

    def setup_method(self):
        """init task trainer."""
        register_modules()

        if not os.path.exists("./checkpoint_download/qwenvl/qwen.tiktoken"):
            mf.tools.download_tools.download_with_progress_bar(
                'https://modelers.cn/coderepo/web/v1/file/MindSpore-Lab/Qwen-VL/main/media/qwen.tiktoken',
                "./checkpoint_download/qwenvl/qwen.tiktoken"
            )

        config = MindFormerConfig(os.path.join(MFPATH, "research/qwenvl/predict_qwenvl_9.6b.yaml"))
        config.model.model_config.vision_model.model_config.num_hidden_layers = 1
        config.model.model_config.llm_model.model_config.num_layers = 1
        config.processor.tokenizer.vocab_file = "./checkpoint_download/qwenvl/qwen.tiktoken"

        build_context(config)

        model = build_network(config.model)

        self.task_trainer = Trainer(task='image_to_text_generation', model=model, args=config)

    @pytest.mark.run(order=1)
    def test_predict(self):
        """
        Feature: Trainer.predict()
        Description: Test trainer for predict.
        Expectation: TypeError, ValueError, RuntimeError
        """
        img_url = 'https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwenvl/demo.jpeg'
        prompt = 'Describe the image in English:'
        query_item = [{"image": img_url}, {"text": prompt}]
        self.task_trainer.predict(input_data=[query_item], max_length=20)
