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
"""
Test module for testing build_model for mindformers.
"""
import pytest

from mindformers.models.build_model import build_encoder, build_head
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister


class DummyEncoder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

class DummyHead:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes

MindFormerRegister.register(MindFormerModuleType.ENCODER, "dummy_enc")(DummyEncoder)
MindFormerRegister.register(MindFormerModuleType.HEAD, "dummy_head")(DummyHead)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_build_encoder():
    """
    Feature: build_encoder()
    Description: Test build_encoder().
    Expectation: Run successfully.
    """
    encoder_config = None
    class_name = None
    encoder = build_encoder(encoder_config, class_name=class_name)
    assert encoder is None
    encoder = build_encoder(class_name=DummyEncoder)
    assert encoder is not None
    encoder_config = {"type": DummyEncoder}
    encoder = build_encoder(encoder_config)
    assert encoder is not None
    encoder_config = [{"type": DummyEncoder}, {"type": DummyEncoder}]
    encoder = build_encoder(encoder_config)
    assert encoder is not None


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_build_head():
    """
    Feature: build_head()
    Description: Test build_head().
    Expectation: Run successfully.
    """
    head_config = None
    class_name = None
    head = build_head(head_config, class_name=class_name)
    assert head is None
    head = build_head(class_name=DummyHead)
    assert head is not None
    head_config = {"type": DummyHead}
    head = build_head(head_config)
    assert head is not None
    head_config = [{"type": DummyHead}, {"type": DummyHead}]
    head = build_head(head_config)
    assert head is not None
