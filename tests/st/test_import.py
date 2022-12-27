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
Test API Import of MindFormers

How to run this:
windows:  pytest .\\tests\\st\\test_import.py
linux:  pytest ./tests/st/test_import.py
"""
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_build_import():
    """
    Feature: Import API from MindFormers
    Description: Test API Import of MindFormers
    Expectation: ImportError
    """
    from mindformers import build_pipeline, build_mask, \
        build_transforms, build_trainer, build_optim, \
        build_loss, build_metric, build_sampler, build_head, \
        build_processor, build_lr, build_model, build_model_config, \
        build_encoder, build_callback, build_wrapper, build_dataset, \
        build_dataset_loader, build_tokenizer
    build_lr()
    build_head()
    build_encoder()
    build_dataset()
    build_tokenizer()
    build_pipeline()
    build_mask()
    build_transforms()
    build_trainer()
    build_optim()
    build_loss()
    build_metric()
    build_sampler()
    build_processor()
    build_model()
    build_model_config()
    build_callback()
    build_wrapper()
    build_dataset_loader()
