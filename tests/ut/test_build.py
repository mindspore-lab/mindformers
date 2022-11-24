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
Test module for testing the interface used for xformer.
How to run this:
pytest tests/ut/test_build.py
"""

from collections import OrderedDict


def test_build_lr():
    from xformer.common.lr import build_lr
    lr_config = OrderedDict({'lr_schedule': {
        'type': 'CosineDecayLR',
        'min_lr': 0.00001,
        'max_lr': 0.0001,
        'decay_steps': 1000
    }})
    print(build_lr(lr_config.lr_schedule))


def test_build_optim():
    from xformer.common.optim import build_optim


def test_build_loss():
    from xformer.common.loss import build_loss


def test_build_callback():
    from xformer.common.callback import build_callback


def test_build_metric():
    from xformer.common.metric import build_metric


def test_build_trainer():
    from xformer.trainer import build_trainer


def test_build_model():
    from xformer.models import build_model, \
        build_model_config, build_head, build_encoder, build_tokenizer


def test_build_dataset():
    from xformer.dataset import build_dataset, \
        build_dataset_loader, build_mask, build_transforms


def test_build_module():
    from xformer.modules import build_module


def test_build_pipeline():
    from xformer.pipeline import build_pipeline


def test_build_wrapper():
    from xformer.wrapper import build_wrapper


def test_build_processor():
    from xformer.common.lr import build_lr


if __name__ == "__main__":
    test_build_lr()
