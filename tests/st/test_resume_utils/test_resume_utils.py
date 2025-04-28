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
Test module for testing resume training.
How to run this:
    pytest tests/st/test_resume/test_resume_utils.py
"""
import os
import json
import pytest

from mindformers.utils.resume_ckpt_utils import get_resume_checkpoint

cur_dir = os.path.dirname(os.path.abspath(__file__))

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_resume_checkpoint():
    """
    Feature: get_resume_checkpoint
    Description: While resume the checkpoint to train.
    Expectation: success.
    """
    os.makedirs(os.path.join(cur_dir, "rank_0"), exist_ok=True)
    resume_training = os.path.join(cur_dir, "rank_0", "test_rank_0-1_1.ckpt")
    with open(resume_training, 'w') as file:
        pass
    resume_ckpt_ = get_resume_checkpoint(cur_dir, resume_training, "ckpt")
    assert resume_ckpt_ == resume_training

    hyper_param_file = os.path.join(cur_dir, "hyper_param.safetensors")
    with open(hyper_param_file, 'w') as file:
        pass
    resume_ckpt_ = get_resume_checkpoint(cur_dir, True, "safetensors")
    assert resume_ckpt_ == cur_dir

    resume_ckpt_ = get_resume_checkpoint(cur_dir, False, "ckpt")
    assert resume_ckpt_ is None

    last_checkpoint = os.path.join(cur_dir, "rank_0", "test_rank_0-1_2.ckpt")
    with open(last_checkpoint, 'w') as file:
        pass
    resume_ckpt_ = get_resume_checkpoint(cur_dir, True, "ckpt")
    assert resume_ckpt_ == last_checkpoint

    meta_json = os.path.join(cur_dir, "rank_0", "meta.json")
    with open(meta_json, 'w') as file:
        meta_data = {
            'last_epoch': 1,
            'last_step': 1,
            'last_ckpt_file': 'test_rank_0-1_1.ckpt',
        }
        json.dump(meta_data, file, indent=4)
    resume_ckpt_ = get_resume_checkpoint(cur_dir, True, "ckpt")
    assert resume_ckpt_ == resume_training
