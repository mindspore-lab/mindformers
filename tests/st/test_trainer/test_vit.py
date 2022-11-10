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
Test module for testing the ViT
How to run this:
pytest tests/test_vit.py
"""
import os
import pytest
from mindtransformer.data.imagenet_dataset import create_dataset

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_trainer_vit_train():
    """
    Feature: The ViT training test using CPU from python class
    Description: Using cpu to train ViT without basic error
    Expectation: The returned ret is not 0.
    """
    from mindtransformer.trainer import Trainer, TrainingConfig

    class ViTTrainer(Trainer):
        """GPT trainer"""
        def build_model(self, model_config):
            from mindtransformer.models.vit import ViTWithLoss
            my_net = ViTWithLoss(model_config)
            return my_net

        def build_model_config(self):
            from mindtransformer.models.vit import ViTConfig
            return ViTConfig(batch_size=2)

        def build_dataset(self):
            "build fake dataset for testing"
            ds = create_dataset(dataset_path="/home/workspace/mindtransformer/gpt/train",
                                do_train=True,
                                image_size=224,
                                interpolation='BILINEAR',
                                autoaugment=1,
                                mixup=0.2,
                                crop_min=0.05,
                                batch_size=2,
                                num_workers=1,
                                num_classes=1000)
            return ds

        def build_lr(self):
            return 0.01

    trainer = ViTTrainer(TrainingConfig(device_target='CPU', epoch_size=1, sink_size=3, global_batch_size=2))
    trainer.train()

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_trainer_vit_by_cmd():
    """
    Feature: The ViT training test using CPU from python class
    Description: Using cpu to train ViT without basic error
    Expectation: The returned ret is not 0.
    """
    res = os.system("""
            python -m mindtransformer.models.vit.vit_trainer \
                --epoch_size=1 \
                --dataset_name="imagenet" \
                --train_data_path="/home/workspace/mindtransformer/vit/train" \
                --optimizer="adamw"  \
                --parallel_mode="stand_alone" \
                --global_batch_size=2 \
                --init_loss_scale_value=1 \
                --full_batch=False \
                --device_target=CPU  """)

    res1 = os.system("""
            python -m mindtransformer.models.vit.vit_trainer \
                --epoch_size=1 \
                --dataset_name="imagenet" \
                --train_data_path="/home/workspace/mindtransformer/vit/train" \
                --optimizer="adamw"  \
                --parallel_mode="stand_alone" \
                --global_batch_size=2 \
                --init_loss_scale_value=1 \
                --full_batch=False \
                --device_target=GPU  """)

    assert res == 0
    assert res1 == 0
