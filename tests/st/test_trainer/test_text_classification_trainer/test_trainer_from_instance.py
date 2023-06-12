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
Test module for testing the interface used for mindformers.

windows:
pytest .\\tests\\st\\test_trainer\\test_text_classification_trainer\\test_trainer_from_instance.py
linux:
pytest ./tests/st/test_trainer/test_text_classification_trainer/test_trainer_from_instance.py
"""
# import pytest
import numpy as np
import mindspore as ms
from mindspore.nn import AdamWeightDecay
from mindspore.dataset import GeneratorDataset

from mindformers.models import BertForMultipleChoice, BertConfig
from mindformers.trainer import Trainer
from mindformers.trainer.config_args import ConfigArguments, RunnerConfig
from mindformers.core.lr import build_lr


# @pytest.mark.level0
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.env_onecard
class TestTrainer:
    """A test class for testing Trainer"""
    def generator(self):
        """dataset generator"""
        data = np.random.randint(low=0, high=15, size=(128,)).astype(np.int32)
        input_mask = np.ones_like(data)
        token_type_id = np.zeros_like(data)
        label_ids = np.array([1]).astype(np.int32)
        train_data = (data, input_mask, token_type_id, label_ids)
        for _ in range(64):
            yield train_data

    def test_trainer_train_from_instance(self):
        """
        Feature: Create Trainer From Instance
        Description: Test Trainer API to train from self-define instance API.
        Expectation: TypeError
        """
        runner_config = RunnerConfig(
            epochs=2, batch_size=64,
            sink_mode=False, sink_size=-1, initial_epoch=0,
            has_trained_epoches=0, has_trained_steps=0
        )
        config = ConfigArguments(seed=2022, runner_config=runner_config)
        bert_config = BertConfig.from_pretrained('txtcls_bert_base_uncased')
        bert_txt_cls_model = BertForMultipleChoice(bert_config)

        dataset = GeneratorDataset(self.generator, column_names=["input_ids", "input_mask", "segment_ids", "label_ids"])
        dataset = dataset.batch(batch_size=64)
        steps_per_epoch = dataset.get_dataset_size()
        total_steps = steps_per_epoch * config.runner_config['epochs']

        lr_schedule = build_lr(class_name='WarmUpDecayLR',
                               learning_rate=0.00005,
                               end_learning_rate=0.000001,
                               warmup_steps=int(0.1 * total_steps),
                               decay_steps=total_steps)
        optimizer = AdamWeightDecay(
            params=bert_txt_cls_model.trainable_params(),
            learning_rate=lr_schedule, weight_decay=0.01
        )

        loss_cb = ms.LossMonitor(per_print_times=1)
        callbacks = [loss_cb]

        trainer = Trainer(task='text_classification',
                          model=bert_txt_cls_model,
                          config=config,
                          optimizers=optimizer,
                          train_dataset=dataset,
                          callbacks=callbacks)
        trainer.train(resume_or_finetune_from_checkpoint=False)
