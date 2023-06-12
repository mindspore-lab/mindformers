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
pytest .\\tests\\st\\test_trainer\\test_token_classification_trainer\\test_trainer_from_instance.py
linux:
pytest ./tests/st/test_trainer/test_token_classification_trainer/test_trainer_from_instance.py
"""
import os
import json
# import pytest
import mindspore as ms
from mindspore.nn import AdamWeightDecay

from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.register.config import MindFormerConfig
from mindformers.models import BertForTokenClassification, BertConfig
from mindformers.trainer import Trainer
from mindformers.trainer.config_args import ConfigArguments, RunnerConfig
from mindformers.dataset.build_dataset import build_dataset
from mindformers.core.lr import build_lr


# @pytest.mark.level0
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.env_onecard
class TestTrainer:
    """A test class for testing Trainer"""

    def setup_method(self):
        """prepare for test"""
        project_path = MindFormerBook.get_project_path()

        config_path = os.path.join(
            project_path, "configs", "tokcls",
            "run_tokcls_bert_base_chinese.yaml"
        )
        new_dataset_dir = "./test_tokcls_trainer/"
        self.config = MindFormerConfig(config_path)
        self.config.train_dataset_task.dataset_config.data_loader.dataset_dir = new_dataset_dir

        self.make_local_directory(new_dataset_dir)
        self.make_dataset(new_dataset_dir, repeat_num=10)

    def test_trainer_train_from_instance(self):
        """
        Feature: Create Trainer From Instance
        Description: Test Trainer API to train from self-define instance API.
        Expectation: TypeError
        """
        self.setup_method()
        runner_config = RunnerConfig(
            epochs=2, batch_size=24,
            sink_mode=False, sink_size=-1, initial_epoch=0,
            has_trained_epoches=0, has_trained_steps=0
        )
        config = ConfigArguments(seed=2022, runner_config=runner_config)
        bert_config = BertConfig.from_pretrained('tokcls_bert_base_chinese')
        bert_token_cls_model = BertForTokenClassification(bert_config)
        bert_token_cls_model.set_train(mode=True)

        dataset = build_dataset(self.config.train_dataset_task)
        steps_per_epoch = dataset.get_dataset_size()
        total_steps = steps_per_epoch * config.runner_config['epochs']

        lr_scheduler = build_lr(class_name='linear', learning_rate=0.00001,
                                warmup_steps=total_steps*0.1, total_steps=total_steps)
        optimizer = AdamWeightDecay(
            params=bert_token_cls_model.trainable_params(),
            learning_rate=lr_scheduler, weight_decay=0.01
        )

        loss_cb = ms.LossMonitor(per_print_times=1)
        callbacks = [loss_cb]

        trainer = Trainer(task='token_classification',
                          model=bert_token_cls_model,
                          config=config,
                          optimizers=optimizer,
                          train_dataset=dataset,
                          callbacks=callbacks)
        trainer.train(resume_or_finetune_from_checkpoint=False)

    def make_local_directory(self, new_dataset_dir):
        """make local directory"""
        os.makedirs(new_dataset_dir, exist_ok=True)

    def make_dataset(self, new_dataset_dir, repeat_num=10):
        """make a fake cluener dataset"""
        sample_data = [{"text": "表身刻有代表日内瓦钟表匠freresoltramare的“fo”字样。", "label": {"position": {"钟表匠": [[9, 11]]}}},
                       {"text": "电影：《杀手：代号47》", "label": {"movie": {"《杀手：代号47》": [[3, 11]]}}},
                       {"text": "的时间会去玩玩星际2。", "label": {"game": {"星际2": [[7, 9]]}}}]

        train_file = os.path.join(new_dataset_dir, "train.json")
        with open(train_file, 'w', encoding='utf-8') as filer:
            for _ in range(repeat_num):
                for data in sample_data:
                    filer.write(json.dumps(data) + '\n')
