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
pytest .\\tests\\st\\test_trainer\\test_question_answering_trainer\\test_trainer_from_instance.py
linux:
pytest ./tests/st/test_trainer/test_question_answering_trainer/test_trainer_from_instance.py
"""
import os
import json
# import shutil
# import pytest
import mindspore as ms
from mindspore.nn import AdamWeightDecay

from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.register.config import MindFormerConfig
from mindformers.models import BertForQuestionAnswering, BertConfig
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
            project_path, "configs", "qa",
            "run_qa_bert_base_uncased.yaml"
        )
        self.new_dataset_dir = "./squad/"
        self.config = MindFormerConfig(config_path)
        self.config.train_dataset_task.dataset_config.data_loader.dataset_dir = self.new_dataset_dir

        self.make_local_directory()
        self.make_dataset(repeat_num=20)

    def test_trainer_train_from_instance(self):
        """
        Feature: Create Trainer From Instance
        Description: Test Trainer API to train from self-define instance API.
        Expectation: TypeError
        """
        self.setup_method()
        runner_config = RunnerConfig(
            epochs=2, batch_size=12,
            sink_mode=False, sink_size=-1, initial_epoch=0,
            has_trained_epoches=0, has_trained_steps=0
        )
        config = ConfigArguments(seed=2022, runner_config=runner_config)
        bert_config = BertConfig.from_pretrained('qa_bert_base_uncased')
        bert_token_cls_model = BertForQuestionAnswering(bert_config)
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

        trainer = Trainer(task='question_answering',
                          model=bert_token_cls_model,
                          config=config,
                          optimizers=optimizer,
                          train_dataset=dataset,
                          callbacks=callbacks)
        trainer.train(resume_or_finetune_from_checkpoint=False)

    def make_local_directory(self):
        """make local directory"""
        os.makedirs(self.new_dataset_dir, exist_ok=True)

    # def teardown_method(self):
    #     """delete local directory"""
    #     shutil.rmtree(self.new_dataset_dir)

    def make_dataset(self, repeat_num=20):
        """make a fake SQuAD dataset"""
        test_data = {}
        entry = {'title': 'Prime_minister', 'paragraphs': [{'context': 'A prime minister is the' \
            'most senior minister of cabinet in the executive branch of government, often in a parliamentary' \
            'or semi-presidential system.', 'qas': [{'answers': [{'answer_start': 63, 'text': 'executive'}], \
            'question': 'What branch of government does the prime minister lead?', \
            'id': '56dd20d966d3e219004dabf3'}]}]}

        test_data["data"] = []
        for _ in range(repeat_num):
            test_data["data"].append(entry)

        train_file = os.path.join(self.new_dataset_dir, "train-v1.1.json")
        with open(train_file, 'w', encoding='utf-8') as filer:
            filer.write(json.dumps(test_data))
