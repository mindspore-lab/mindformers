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
Test Module for testing TextClassificationTrainDataset.

How to run this:
windows:
pytest .\\tests\\st\\test_trainer\\test_text_classification_trainer\\test_run_mindformer.py
linux:
pytest ./tests/st/test_trainer/test_text_classification_trainer/test_run_mindformer.py
"""
import os
import shutil
import pytest

from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.register.config import MindFormerConfig
from mindformers.tools.download_tools import download_with_progress_bar


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
class TestRunMindFormer:
    """A test class for testing run mindformer"""
    def setup_method_one(self):
        """prepare for test"""
        self.new_dataset_train_dir = "./test_txtcls_run_mindformer/train/"
        self.new_dataset_eval_dir = "./test_txtcls_run_mindformer/eval/"
        shutil.rmtree(self.new_dataset_train_dir)
        shutil.rmtree(self.new_dataset_eval_dir)

    def setup_method_two(self):
        """prepare for test"""
        project_path = MindFormerBook.get_project_path()

        config_path = os.path.join(
            project_path, "configs", "txtcls",
            "run_txtcls_bert_base_uncased.yaml"
        )
        self.config = MindFormerConfig(config_path)

        self.config.train_dataset_task.dataset_config.data_loader.dataset_dir = self.new_dataset_train_dir
        download_with_progress_bar(['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com',
                                    '/XFormer_for_mindspore/bert/downstream_tasks/txtcls/dataset/mnli/train/',
                                    'train.tfrecord'], self.new_dataset_train_dir)

        self.config.eval_dataset_task.dataset_config.data_loader.dataset_dir = self.new_dataset_eval_dir
        download_with_progress_bar(['https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com',
                                    '/XFormer_for_mindspore/bert/downstream_tasks/txtcls/dataset/mnli/eval/',
                                    'eval.tfrecord'], self.new_dataset_eval_dir)

    def test_trainer_train_method(self):
        """
        Feature: TextClassificationDataset by run_mindformer.py
        Description: use TextClassificationDataset with run_mindformer.py
        Expectation: TypeError, ValueError
        """
        yaml_path = os.path.join(MindFormerBook.get_project_path(),
                                 "configs", "txtcls", "run_txtcls_bert_base_uncased.yaml")
        command = "python run_mindformer.py --config " + yaml_path + " --run_mode finetune"
        os.system(command)

    def test_trainer_eval_method(self):
        """
        Feature: TextClassificationDataset by run_mindformer.py
        Description: use TextClassificationDataset with run_mindformer.py
        Expectation: TypeError, ValueError
        """
        yaml_path = os.path.join(MindFormerBook.get_project_path(),
                                 "configs", "txtcls", "run_txtcls_bert_base_uncased.yaml")
        command = "python run_mindformer.py --config " + yaml_path + \
                  " --run_mode eval --load_checkpoint txtcls_bert_base_uncased_mnli"
        os.system(command)
