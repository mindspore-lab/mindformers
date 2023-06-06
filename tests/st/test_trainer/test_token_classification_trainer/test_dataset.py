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
Test Module for testing clip_pretrain dataset for clip trainer.

How to run this:
windows:
pytest .\\tests\\st\\test_trainer\\test_token_classification_trainer\\test_dataset.py
linux:
pytest ./tests/st/test_trainer/test_token_classification_trainer/test_dataset.py
"""
import os
import json
import pytest

from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.register.config import MindFormerConfig
from mindformers.dataset.build_dataset import build_dataset


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
class TestTokenClassificationTrainDataset:
    """A test class for testing TestTokenClassificationTrainDataset classes"""
    def setup_method(self):
        """prepare for test"""
        project_path = MindFormerBook.get_project_path()

        config_path = os.path.join(
            project_path, "configs", "tokcls",
            "run_tokcls_bert_base_chinese_cluener.yaml"
        )

        new_dataset_dir = "./test_tokcls_dataset/"
        self.config = MindFormerConfig(config_path)
        self.config.train_dataset_task.dataset_config.data_loader.dataset_dir = new_dataset_dir

        self.make_local_directory(new_dataset_dir)
        self.make_dataset(new_dataset_dir, repeat_num=10)

    def test_dataset(self):
        """
        Feature: TokenClassificationTrainDataset
        Description: A data set for token classification train dataset
        Expectation: TypeError, ValueError
        """
        self.setup_method()
        data_loader = build_dataset(self.config.train_dataset_task)
        for item in data_loader:
            assert item[0].shape == (24, 128)
            assert item[1].shape == (24, 128)
            assert item[2].shape == (24, 128)
            assert item[3].shape == (24, 128)

    def make_local_directory(self, new_dataset_dir):
        """make local directory"""
        os.makedirs(new_dataset_dir, exist_ok=True)

    def make_dataset(self, new_dataset_dir, repeat_num=10):
        """make a fake cluener dataset"""
        train_data = [{"text": "表身刻有代表日内瓦钟表匠freresoltramare的“fo”字样。", "label": {"position": {"钟表匠": [[9, 11]]}}},
                      {"text": "电影：《杀手：代号47》", "label": {"movie": {"《杀手：代号47》": [[3, 11]]}}},
                      {"text": "的时间会去玩玩星际2。", "label": {"game": {"星际2": [[7, 9]]}}}]

        train_file = os.path.join(new_dataset_dir, "train.json")
        with open(train_file, 'w', encoding='utf-8') as filer:
            for _ in range(repeat_num):
                for data in train_data:
                    filer.write(json.dumps(data) + '\n')
