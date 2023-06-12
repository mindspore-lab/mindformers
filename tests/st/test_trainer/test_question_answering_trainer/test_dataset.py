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
Test Module for testing question answering dataset for question answering trainer.

How to run this:
windows:
pytest .\\tests\\st\\test_trainer\\test_question_answering_trainer\\test_dataset.py
linux:
pytest ./tests/st/test_trainer/test_question_answering_trainer/test_dataset.py
"""
import os
import json
# import shutil
import pytest

from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.register.config import MindFormerConfig
from mindformers.dataset.build_dataset import build_dataset


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
class TestQuestionAnsweringTrainDataset:
    """A test class for testing TestQuestionAnsweringTrainDataset classes"""
    def setup_method(self):
        """prepare for test"""
        project_path = MindFormerBook.get_project_path()

        config_path = os.path.join(
            project_path, "configs", "qa",
            "run_qa_bert_base_uncased.yaml"
        )

        self.new_dataset_dir = "./test_qa_dataset/"
        self.config = MindFormerConfig(config_path)
        self.config.train_dataset_task.dataset_config.data_loader.dataset_dir = self.new_dataset_dir

        self.make_local_directory()
        self.make_dataset(repeat_num=30)

    def test_dataset(self):
        """
        Feature: TestQuestionAnsweringTrainDataset
        Description: A data set for question answering train dataset
        Expectation: TypeError, ValueError
        """
        self.setup_method()
        data_loader = build_dataset(self.config.train_dataset_task)
        for item in data_loader:
            assert item[0].shape == (12, 384)
            assert item[1].shape == (12, 384)
            assert item[2].shape == (12, 384)
            assert item[3].shape == (12,)
            assert item[4].shape == (12,)
            assert item[5].shape == (12,)

    def make_local_directory(self):
        """make local directory"""
        os.makedirs(self.new_dataset_dir, exist_ok=True)

    # def teardown_method(self):
    #     """delete local directory"""
    #     shutil.rmtree(self.new_dataset_dir)

    def make_dataset(self, repeat_num=30):
        """make a fake SQuAD dataset"""
        test_train_data = {}
        entry = {'title': 'Prime_minister', 'paragraphs': [{'context': 'A prime minister is the' \
            'most senior minister of cabinet in the executive branch of government, often in a parliamentary' \
            'or semi-presidential system. In many systems, the prime minister selects and may dismiss other' \
            'members of the cabinet, and allocates posts to members within the government. In most systems,' \
            'the prime minister is the presiding member and chairman of the cabinet. In a minority of systems,' \
            'notably in semi-presidential systems of government, a prime minister is the official who is appointed' \
            'to manage the civil service and execute the directives of the head of state.', 'qas': [{'answers': \
            [{'answer_start': 63, 'text': 'executive'}], 'question': 'What branch of government does the prime' \
            'minister lead?', 'id': '56dd20d966d3e219004dabf3'}]}]}

        test_train_data["data"] = []
        for _ in range(repeat_num):
            test_train_data["data"].append(entry)

        train_file = os.path.join(self.new_dataset_dir, "train-v1.1.json")
        with open(train_file, 'w', encoding='utf-8') as filer:
            filer.write(json.dumps(test_train_data))
