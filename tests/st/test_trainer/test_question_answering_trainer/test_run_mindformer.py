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
Test Module for testing QuestionAnsweringDataset

How to run this:
windows:
pytest .\\tests\\st\\test_trainer\\test_question_answering_trainer\\test_run_mindformer.py
linux:
pytest ./tests/st/test_trainer/test_question_answering_trainer/test_run_mindformer.py
"""
import os
import json
# import shutil
# import pytest

from mindformers.mindformer_book import MindFormerBook


# @pytest.mark.level0
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.env_onecard
class TestRunMindFormer:
    """A test class for testing run mindformer"""
    def setup_method(self):
        """prepare for test"""

        self.new_dataset_dir = "./squad/"

        self.make_local_directory()
        self.make_dataset(repeat_num=20)

    def test_trainer_train_method(self):
        """
        Feature: QuestionAnsweringDataset by run_mindformer.py
        Description: use QuestionAnsweringDataset with run_mindformer.py
        Expectation: TypeError, ValueError
        """
        yaml_path = os.path.join(MindFormerBook.get_project_path(),
                                 "configs", "qa", "run_qa_bert_base_uncased.yaml")
        command = "python run_mindformer.py --config " + yaml_path + " --run_mode finetune"
        os.system(command)

    def test_trainer_eval_method(self):
        """
        Feature: QuestionAnsweringDataset by run_mindformer.py
        Description: use QuestionAnsweringDataset with run_mindformer.py
        Expectation: TypeError, ValueError
        """
        yaml_path = os.path.join(MindFormerBook.get_project_path(),
                                 "configs", "qa", "run_qa_bert_base_uncased.yaml")
        command = "python run_mindformer.py --config " + yaml_path + \
                  " --run_mode eval --load_checkpoint qa_bert_base_uncased_squad"
        os.system(command)

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

        dev_file = os.path.join(self.new_dataset_dir, "dev-v1.1.json")
        with open(dev_file, 'w', encoding='utf-8') as filer:
            filer.write(json.dumps(test_data))
