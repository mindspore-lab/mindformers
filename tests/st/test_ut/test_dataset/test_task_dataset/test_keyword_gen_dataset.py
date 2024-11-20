# Copyright 2024 Huawei Technologies Co., Ltd
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
"""test keyword_gen_dataset"""
import copy
import os
import unittest
import tempfile
import numpy as np
import pytest
import mindspore.dataset as ds
from mindformers import MindFormerConfig
from mindformers.dataset import KeyWordGenDataset, ADGenDataLoader
from tests.st.test_ut.test_dataset.get_test_data import get_adgen_data, get_mindrecord_data
from tests.st.test_ut.test_tokenizers.get_vocab_model import get_sp_vocab_model
from tests.utils.model_tester import create_glm_tokenizer


def build_prompt(query, history=None):
    """build prompt"""
    if history is None:
        history = []
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(
            i + 1, old_query, response)
    prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
    return prompt


def check_dataset_config(dataset_config, params):
    """Check `dataset_config`, If it is empty, use the input parameter to create a new `dataset_config`."""
    if not dataset_config:
        params.pop("dataset_config")
        kwargs = params.pop("kwargs") if params.get("kwargs") else {}
        params.update(kwargs)
        dataset_config = MindFormerConfig(**params)
    return dataset_config


def init_dataset_config(dataset_config):
    """Init dataset config."""
    ds.config.set_seed(dataset_config.seed)
    ds.config.set_prefetch_size(dataset_config.prefetch_size)
    ds.config.set_numa_enable(dataset_config.numa_enable)


class TestKeywordGenDataset(unittest.TestCase):
    """A test class for testing KeyWordGenDataset"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        get_adgen_data(cls.path)
        get_sp_vocab_model("llama", cls.path)
        cls.tokenizer = create_glm_tokenizer()
        cls.tokenizer.build_prompt = build_prompt
        cls.data_loader = ADGenDataLoader(
            dataset_dir=os.path.join(cls.path, "train.json"),
            shuffle=False,
            phase="train",
            origin_columns=['content', 'summary']
        )
        cls.raw_config = {
            "dataset_config": None,
            "data_loader": cls.data_loader,
            "tokenizer": cls.tokenizer,
            "input_columns": ['input_ids', 'labels'],
            "max_source_length": 16,
            "max_target_length": 16,
            "ignore_pad_token_for_loss": True,
            "phase": 'train',
            "batch_size": 1,
            "auto_tune": False,
            "profile": False,
            "seed": 0,
            "prefetch_size": 1,
            "numa_enable": False,
            "filepath_prefix": './autotune',
            "autotune_per_step": 10,
            "repeat": 1,
            "num_parallel_workers": 8,
            "drop_remainder": True
        }
        cls.dataset_config = check_dataset_config(None, copy.deepcopy(cls.raw_config))
        cls.res_train_input_version1 = [5, 65421, 61, 67329, 32, 98339, 61, 72043, 32, 65347, 61, 70872, 32, 69768,
                                        61, 130001, 130004, 5, 87052, 96914, 81471, 64562, 65759, 64493, 64988, 6,
                                        65840, 65388, 74531, 63825, 75786, 130005, 3]
        cls.res_train_label_version1 = [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                                        -100, -100, -100, 5, 87052, 96914, 81471, 64562, 65759, 64493, 64988, 6, 65840,
                                        65388, 74531, 63825, 75786, 130005, -100, -100]
        cls.res_train_input_version2 = [53, 6945, 5, 9, 42, 4, 4, 64286, 12, 65421, 61, 67329, 32, 98339, 130001,
                                        130004, 5, 87052, 96914, 81471, 64562, 65759, 64493, 64988, 6, 65840, 65388,
                                        74531, 63825, 75786, 64009, 63823, 130005]
        cls.res_train_label_version2 = [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                                        -100, -100, 5, 87052, 96914, 81471, 64562, 65759, 64493, 64988, 6, 65840, 65388,
                                        74531, 63825, 75786, 64009, 63823, 130005, -100]
        cls.res_train_input_version3 = [5, 65421, 61, 67329, 32, 98339, 61, 72043, 32, 65347, 61, 70872, 32, 69768,
                                        130001, 130004, 5, 87052, 96914, 81471, 64562, 65759, 64493, 64988, 6, 65840,
                                        65388, 74531, 63825, 75786, 64009, 63823, 130005]
        cls.res_train_label_version3 = [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                                        -100, -100, 5, 87052, 96914, 81471, 64562, 65759, 64493, 64988, 6, 65840, 65388,
                                        74531, 63825, 75786, 64009, 63823, 130005, -100]
        cls.res_eval_input_version1 = [5, 65421, 61, 67329, 32, 98339, 61, 72043, 32, 65347, 61, 130001, 130004,
                                       3, 3, 3]
        cls.res_eval_label_version1 = [5, 87052, 96914, 81471, 64562, 65759, 64493, 63848, 130001, 130004, 3, 3, 3,
                                       3, 3, 3]
        cls.res_eval_input_version2 = [53, 6945, 5, 9, 42, 4, 4, 64286, 12, 65421, 130001, 130004, 3, 3, 3, 3]
        cls.res_eval_label_version2 = [5, 87052, 96914, 81471, 64562, 65759, 64493, 64988, 130001, 130004, 3, 3, 3,
                                       3, 3, 3]
        cls.res_eval_input_version3 = [53, 6945, 5, 9, 42, 4, 4, 64286, 12, 65421, 130001, 130004, 3, 3, 3, 3]
        cls.res_eval_label_version3 = [5, 87052, 96914, 81471, 64562, 65759, 64493, 64988, 130001, 130004, 3, 3, 3,
                                       3, 3, 3]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        data_loader = ADGenDataLoader(
            dataset_dir=os.path.join(self.path, "train.json"),
            shuffle=False,
            phase="train",
            origin_columns=['content', 'summary']
        )
        dataset = KeyWordGenDataset(
            data_loader=data_loader,
            tokenizer=self.tokenizer,
            input_columns=['input_ids', 'labels'],
            max_source_length=16,
            max_target_length=16,
            ignore_pad_token_for_loss=True,
            phase='train',
            batch_size=1
        )
        for item in dataset:
            assert len(item) == 4
            assert all(item[0].asnumpy()[0] == self.res_train_input_version1)
            assert all(item[1].asnumpy()[0] == self.res_train_label_version1)
            assert item[2].shape == (1, 2, 33)
            assert item[3].shape == (1, 1, 33, 33)
            break

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_train_version(self):
        """test train & version logic"""
        data_loader = ADGenDataLoader(
            dataset_dir=os.path.join(self.path, "train.json"),
            shuffle=False,
            phase="train",
            origin_columns=['content', 'summary']
        )
        dataset = KeyWordGenDataset(
            data_loader=data_loader,
            tokenizer=self.tokenizer,
            input_columns=['input_ids', 'labels'],
            max_source_length=16,
            max_target_length=16,
            ignore_pad_token_for_loss=True,
            phase='train',
            batch_size=1,
            version=2
        )
        for item in dataset:
            assert len(item) == 2
            assert all(item[0].asnumpy()[0] == self.res_train_input_version2)
            assert all(item[1].asnumpy()[0] == self.res_train_label_version2)
            break
        data_loader = ADGenDataLoader(
            dataset_dir=os.path.join(self.path, "train.json"),
            shuffle=False,
            phase="train",
            origin_columns=['content', 'summary']
        )
        dataset = KeyWordGenDataset(
            data_loader=data_loader,
            tokenizer=self.tokenizer,
            input_columns=['input_ids', 'labels'],
            max_source_length=16,
            max_target_length=16,
            ignore_pad_token_for_loss=True,
            phase='train',
            batch_size=1,
            version=3
        )
        for item in dataset:
            assert len(item) == 2
            assert all(item[0].asnumpy()[0] == self.res_train_input_version3)
            assert all(item[1].asnumpy()[0] == self.res_train_label_version3)
            break

    # pylint: disable=W0212
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_train_function(self):
        """test train & function logic"""
        res = KeyWordGenDataset._train_dataset_function(
            prompt=np.array("类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤"),
            answer=np.array("宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。"),
            dataset_config=self.dataset_config, tokenizer=self.tokenizer
        )
        assert len(res) == 4
        assert res[0] == self.res_train_input_version1
        assert res[1] == self.res_train_label_version1
        assert res[2].shape == (2, 33)
        assert res[3].shape == (1, 33, 33)
        res = KeyWordGenDataset._train_dataset_functionv2(
            prompt=np.array("类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤"),
            answer=np.array("宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。"),
            dataset_config=self.dataset_config, tokenizer=self.tokenizer
        )
        assert len(res) == 2
        assert res[0] == self.res_train_input_version2
        assert res[1] == self.res_train_label_version2
        res = KeyWordGenDataset._train_dataset_functionv3(
            prompt=np.array("类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤"),
            answer=np.array("宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。"),
            dataset_config=self.dataset_config, tokenizer=self.tokenizer
        )
        assert len(res) == 2
        assert res[0] == self.res_train_input_version3
        assert res[1] == self.res_train_label_version3

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_eval(self):
        """test eval logic"""
        data_loader = ADGenDataLoader(
            dataset_dir=os.path.join(self.path, "train.json"),
            shuffle=False,
            phase="train",
            origin_columns=['content', 'summary']
        )
        dataset = KeyWordGenDataset(
            data_loader=data_loader,
            tokenizer=self.tokenizer,
            input_columns=['input_ids', 'labels'],
            max_source_length=16,
            max_target_length=16,
            ignore_pad_token_for_loss=True,
            phase='eval',
            batch_size=1
        )
        for item in dataset:
            assert len(item) == 2
            assert all(item[0].asnumpy()[0] == self.res_eval_input_version1)
            assert all(item[1].asnumpy()[0] == self.res_eval_label_version1)
            break

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_eval_version(self):
        """test eval & version logic"""
        data_loader = ADGenDataLoader(
            dataset_dir=os.path.join(self.path, "train.json"),
            shuffle=False,
            phase="train",
            origin_columns=['content', 'summary']
        )
        dataset = KeyWordGenDataset(
            data_loader=data_loader,
            tokenizer=self.tokenizer,
            input_columns=['input_ids', 'labels'],
            max_source_length=16,
            max_target_length=16,
            ignore_pad_token_for_loss=True,
            phase='eval',
            batch_size=1,
            version=2
        )
        for item in dataset:
            assert len(item) == 2
            assert all(item[0].asnumpy()[0] == self.res_eval_input_version2)
            assert all(item[1].asnumpy()[0] == self.res_eval_label_version2)
            break

    # pylint: disable=W0212
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_eval_function(self):
        """test eval & function logic"""
        res = KeyWordGenDataset._eval_dataset_function(
            prompt=np.array("类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤"),
            answer=np.array("宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。"),
            dataset_config=self.dataset_config, tokenizer=self.tokenizer
        )
        assert len(res) == 2
        assert res[0] == self.res_eval_input_version1
        assert res[1] == self.res_eval_label_version1

        res = KeyWordGenDataset._eval_dataset_functionv2(
            prompt=np.array("类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤"),
            answer=np.array("宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。"),
            dataset_config=self.dataset_config, tokenizer=self.tokenizer
        )
        assert len(res) == 2
        assert res[0] == self.res_eval_input_version2
        assert res[1] == self.res_eval_label_version2

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_process_mindrecord_data(self):
        """test process_mindrecord_data logic"""
        get_mindrecord_data(self.path)
        raw_config = copy.deepcopy(self.raw_config)
        raw_config["data_loader"] = {"dataset_dir": os.path.join(self.path, "test.mindrecord"), "type": "MindDataset"}
        dataset_config = check_dataset_config(None, raw_config)
        dataset = KeyWordGenDataset(dataset_config)
        for item in dataset:
            assert len(item) == 2
            assert all(item[0].asnumpy()[0] == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24])
            assert all(item[1].asnumpy()[0] == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24])
            break
        raw_config = copy.deepcopy(self.raw_config)
        raw_config["data_loader"] = {"dataset_dir": self.path, "type": "MindDataset"}
        dataset_config = check_dataset_config(None, raw_config)
        dataset = KeyWordGenDataset(dataset_config)
        for item in dataset:
            assert len(item) == 2
            assert all(item[0].asnumpy()[0] == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24])
            assert all(item[1].asnumpy()[0] == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24])
            break
        raw_config = copy.deepcopy(self.raw_config)
        raw_config["data_loader"] = {"dataset_files": os.path.join(self.path, "test.mindrecord"), "type": "MindDataset"}
        dataset_config = check_dataset_config(None, raw_config)
        dataset = KeyWordGenDataset(dataset_config)
        for item in dataset:
            assert len(item) == 2
            assert all(item[0].asnumpy()[0] == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24])
            assert all(item[1].asnumpy()[0] == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24])
            break
        raw_config = copy.deepcopy(self.raw_config)
        raw_config["data_loader"] = {"dataset_files": [os.path.join(self.path, "test.mindrecord")],
                                     "type": "MindDataset"}
        dataset_config = check_dataset_config(None, raw_config)
        dataset = KeyWordGenDataset(dataset_config)
        for item in dataset:
            assert len(item) == 2
            assert all(item[0].asnumpy()[0] == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24])
            assert all(item[1].asnumpy()[0] == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24])
            break
        with pytest.raises(ValueError):
            raw_config = copy.deepcopy(self.raw_config)
            raw_config["data_loader"] = {"dataset_files": "",
                                         "type": "MindDataset"}
            dataset_config = check_dataset_config(None, raw_config)
            assert KeyWordGenDataset(dataset_config)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_process_raw_text_data(self):
        """test process_raw_text_data logic"""
        raw_config = copy.deepcopy(self.raw_config)
        raw_config["data_loader"] = {
            "dataset_dir": os.path.join(self.path, "train.json"),
            "type": "ADGenDataLoader",
            "phase": "train",
            "shuffle": False,
            "origin_columns": ['content', 'summary']
        }
        dataset_config = check_dataset_config(None, raw_config)
        dataset = KeyWordGenDataset(dataset_config)
        for item in dataset:
            assert len(item) == 4
            assert all(item[0].asnumpy()[0] == self.res_train_input_version1)
            assert all(item[1].asnumpy()[0] == self.res_train_label_version1)
            assert item[2].shape == (1, 2, 33)
            assert item[3].shape == (1, 1, 33, 33)
            break
