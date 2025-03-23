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
"""test training dataloader"""
import os
import tempfile
import unittest
from unittest.mock import patch
import pytest
from mindformers.dataset.dataloader.training_dataloader import TrainingDataset, TrainingDataLoader, run_cmd
from mindformers import LlamaTokenizer
from mindformers.dataset.dataloader.datareaders import wikitext_reader
from tests.st.test_ut.test_dataset.get_test_data import get_wikitext_data, get_json_data
from tests.st.test_ut.test_tokenizers.get_vocab_model import get_sp_vocab_model


temp_dir = tempfile.TemporaryDirectory()
path = temp_dir.name
get_sp_vocab_model("llama", path)
tokenizer_model_path = os.path.join(path, "llama_tokenizer.model")
tokenizer = LlamaTokenizer(vocab_file=tokenizer_model_path)


class TestTrainingDataloader(unittest.TestCase):
    """ A test class for testing Training dataloader"""

    @classmethod
    def setUpClass(cls):
        get_wikitext_data(path)
        cls.path = os.path.join(path, "wiki.train.tokens")
        cls.tokenizer = tokenizer

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        dataloader = TrainingDataLoader(
            self.path,
            column_names=["input_ids", "attention_mask"],
            dataset_name="wikitext",
            file_format="tokens",
            tokenizer=self.tokenizer,
            max_length=8,
            shuffle=False
        )
        dataloader = dataloader.batch(1)
        for item in dataloader:
            assert len(item) == 2
            assert all(item[0][0].asnumpy() == [1, 83, 88, 135, 57, 58, 187, 135])
            assert all(item[1][0].asnumpy() == [1, 1, 1, 1, 1, 1, 1, 1])
            break

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_max_length(self):
        """test max length logic"""
        with pytest.raises(TypeError):
            assert TrainingDataLoader(
                self.path,
                column_names=["input_ids", "attention_mask"],
                dataset_name="wikitext",
                file_format="tokens",
                tokenizer=self.tokenizer,
                max_length=0,
                shuffle=False
            )

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_other(self):
        """test other logic"""
        dataloader = TrainingDataLoader(
            path,
            column_names=["input_ids", "attention_mask"],
            dataset_name="wikitext",
            file_format="tokens",
            tokenizer=self.tokenizer,
            max_length=8,
            shuffle=False,
            read_function=wikitext_reader,
        )
        dataloader = dataloader.batch(1)
        for item in dataloader:
            assert len(item) == 2
            assert all(item[0][0].asnumpy() == [1, 83, 88, 135, 57, 58, 187, 135])
            assert all(item[1][0].asnumpy() == [1, 1, 1, 1, 1, 1, 1, 1])
            break


class TestTrainingDataSet(unittest.TestCase):
    """ A test class for testing Training dataset"""

    @classmethod
    def setUpClass(cls):
        cls.path = path
        get_json_data(cls.path)
        cls.tokenizer = tokenizer

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        dataset = TrainingDataset(
            self.path,
            column_names=["input_ids", "attention_mask"],
            file_format="json",
            tokenizer=self.tokenizer,
            max_length=8,
            text_col="input",
            shuffle=False)
        assert hasattr(dataset, "current_samples")
        assert all(dataset.current_samples[0] == [83, 88, 135, 57, 58, 187, 135, 89])
        assert dataset.column_names == ['input_ids', 'attention_mask']
        assert dataset.text_col == "input"

    # pylint: disable=W0212
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_other(self):
        """test other logic"""
        dataset = TrainingDataset(
            os.path.join(self.path, "train.json"),
            column_names=["input_ids", "attention_mask"],
            file_format="json",
            tokenizer=self.tokenizer,
            max_length=8,
            text_col="input",
            shuffle=False
        )
        res = next(dataset)
        assert res == ([1, 83, 88, 135, 57, 58, 187, 135], [1, 1, 1, 1, 1, 1, 1, 1])
        res = dataset._check_format(dataset_dir=dataset.dataset_dir, file_format=dataset.format)
        assert res == "json"
        res = dataset._check_format(dataset_dir=dataset.dataset_dir, file_format="")
        assert res == "json"
        with pytest.raises(ValueError):
            assert dataset._check_format(dataset_dir=self.path, file_format="")
        res = dataset._tokenizer_func("An increasing sequence: one, two, three。")
        assert res == [48, 87, 85, 157, 65, 135, 67, 135, 80, 167]
        res = dataset._get_all_samples_number()
        assert res == 1
        dataset._reset_iter_index()
        assert dataset.iter_index == 0
        assert all(dataset.current_samples[0] == [83, 88, 135, 57, 58, 187, 135, 89])
        res = dataset._check_tokenizer({
            "type": "LlamaTokenizer", "vocab_file": os.path.join(self.path, "llama_tokenizer.model")
        })
        assert isinstance(res, LlamaTokenizer)


class MockRun:
    """mock run"""
    def __init__(self):
        self.returncode = 0
        self.stderr = "stderr"
        self.stdout = "stdout"


@patch("subprocess.run")
@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_run_cmd(mock_get):
    """
    Feature: training_dataloader.run_cmd
    Description: test run_cmd function
    Expectation: success
    """
    mockrun = MockRun()
    mock_get.return_value = mockrun
    assert run_cmd("mock input", pipeline=True) == (True, "stdout")
    mockrun.returncode = 1
    mock_get.return_value = mockrun
    assert run_cmd("mock input") == (False, "stderr")
    mockrun.stderr = "No such file"
    assert run_cmd("mock input") == (False, "No such file")
    mockrun.stderr = "Files exists"
    assert run_cmd("mock input") == (False, "Files exists")
    mockrun.stderr = "permission denied"
    assert run_cmd("mock input") == (False, "permission denied")
