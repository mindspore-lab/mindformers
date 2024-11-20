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
"""test squad dataloader"""
import os
import unittest
import tempfile
import pytest
from mindformers import LlamaTokenizer
from mindformers.dataset.dataloader.squad_dataloader import SQuADDataLoader, SQuADDataset, _improve_answer_span
from tests.st.test_ut.test_dataset.get_test_data import get_squad_data
from tests.st.test_ut.test_tokenizers.get_vocab_model import get_sp_vocab_model


temp_dir = tempfile.TemporaryDirectory()
path = temp_dir.name
get_sp_vocab_model("llama", path)
tokenizer_model_path = os.path.join(path, "llama_tokenizer.model")
tokenizer = LlamaTokenizer(vocab_file=tokenizer_model_path)


class TestSQuADDataloader(unittest.TestCase):
    """ A test class for testing SQuAD dataloader"""

    @classmethod
    def setUpClass(cls):
        cls.path = path
        get_squad_data(cls.path)
        cls.tokenizer = tokenizer

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        dataloader = SQuADDataLoader(self.path, tokenizer=self.tokenizer, max_seq_len=15)
        columns = dataloader.column_names
        assert set(columns) == {
            'input_ids', 'input_mask', 'token_type_id', 'start_positions', 'end_positions', 'unique_id'
        }
        dataloader = dataloader.batch(1)
        for item in dataloader:
            assert len(item) == 6
            assert all(item[0][0].asnumpy() == [0, 83, 88, 0, 48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 0])
            assert all(item[1][0].asnumpy() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            assert all(item[2][0].asnumpy() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            assert all(item[3].asnumpy() == [8])
            assert all(item[4].asnumpy() == [8])
            break

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_not_dir(self):
        """test not dir input logic"""
        mock_dir = "not_a_dir"
        with pytest.raises(ValueError):
            assert SQuADDataLoader(mock_dir, tokenizer=self.tokenizer)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_stage(self):
        """test stage logic"""
        dataloader = SQuADDataLoader(self.path, tokenizer=self.tokenizer, max_seq_len=15, stage="dev")
        columns = dataloader.column_names
        assert set(columns) == {
            'input_ids', 'input_mask', 'token_type_id', 'start_positions', 'end_positions', 'unique_id'
        }
        dataloader = dataloader.batch(1)
        for item in dataloader:
            assert len(item) == 6
            assert all(item[0][0].asnumpy() == [0, 83, 88, 0, 48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 0])
            assert all(item[1][0].asnumpy() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            assert all(item[2][0].asnumpy() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            assert all(item[3].asnumpy() == [-1])
            assert all(item[4].asnumpy() == [-1])
            break
        mock_stage = "mock_stage"
        with pytest.raises(ValueError):
            assert SQuADDataLoader(self.path, tokenizer=self.tokenizer, stage=mock_stage)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_column_names(self):
        """test column names logic"""
        with pytest.raises(TypeError):
            column_names = 1
            assert SQuADDataLoader(self.path, tokenizer=self.tokenizer, column_names=column_names)
        with pytest.raises(Exception):
            column_names = [1]
            assert SQuADDataLoader(self.path, tokenizer=self.tokenizer, column_names=column_names)
        with pytest.raises(Exception):
            column_names = [1, 2, 3, 4, 5, 6]
            assert SQuADDataLoader(self.path, tokenizer=self.tokenizer, column_names=column_names)


class TestSQuADDataSet(unittest.TestCase):
    """ A test class for testing SQuAD dataset"""

    @classmethod
    def setUpClass(cls):
        cls.path = path
        get_squad_data(cls.path)
        cls.tokenizer = tokenizer

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        dataset = SQuADDataset(self.path, tokenizer=self.tokenizer, max_seq_len=15)
        assert hasattr(dataset, "examples")
        assert dataset.examples[0].answer_text == 'one'
        assert dataset.examples[0].context_text == 'An increasing sequence: one, two, three.'
        assert dataset.examples[0].doc_tokens == ['An', 'increasing', 'sequence:', 'one,', 'two,', 'three.']
        assert dataset.examples[0].end_position == 3
        assert dataset.examples[0].start_position == 3
        assert dataset.examples[0].is_impossible is False
        assert dataset.examples[0].question_text == '华为是一家总部位于中国深圳的多元化科技公司'

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_pad(self):
        """test pad logic"""
        dataset = SQuADDataset(self.path, tokenizer=self.tokenizer, max_seq_len=20)
        assert hasattr(dataset, "max_seq_len")
        assert dataset.max_seq_len == 20

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_stage(self):
        """test stage logic"""
        with pytest.raises(ValueError):
            stage = "mock_stage"
            assert SQuADDataset(self.path, tokenizer=self.tokenizer, stage=stage)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_dataset_dir(self):
        """test not dir input logic"""
        with pytest.raises(ValueError):
            mock_dir = "not_a_dir"
            assert SQuADDataset(dataset_dir=mock_dir, tokenizer=self.tokenizer)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_getitem(self):
        """test __getitem__ logic"""
        dataset = SQuADDataset(self.path, tokenizer=self.tokenizer, max_seq_len=15)
        item = dataset[0]
        assert len(item) == 6
        assert item[0] == [0, 83, 88, 0, 48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 0]
        assert item[1] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        assert item[2] == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        assert item[3] == 8
        assert item[4] == 8

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_len(self):
        """test __len__ logic"""
        dataset = SQuADDataset(self.path, tokenizer=self.tokenizer, max_seq_len=15)
        assert len(dataset) == 1


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_improve_answer_span():
    """
    Feature: squad_dataloader._improve_answer_span
    Description: test _improve_answer_span function
    Expectation: success
    """
    res = _improve_answer_span(tokenizer=tokenizer, doc_tokens="", input_start=2, input_end=1, orig_answer_text="")
    assert res == (2, 1)
