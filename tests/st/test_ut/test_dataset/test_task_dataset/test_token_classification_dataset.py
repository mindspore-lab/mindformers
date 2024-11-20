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
"""test token_classification_dataset"""
import copy
import os
import unittest
import tempfile
import pytest
from mindformers import BloomTokenizer, MindFormerConfig
from mindformers.dataset import TokenClassificationDataset, CLUENERDataLoader, TokenizeWithLabel, LabelPadding
from tests.st.test_ut.test_dataset.get_test_data import get_cluener_data
from tests.st.test_ut.test_tokenizers.get_vocab_model import get_bbpe_vocab_model


def check_dataset_config(dataset_config, params):
    """Check `dataset_config`, If it is empty, use the input parameter to create a new `dataset_config`."""
    if not dataset_config:
        params.pop("dataset_config")
        kwargs = params.pop("kwargs") if params.get("kwargs") else {}
        params.update(kwargs)
        dataset_config = MindFormerConfig(**params)
    return dataset_config


class TestTokenClassificationDataset(unittest.TestCase):
    """A test class for testing TokenClassificationDataset"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        get_cluener_data(cls.path)
        get_bbpe_vocab_model("bloom", cls.path)
        cls.tokenizer_model_path = os.path.join(cls.path, "bloom_tokenizer.json")
        cls.raw_config = {
            "dataset_config": None,
            "data_loader": {
                "dataset_dir": cls.path,
                "type": "CLUENERDataLoader",
                "shuffle": False,
                "stage": "train"
            },
            "text_transforms": {
                "type": "TokenizeWithLabel",
                "max_length": 16,
                "padding": "max_length"
            },
            "label_transforms": {
                "type": "LabelPadding",
                "max_length": 16,
                "padding_value": 0
            },
            "input_columns": ["text", "label_id"],
            "output_columns": ["input_ids", "token_type_ids", "attention_mask", "label_id"],
            "column_order": ["input_ids", "token_type_ids", "attention_mask", "label_id"],
            "python_multiprocessing": False,
            "batch_size": 1,
            "auto_tune": False,
            "profile": False,
            "seed": 0,
            "prefetch_size": 1,
            "numa_enable": False,
            "filepath_prefix": './autotune',
            "autotune_per_step": 10,
            "repeat": 1,
            "num_parallel_workers": 1,
            "drop_remainder": True,
            "tokenizer": {
                "vocab_file": cls.tokenizer_model_path,
                "type": "BloomTokenizer"
            }
        }
        cls.dataset_config = check_dataset_config(None, copy.deepcopy(cls.raw_config))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        tokenizer = BloomTokenizer(vocab_file=self.tokenizer_model_path)

        data_loader = CLUENERDataLoader(dataset_dir=self.path, stage='train', column_names=['text', 'label_id'])
        text_transforms = TokenizeWithLabel(max_length=16, padding='max_length', tokenizer=tokenizer)
        label_transforms = LabelPadding(max_length=16, padding_value=0)
        dataset = TokenClassificationDataset(data_loader=data_loader, text_transforms=text_transforms,
                                             label_transforms=label_transforms, tokenizer=tokenizer,
                                             input_columns=['text', 'label_id'],
                                             output_columns=['input_ids', 'token_type_ids', 'attention_mask',
                                                             'label_id'],
                                             batch_size=1
                                             )
        for item in dataset:
            assert len(item) == 4
            assert item[3].asnumpy()[0].tolist() == [0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 7, 17, 17, 0, 0, 0]
            break

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_other(self):
        """test other logic"""
        dataset = TokenClassificationDataset(dataset_config=self.dataset_config)
        for item in dataset:
            assert len(item) == 4
            assert item[3].asnumpy()[0].tolist() == [0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 7, 17, 17, 0, 0, 0]
            break
