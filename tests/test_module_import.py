# Copyright 2022 Huawei Technologies Co., Ltd
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
Test module for testing the interface used for transformer.
How to run this:
pytest tests/test_module_import.py
"""


def test_imports_gpt():
    """
    Feature: The GPT import
    Description: Test to import the gpt model.
    Expectation: No import error
    """
    from transformer.models import gpt
    gpt.GPTModel(gpt.GPTConfig(num_layers=1, hidden_size=8, num_heads=1))


def test_imports_bert():
    """
    Feature: The BERT import
    Description: Test to import the gpt model.
    Expectation: No import error
    """
    from transformer.models import bert
    bert.BertModel(bert.BertConfig(num_layers=1, embedding_size=8, num_heads=1), is_training=False)


def test_imports_trainer():
    """
    Feature: The Trainer import
    Description: Test to import the trainer.
    Expectation: No import error
    """
    from transformer.trainer import Trainer, TrainingConfig
    Trainer(TrainingConfig(recompute=False, auto_model="bert", device_target="CPU"))
