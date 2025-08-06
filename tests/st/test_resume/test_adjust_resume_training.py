#  Copyright 2025 Huawei Technologies Co., Ltd
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ============================================================================
"""test adjust resume training"""
import pytest
from mindspore.dataset import GeneratorDataset
from mindformers.models.llama.llama import LlamaForCausalLM
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers import Trainer, TrainingArguments
from tests.st.test_model.test_mixtral_model.test_trainer import generator_train


class DummyTrainer(Trainer):
    def __init__(self, resume_training=True, load_checkpoint=""):
        args = TrainingArguments()
        train_dataset = GeneratorDataset(generator_train, column_names=["input_ids"]).batch(batch_size=4)
        model_config = LlamaConfig(num_layers=1, hidden_size=1, num_heads=1, seq_length=1, vocab_size=1)
        model = LlamaForCausalLM(model_config)
        super().__init__(task='text_generation', model=model, args=args, train_dataset=train_dataset)

        self.config.resume_training = resume_training
        self.config.load_checkpoint = load_checkpoint


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_empty_string_checkpoint():
    """
    Feature: Resume training with empty checkpoint path
    Description: Set resume_training=True and load_checkpoint="" (empty string)
    Expectation: resume_training is set to False and load_checkpoint remains ""
    """
    trainer = DummyTrainer(resume_training=True, load_checkpoint="")
    # pylint: disable=W0212
    trainer._adjust_resume_training_if_ckpt_path_invalid()
    assert trainer.config.resume_training is False
    assert trainer.config.load_checkpoint == ""


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_empty_directory_checkpoint(tmp_path):
    """
    Feature: Resume training with empty directory as checkpoint
    Description: Set resume_training=True and load_checkpoint to an empty directory path
    Expectation: resume_training is set to False and load_checkpoint is reset to ""
    """
    trainer = DummyTrainer(resume_training=True, load_checkpoint=str(tmp_path))
    # pylint: disable=W0212
    trainer._adjust_resume_training_if_ckpt_path_invalid()
    assert trainer.config.resume_training is False
    assert trainer.config.load_checkpoint == ""


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_valid_checkpoint(tmp_path):
    """
    Feature: Resume training with valid checkpoint
    Description: Set resume_training=True and load_checkpoint to a directory with a file
    Expectation: resume_training remains True and load_checkpoint path is preserved
    """
    file_path = tmp_path / "checkpoint.safetensors"
    file_path.write_text("dummy checkpoint")
    trainer = DummyTrainer(resume_training=True, load_checkpoint=str(tmp_path))
    # pylint: disable=W0212
    trainer._adjust_resume_training_if_ckpt_path_invalid()
    assert trainer.config.resume_training is True
    assert trainer.config.load_checkpoint == str(tmp_path)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_resume_training_false():
    """
    Feature: Skip resume training when flag is False
    Description: Set resume_training=False regardless of checkpoint path
    Expectation: No change, resume_training remains False and load_checkpoint remains ""
    """
    trainer = DummyTrainer(resume_training=False, load_checkpoint="")
    # pylint: disable=W0212
    trainer._adjust_resume_training_if_ckpt_path_invalid()
    assert trainer.config.resume_training is False
    assert trainer.config.load_checkpoint == ""
