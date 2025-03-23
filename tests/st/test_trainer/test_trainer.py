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
"""
Test module for testing the trainer interface used for mindformers.
How to run this:
    pytest tests/st/test_trainer/test_trainer.py
"""
import os
import copy
import numpy as np
import pytest

import mindspore as ms
from mindspore.dataset import GeneratorDataset

from mindformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from mindformers import Trainer, TrainingArguments
from mindformers.core.callback import MFLossMonitor

ms.set_context(mode=0)

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

EPOCHS = 1
NUM_LAYERS = 1
HIDDEN_SIZE = 16
NUM_HEADS = 2
SEQ_LENGTH = 32
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
EVAL_STEPS = 4
DATA_SIZE = 8


def generator_train():
    """train dataset generator"""
    seq_len = SEQ_LENGTH + 1
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    for _ in range(DATA_SIZE):
        yield input_ids


def generator_eval():
    """eval dataset generator"""
    seq_len = SEQ_LENGTH
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    for _ in range(DATA_SIZE):
        yield input_ids


class IterableTrain:
    """train iterable dataset"""

    def __init__(self):
        self._index = 0
        seq_len = SEQ_LENGTH + 1
        self.input_ids = np.random.randint(low=0, high=15, size=(DATA_SIZE, seq_len)).astype(np.int32)

    def __next__(self):
        if self._index >= len(self.input_ids):
            raise StopIteration
        item = self.input_ids[self._index]
        self._index += 1
        return item

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self.input_ids)


class IterableEval:
    """eval iterable dataset"""

    def __init__(self):
        self._index = 0
        seq_len = SEQ_LENGTH
        self.input_ids = np.random.randint(low=0, high=15, size=(DATA_SIZE, seq_len)).astype(np.int32)

    def __next__(self):
        if self._index >= len(self.input_ids):
            raise StopIteration
        item = self.input_ids[self._index]
        self._index += 1
        return item

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self.input_ids)


class AccessibleTrain:
    """train accessible dataset"""

    def __init__(self):
        seq_len = SEQ_LENGTH + 1
        self.input_ids = np.random.randint(low=0, high=15, size=(16, seq_len)).astype(np.int32)

    def __getitem__(self, index):
        return self.input_ids[index]

    def __len__(self):
        return len(self.input_ids)


class AccessibleEval:
    """eval accessible dataset"""

    def __init__(self):
        seq_len = SEQ_LENGTH
        self.input_ids = np.random.randint(low=0, high=15, size=(16, seq_len)).astype(np.int32)

    def __getitem__(self, index):
        return self.input_ids[index]

    def __len__(self):
        return len(self.input_ids)


MODEL_CONFIG = LlamaConfig(num_layers=NUM_LAYERS, hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS, seq_length=SEQ_LENGTH)
MODEL = LlamaForCausalLM(MODEL_CONFIG)

PREDICT_MODEL_CONFIG = LlamaConfig(num_layers=NUM_LAYERS, hidden_size=HIDDEN_SIZE,
                                   num_heads=NUM_HEADS, seq_length=SEQ_LENGTH, use_past=True)
PREDICT_MODEL = LlamaForCausalLM(MODEL_CONFIG)

TOKENIZER = LlamaTokenizer(vocab_file=f"{root_path}/utils/llama2_tokenizer/tokenizer.model")

MINDFORMER_CONFIG = Trainer.get_task_config(task="text_generation", model_name="llama2_7b")
MINDFORMER_CONFIG.model.model_config.num_layers = NUM_LAYERS
MINDFORMER_CONFIG.model.model_config.hidden_size = HIDDEN_SIZE
MINDFORMER_CONFIG.model.model_config.num_heads = NUM_HEADS
MINDFORMER_CONFIG.model.model_config.seq_length = SEQ_LENGTH
MINDFORMER_CONFIG.model.model_config.checkpoint_name_or_path = ""
MINDFORMER_CONFIG.eval_step_interval = EVAL_STEPS

PREDICT_MINDFORMER_CONFIG = copy.deepcopy(MINDFORMER_CONFIG)
PREDICT_MINDFORMER_CONFIG.model.model_config.use_past = True

TRAINING_ARGUMENTS = TrainingArguments(do_eval=True,
                                       metric_type="PerplexityMetric",
                                       eval_steps=EVAL_STEPS,
                                       per_device_train_batch_size=TRAIN_BATCH_SIZE,
                                       per_device_eval_batch_size=EVAL_BATCH_SIZE,
                                       train_dataset_in_columns=["input_ids"],
                                       eval_dataset_in_columns=["input_ids"],
                                       num_train_epochs=EPOCHS)

TRAIN_DATASET = GeneratorDataset(generator_train, column_names=["input_ids"])
TRAIN_DATASET_FOR_TRAINER_WITH_ARGS = TRAIN_DATASET.batch(batch_size=TRAIN_BATCH_SIZE)
TRAIN_DATASET_FOR_TRAINER_WITHOUT_ARGS = copy.deepcopy(TRAIN_DATASET_FOR_TRAINER_WITH_ARGS)
EVAL_DATASET = GeneratorDataset(generator_eval, column_names=["input_ids"])
EVAL_DATASET_FOR_TRAINER_WITH_ARGS = EVAL_DATASET.batch(batch_size=EVAL_BATCH_SIZE)
EVAL_DATASET_FOR_TRAINER_WITHOUT_ARGS = copy.deepcopy(EVAL_DATASET_FOR_TRAINER_WITH_ARGS)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
class TestTrainerInit:
    """A test class for testing Trainer module."""

    def test_init_missing_arguments(self):
        """
        Feature: Trainer.__init__()
        Description: Test trainer init missing arguments.
        Expectation: ValueError
        """
        test_cases = [
            (None, "general", None, None, "Neither `task`, `model`, `model_name`, nor `args` are configured."),
            (None, "general", "llama2_7b", None, "The `task` is needed"),
            (None, "general", None, "llama2_7b", "The `task` is needed"),
            (None, "general", "llama2_7b", "llama2_7b", "The `task` is needed"),
            (None, "general", MODEL, None, "The `args` is needed"),
            (None, "general", MODEL, "llama2_7b", "The `task` is needed"),
            (None, "text_generation", None, None, "A model name is needed"),
            (TRAINING_ARGUMENTS, "general", None, None, "A model instance is needed"),
            (TRAINING_ARGUMENTS, "general", "llama2_7b", None, "The `task` is needed"),
            (TRAINING_ARGUMENTS, "general", None, "llama2_7b", "The `task` is needed"),
            (TRAINING_ARGUMENTS, "general", "llama2_7b", "llama2_7b", "The `task` is needed"),
            (TRAINING_ARGUMENTS, "text_generation", None, None, "A model name is needed")]
        for test_case in test_cases:
            args, task, model, model_name, expected = test_case
            with pytest.raises(ValueError) as excinfo:
                Trainer(args=args, task=task, model=model, model_name=model_name)
            assert expected in str(excinfo.value)

    def test_init_with_valid_arguments(self):
        """
        Feature: Trainer.__init__()
        Description: Test trainer init with valid arguments.
        Expectation: No exception
        """
        test_cases = [
            (None, "text_generation", "llama2_7b", None),
            (None, "text_generation", None, "llama2_7b"),
            (None, "text_generation", "llama2_7b", "llama2_7b"),
            (None, "text_generation", MODEL, None),
            (None, "text_generation", MODEL, "llama2_7b"),
            (MINDFORMER_CONFIG, "general", None, None),
            (MINDFORMER_CONFIG, "general", "llama2_7b", None),
            (MINDFORMER_CONFIG, "general", None, "llama2_7b"),
            (MINDFORMER_CONFIG, "general", "llama2_7b", "llama2_7b"),
            (MINDFORMER_CONFIG, "general", MODEL, None),
            (MINDFORMER_CONFIG, "general", MODEL, "llama2_7b"),
            (MINDFORMER_CONFIG, "text_generation", None, None),
            (MINDFORMER_CONFIG, "text_generation", "llama2_7b", None),
            (MINDFORMER_CONFIG, "text_generation", None, "llama2_7b"),
            (MINDFORMER_CONFIG, "text_generation", "llama2_7b", "llama2_7b"),
            (MINDFORMER_CONFIG, "text_generation", MODEL, None),
            (MINDFORMER_CONFIG, "text_generation", MODEL, "llama2_7b"),
            (TRAINING_ARGUMENTS, "general", MODEL, None),
            (TRAINING_ARGUMENTS, "general", MODEL, "llama2_7b"),
            (TRAINING_ARGUMENTS, "text_generation", "llama2_7b", None),
            (TRAINING_ARGUMENTS, "text_generation", None, "llama2_7b"),
            (TRAINING_ARGUMENTS, "text_generation", "llama2_7b", "llama2_7b"),
            (TRAINING_ARGUMENTS, "text_generation", MODEL, None),
            (TRAINING_ARGUMENTS, "text_generation", MODEL, "llama2_7b")]
        for test_case in test_cases:
            args, task, model, model_name = test_case
            Trainer(args=args, task=task, model=model, model_name=model_name)

    def test_add_callback(self):
        """
        Feature: Trainer with callbacks
        Description: Test trainer add callback.
        Expectation: No exception
        """
        trainer = Trainer(args=TRAINING_ARGUMENTS, model=MODEL)
        callback = MFLossMonitor()

        trainer.add_callback(callback)
        assert trainer.callback_list == "MFLossMonitor"

        trainer.add_callback(MFLossMonitor)
        assert trainer.callback_list == "MFLossMonitor\nMFLossMonitor"

    def test_remove_callback(self):
        """
        Feature: Trainer with callbacks
        Description: Test trainer remove callback.
        Expectation: No exception
        """
        trainer = Trainer(args=TRAINING_ARGUMENTS, model=MODEL)
        callback = MFLossMonitor()

        trainer.add_callback(callback)
        trainer.remove_callback(callback)
        assert trainer.callback_list == ""

        trainer.add_callback(callback)
        trainer.remove_callback(MFLossMonitor)
        assert trainer.callback_list == ""

    def test_pop_callback(self):
        """
        Feature: Trainer with callbacks
        Description: Test trainer pop callback.
        Expectation: No exception
        """
        trainer = Trainer(args=TRAINING_ARGUMENTS, model=MODEL)
        callback = MFLossMonitor()

        trainer.add_callback(callback)
        callback1 = trainer.pop_callback(callback)
        assert trainer.callback_list == ""

        trainer.add_callback(callback)
        callback2 = trainer.pop_callback(MFLossMonitor)
        assert callback1 == callback2


def run_trainer(args, task, model, model_name, train_dataset, eval_dataset, tokenizer):
    """static method of running trainer."""
    trainer = Trainer(args=args, task=task, model=model, model_name=model_name,
                      train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer)
    trainer.config.runner_config.epochs = EPOCHS
    trainer.config.eval_step_interval = EVAL_STEPS

    trainer.train(do_eval=True)
    trainer.evaluate(eval_dataset=eval_dataset)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_trainer_with_args():
    """
    Feature: Trainer
    Description: Test trainer without args.
    Expectation: No exception
    """
    run_trainer(MINDFORMER_CONFIG, "text_generation", None, "llama2_7b", TRAIN_DATASET_FOR_TRAINER_WITH_ARGS,
                EVAL_DATASET_FOR_TRAINER_WITH_ARGS, TOKENIZER)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_trainer_without_args():
    """
    Feature: Trainer
    Description: Test trainer without args.
    Expectation: No exception
    """
    run_trainer(None, "text_generation", MODEL, "llama2_7b", TRAIN_DATASET_FOR_TRAINER_WITHOUT_ARGS,
                EVAL_DATASET_FOR_TRAINER_WITHOUT_ARGS, TOKENIZER)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_trainer_with_generator():
    """
    Feature: Trainer
    Description: Test trainer with generator.
    Expectation: No exception
    """
    run_trainer(MINDFORMER_CONFIG, "general", MODEL, None, generator_train, generator_eval, TOKENIZER)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_trainer_with_iterable_dataset():
    """
    Feature: Trainer
    Description: Test trainer with iterable dataset.
    Expectation: No exception
    """
    run_trainer(TRAINING_ARGUMENTS, "general", MODEL, None, IterableTrain(), IterableEval(), TOKENIZER)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_trainer_with_accessible_dataset():
    """
    Feature: Trainer
    Description: Test trainer with accessible dataset.
    Expectation: No exception
    """
    run_trainer(TRAINING_ARGUMENTS, "text_generation", MODEL, "llama2_7b", AccessibleTrain(), AccessibleEval(),
                TOKENIZER)
