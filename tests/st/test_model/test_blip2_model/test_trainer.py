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
Test module for testing the blip2 interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_blip2_model/test_trainer.py
"""
import numpy as np
import pytest
from PIL import Image

import mindspore as ms
from mindspore.dataset import GeneratorDataset

from mindformers import Trainer, TrainingArguments, Blip2Llm, LlamaTokenizer
from mindformers.models.bert.bert_tokenizer import BertTokenizer
from mindformers.models.blip2 import Blip2Qformer, Blip2Classifier, Blip2Config, Blip2ImageProcessor
from mindformers.models.blip2.blip2_itm_evaluator import Blip2ItmEvaluator
from mindformers.models.blip2.blip2_llm import Blip2ImageToTextGeneration
from mindformers.models.blip2.qformer import QFormerConfig
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers.models.vit.vit import ViTConfig

ms.set_context(mode=0)


def generator_train():
    """train dataset generator"""
    seq_len = 8
    images = np.random.randint(low=0, high=255, size=(3, 32, 32)).astype(np.float32)
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    for _ in range(4):
        yield images, input_ids


def generator_eval():
    """eval dataset generator"""
    seq_len = 8
    images = np.random.randint(low=0, high=255, size=(3, 32, 32)).astype(np.float32)
    input_ids = np.random.randint(low=0, high=15, size=(seq_len)).astype(np.int32)
    for _ in range(4):
        yield images, input_ids


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestBlip2TrainerMethod:
    """A test class for testing pipeline."""

    def setup_method(self):
        """init task trainer."""
        train_dataset = GeneratorDataset(generator_train, column_names=["image", "input_ids"])
        eval_dataset = GeneratorDataset(generator_eval, column_names=["image", "input_ids"])
        self.train_dataset = train_dataset.batch(batch_size=2)
        self.eval_dataset = eval_dataset.batch(batch_size=2)

        qformer_config = QFormerConfig(num_hidden_layers=1,
                                       num_attention_heads=2,
                                       hidden_size=2,
                                       head_embed_dim=2,
                                       encoder_width=2,
                                       intermediate_size=2,
                                       max_position_embeddings=80,
                                       resize_token_embeddings=False)

        vision_config = ViTConfig(image_size=32,
                                  hidden_size=2,
                                  num_hidden_layers=1,
                                  num_attention_heads=2,
                                  intermediate_size=2,
                                  num_labels=10)
        model_config = Blip2Config(vision_config=vision_config,
                                   qformer_config=qformer_config)
        # for train and evaluation
        self.blip2_qformer = Blip2Qformer(model_config)
        self.blip2_evaluator = Blip2ItmEvaluator(model_config, self.blip2_qformer)
        # for prediction
        self.blip2_classifier = Blip2Classifier(model_config)
        self.image_processor = Blip2ImageProcessor(image_size=32)
        self.tokenizer = BertTokenizer.from_pretrained('bert_base_uncased')
        self.training_args = TrainingArguments(batch_size=2, num_train_epochs=1)

    def test_train(self):
        """
        Feature: Trainer.train()
        Description: Test trainer for train.
        Expectation: TypeError, ValueError, RuntimeError
        """
        task_trainer = Trainer(task='contrastive_language_image_pretrain',
                               model=self.blip2_qformer,
                               model_name='blip2_stage1_vit_g',
                               args=self.training_args,
                               tokenizer=self.tokenizer,
                               train_dataset=self.train_dataset,
                               eval_dataset=self.eval_dataset)
        task_trainer.train()

    def test_eval(self):
        """
        Feature: Trainer.evaluate()
        Description: Test trainer for evaluate.
        Expectation: TypeError, ValueError, RuntimeError
        """
        task_evaluater = Trainer(task='image_to_text_retrieval',
                                 model=self.blip2_evaluator,
                                 model_name='blip2_stage1_evaluator',
                                 tokenizer=self.tokenizer,
                                 eval_dataset=self.eval_dataset)
        task_evaluater.evaluate(k_test=1, add_extra_itm_score=False)

    def test_predict(self):
        """
        Feature: Trainer.predict()
        Description: Test trainer for predict.
        Expectation: TypeError, ValueError, RuntimeError
        """
        task_predictor = Trainer(task='zero_shot_image_classification',
                                 model=self.blip2_classifier,
                                 model_name='blip2_stage1_classification',
                                 tokenizer=self.tokenizer,
                                 image_processor=self.image_processor)
        # 加载输入，一张图片
        input_data = Image.new('RGB', (32, 32))

        # 加载指定的权重以完成推理
        task_predictor.predict(input_data=input_data)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestBlip2SecondStageTrainerMethod:
    """A test class for testing blip2 second stage trainers."""

    def setup_method(self):
        """init task trainer."""
        train_batch_size = 2
        train_dataset = GeneratorDataset(generator_train, column_names=["image", "input_ids"])
        self.train_dataset = train_dataset.batch(batch_size=train_batch_size)

        qformer_config = QFormerConfig(num_hidden_layers=1,
                                       num_attention_heads=2,
                                       hidden_size=2,
                                       head_embed_dim=2,
                                       encoder_width=2,
                                       intermediate_size=2,
                                       max_position_embeddings=512,
                                       resize_token_embeddings=False)

        vision_config = ViTConfig(image_size=32,
                                  hidden_size=2,
                                  num_hidden_layers=1,
                                  num_attention_heads=2,
                                  intermediate_size=2,
                                  num_labels=10,
                                  checkpoint_name_or_path="")

        text_config_for_trainer = LlamaConfig(seq_length=39,
                                              hidden_size=8,
                                              num_layers=2,
                                              num_heads=2,
                                              rotary_dtype="float16",
                                              checkpoint_name_or_path="")

        model_config_for_trainer = Blip2Config(batch_size=train_batch_size,
                                               vision_config=vision_config,
                                               qformer_config=qformer_config,
                                               text_config=text_config_for_trainer,
                                               checkpoint_name_or_path="",
                                               max_txt_len=7)

        text_config_for_generator = LlamaConfig(seq_length=128,
                                                hidden_size=8,
                                                num_layers=2,
                                                num_heads=2,
                                                rotary_dtype="float16",
                                                checkpoint_name_or_path="",
                                                use_past=True)
        model_config_for_generator = Blip2Config(batch_size=1,
                                                 vision_config=vision_config,
                                                 qformer_config=qformer_config,
                                                 text_config=text_config_for_generator,
                                                 checkpoint_name_or_path="",
                                                 max_txt_len=8)
        self.blip2_llm = Blip2Llm(model_config_for_trainer)

        # for prediction
        self.blip2_generator = Blip2ImageToTextGeneration(model_config_for_generator)
        self.image_processor = Blip2ImageProcessor(image_size=32)
        self.tokenizer = LlamaTokenizer.from_pretrained('llama_7b')
        self.training_args = TrainingArguments(batch_size=4, num_train_epochs=1)

    def test_train(self):
        """
        Feature: Trainer.train()
        Description: Test trainer for train.
        Expectation: TypeError, ValueError, RuntimeError
        """
        task_trainer = Trainer(task='contrastive_language_image_pretrain',
                               model=self.blip2_llm,
                               model_name='blip2_stage2_vit_g_llama_7b',
                               args=self.training_args,
                               tokenizer=self.tokenizer,
                               train_dataset=self.train_dataset)
        task_trainer.train(train_checkpoint='')

    def test_predict(self):
        """
        Feature: Trainer.predict()
        Description: Test trainer for predict.
        Expectation: TypeError, ValueError, RuntimeError
        """
        # task_predictor = Trainer(task='image_to_text_generation',
        #                          model=self.blip2_generator,
        #                          model_name='itt_blip2_stage2_vit_g_llama_7b',
        #                          tokenizer=self.tokenizer,
        #                          image_processor=self.image_processor)
        # # 加载输入，一张图片
        # input_data = Image.new('RGB', (32, 32))
        #
        # # 加载指定的权重以完成推理
        # task_predictor.predict(input_data=input_data, max_length=8)
