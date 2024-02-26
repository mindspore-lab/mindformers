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
"""test auto tokenizer"""
import json
import shutil
import os
import pytest
# pylint: disable=C0413
from mindformers import GPT2Tokenizer, BertTokenizer
from mindformers.models.auto import AutoTokenizer
# from research.baichuan2.baichuan2_tokenizer import Baichuan2Tokenizer


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestTokenizer:
    """
    test auto tokenizer whether tokenizer compatibles to two mode (origin-yaml and hub-json).
    """
    def setup_class(self):
        """be like class's __init__"""
        self.gpt_model_name = "gpt2"
        self.gpt_save_pretrained_origin_dir = "./gpt2_origin_save_pretrained"
        self.gpt_save_vocabulary_dir = "./gpt2_save_vocabulary"
        self.gpt_repo_id = "mindformersinfra/test_auto_tokenizer_gpt2_ms"
        self.gpt_save_pretrained_experimental_dir = "./gpt2_experimental_save_pretrained"

        self.baichuan_repo_id = "mindformersinfra/test_auto_tokenizer_baichuan2_ms"
        self.baichuan_save_pretrained_origin_dir = "./baichuan_origin_save_pretrained"
        self.baichuan_save_vocabulary_dir = "./baichuan_save_vocabulary"
        self.baichuan_save_pretrained_experimental_dir = "./baichuan_experimental_save_pretrained"

        self.bert_save_vocabulary_dir = "./bert_save_vocabulary"

        self.str_input = "I love Beijing."
        self.gpt_id_output = [40, 1842, 11618, 13]
        self.gpt_tokenizer = AutoTokenizer.from_pretrained(self.gpt_model_name)

        self.baichuan_id_output = [92346, 2950, 19801, 72]
        # self.baichuan_tokenizer = AutoTokenizer.from_pretrained(self.baichuan_repo_id)

        self.bert_id_output = [1045, 2293, 7211, 1012]

    @pytest.mark.level0
    def test_mf_origin_from_pretrained_model_name(self):
        """yaml逻辑：测试from_pretrained接口从模型关键字实例化tokenizer"""
        self.gpt_tokenizer = AutoTokenizer.from_pretrained(self.gpt_model_name)
        assert self.gpt_tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.gpt_id_output

    @pytest.mark.level1
    def test_mf_origin_save_pretrained(self):
        """yaml逻辑：测试save_pretrained(save_json=default or False)接口"""
        self.gpt_tokenizer.save_pretrained(self.gpt_save_pretrained_origin_dir)

    @pytest.mark.level1
    def test_mf_save_vocabulary(self):
        """测试save_vocabulary接口"""
        if not os.path.exists(self.gpt_save_vocabulary_dir):
            os.mkdir(self.gpt_save_vocabulary_dir)
        self.gpt_tokenizer.save_vocabulary(self.gpt_save_vocabulary_dir)

    @pytest.mark.level2
    def test_mf_origin_from_pretrained_dir(self):
        """yaml逻辑：测试from_pretrained接口从save_pretrained的文件夹实例化tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(self.gpt_save_pretrained_origin_dir)
        assert tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.gpt_id_output

    @pytest.mark.level2
    def test_mf_origin_tokenizer_class_from_pretrained_vocab_dir(self):
        """yaml逻辑：测试tokenizer_class.from_pretrained接口从save_vocabulary的文件夹实例化tokenizer"""
        tokenizer = GPT2Tokenizer.from_pretrained(self.gpt_save_vocabulary_dir)
        assert tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.gpt_id_output

    def test_mf_origin_tokenizer_class_from_pretrained_model_name(self):
        """yaml逻辑：测试tokenizer_class.from_pretrained接口从模型关键字实例化tokenizer"""
        tokenizer = GPT2Tokenizer.from_pretrained(self.gpt_model_name)
        assert tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.gpt_id_output

    # def test_mf_experimental_from_pretrained_repo_id(self):
    #     """json逻辑：测试from_pretrained接口从远端repo id实例化tokenizer"""
    #     tokenizer = AutoTokenizer.from_pretrained(self.gpt_repo_id)
    #     assert tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.gpt_id_output

    @pytest.mark.level2
    def test_mf_experimental_save_pretrained(self):
        """json：测试save_pretrained(save_json=True)接口"""
        self.gpt_tokenizer.save_pretrained(self.gpt_save_pretrained_experimental_dir, save_json=True)

    @pytest.mark.level2
    def test_mf_experimental_from_pretrained_dir(self):
        """json逻辑：测试from_pretrained接口从save_pretrained的文件夹实例化tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(self.gpt_save_pretrained_experimental_dir, tokenizer_type="gpt2",
                                                  use_fast=False)
        assert tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.gpt_id_output

    @pytest.mark.level2
    def test_mf_experimental_not_through_tokenizer_config(self):
        """json逻辑：测试不从tokenizer_config.json读取而从其他方式获取tokenizer_class的方式"""
        tokenizer_config_path = os.path.join(self.gpt_save_pretrained_experimental_dir, "tokenizer_config.json")
        os.remove(tokenizer_config_path)
        config_path = os.path.join(self.gpt_save_pretrained_experimental_dir, "config.json")

        # test experimental type: del tokenizer_config.json, and build through config.json - tokenizer_class
        config = {"tokenizer_class": "GPT2Tokenizer"}
        with open(config_path, "w", encoding="utf-8") as w:
            w.write(json.dumps(config))
        tokenizer = AutoTokenizer.from_pretrained(self.gpt_save_pretrained_experimental_dir)
        assert tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.gpt_id_output

        # test experimental type: del tokenizer_config.json, and build through config.json - model_type
        config = {"model_type": "gpt2"}
        with open(config_path, "w", encoding="utf-8") as w:
            w.write(json.dumps(config))
        tokenizer = AutoTokenizer.from_pretrained(self.gpt_save_pretrained_experimental_dir)
        assert tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.gpt_id_output

    # def test_mf_experimental_tokenizer_class_from_pretrained_repo_id(self):
    #     """json逻辑：测试tokenizer_class.from_pretrained接口从远端repo id实例化tokenizer"""
    #     tokenizer = GPT2Tokenizer.from_pretrained(self.gpt_repo_id)
    #     assert tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.gpt_id_output

    # # @pytest.mark.level0
    # def test_plugin_experimental_from_pretrained_repo_id(self):
    #     """外挂tokenizer-json逻辑：测试from_pretrained接口从repo id实例化tokenizer"""
    #     self.baichuan_tokenizer = AutoTokenizer.from_pretrained(self.baichuan_repo_id)
    #     assert self.baichuan_tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == \
    #            self.baichuan_id_output
    #
    # # @pytest.mark.level1
    # def test_plugin_experimental_save_pretrained(self):
    #     """外挂tokenizer-json逻辑：测试save_pretrained(save_json=True)接口"""
    #     self.baichuan_tokenizer.save_pretrained(self.baichuan_save_pretrained_experimental_dir, save_json=True)
    #
    # # @pytest.mark.level1
    # def test_plugin_save_vocabulary(self):
    #     """外挂tokenizer：测试save_vocabulary接口"""
    #     if not os.path.exists(self.baichuan_save_vocabulary_dir):
    #         os.mkdir(self.baichuan_save_vocabulary_dir)
    #     self.baichuan_tokenizer.save_vocabulary(self.baichuan_save_vocabulary_dir)
    #
    # # @pytest.mark.level2
    # def test_plugin_experimental_from_pretrained_dir(self):
    #     """外挂tokenizer-json逻辑：测试from_pretrained接口从save_pretrained的文件夹实例化tokenizer"""
    #     tokenizer = AutoTokenizer.from_pretrained(self.baichuan_save_pretrained_experimental_dir,
    #                                               trust_remote_code=True)
    #     assert tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.baichuan_id_output
    #
    #     try:
    #         AutoTokenizer.from_pretrained(self.baichuan_save_pretrained_experimental_dir)
    #     except ValueError as e:
    #         # when tokenizer class not in mindformers, you can not instantiate it through yaml
    #         print(e)
    #
    # # @pytest.mark.level1
    # def test_plugin_origin_save_pretrained(self):
    #     """外挂tokenizer-yaml逻辑：测试save_pretrained(save_json=default or False)接口"""
    #     self.baichuan_tokenizer.save_pretrained(self.baichuan_save_pretrained_origin_dir)
    #
    # # @pytest.mark.level2
    # def test_plugin_origin_from_pretrained_dir(self):
    #     """外挂tokenizer-yaml逻辑：测试from_pretrained接口从save_pretrained的文件夹实例化tokenizer"""
    #     tokenizer = AutoTokenizer.from_pretrained(self.baichuan_save_pretrained_origin_dir)
    #     assert tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.baichuan_id_output
    #
    # # @pytest.mark.level2
    # def test_plugin_origin_tokenizer_class_from_pretrained_vocab_dir(self):
    #     """外挂tokenizer-yaml逻辑：测试tokenizer_class.from_pretrained接口从save_vocabulary的文件夹实例化tokenizer"""
    #     tokenizer = Baichuan2Tokenizer.from_pretrained(self.baichuan_save_vocabulary_dir)
    #     assert tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.baichuan_id_output

    def test_mf_tokenizer_class_from_pretrained_single_vocab_file(self):
        """hub逻辑：测试tokenizer_class.from_pretrained接口从单个词表文件实例化tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained("bert_base_uncased")

        if not os.path.exists(self.bert_save_vocabulary_dir):
            os.mkdir(self.bert_save_vocabulary_dir)

        tokenizer.save_vocabulary(self.bert_save_vocabulary_dir)

        tokenizer = BertTokenizer.from_pretrained(
            os.path.join(self.bert_save_vocabulary_dir, os.listdir(self.bert_save_vocabulary_dir)[0]))
        assert tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.bert_id_output

    def teardown_class(self):
        """del reference dirs"""
        shutil.rmtree(self.gpt_save_pretrained_origin_dir, ignore_errors=True)
        shutil.rmtree(self.gpt_save_vocabulary_dir, ignore_errors=True)
        shutil.rmtree(self.gpt_save_pretrained_experimental_dir, ignore_errors=True)
        shutil.rmtree(self.baichuan_save_pretrained_experimental_dir, ignore_errors=True)
        shutil.rmtree(self.baichuan_save_vocabulary_dir, ignore_errors=True)
        shutil.rmtree(self.baichuan_save_pretrained_origin_dir, ignore_errors=True)
        shutil.rmtree(self.bert_save_vocabulary_dir, ignore_errors=True)
        # shutil.rmtree("./checkpoint_download", ignore_errors=True)    # cannot be deleted directly
