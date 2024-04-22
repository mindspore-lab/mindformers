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
os.environ["OPENMIND_HUB_ENDPOINT"] = "https://openmind.test.osinfra.cn"
# pylint: disable=C0413
import pytest
from mindformers import GPT2Tokenizer, BertTokenizer
from mindformers.models.auto import AutoTokenizer
from research.baichuan2.baichuan2_tokenizer import Baichuan2Tokenizer


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
        self.baichuan_tokenizer = AutoTokenizer.from_pretrained(self.baichuan_repo_id, trust_remote_code=True)

        self.bert_id_output = [1045, 2293, 7211, 1012]

    @pytest.mark.level0
    def test_mf_origin_from_pretrained_model_name(self):
        """yaml logic: test from_pretrained(model_name) to instantiate tokenizer"""
        self.gpt_tokenizer = AutoTokenizer.from_pretrained(self.gpt_model_name)
        assert self.gpt_tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.gpt_id_output

    @pytest.mark.level1
    def test_mf_origin_save_pretrained(self):
        """yaml logic: test save_pretrained(save_json=default or False)"""
        self.gpt_tokenizer.save_pretrained(self.gpt_save_pretrained_origin_dir)

    @pytest.mark.level1
    def test_mf_save_vocabulary(self):
        """test save_vocabulary"""
        if not os.path.exists(self.gpt_save_vocabulary_dir):
            os.mkdir(self.gpt_save_vocabulary_dir)
        self.gpt_tokenizer.save_vocabulary(self.gpt_save_vocabulary_dir)

    @pytest.mark.level2
    def test_mf_origin_from_pretrained_dir(self):
        """yaml logic: test from_pretrained(directory of save_pretrained) to instantiate tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(self.gpt_save_pretrained_origin_dir)
        assert tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.gpt_id_output

    @pytest.mark.level2
    def test_mf_origin_tokenizer_class_from_pretrained_vocab_dir(self):
        """yaml logic: test tokenizer_class.from_pretrained(directory from save_vocabulary) to instantiate tokenizer"""
        tokenizer = GPT2Tokenizer.from_pretrained(self.gpt_save_vocabulary_dir)
        assert tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.gpt_id_output

    def test_mf_origin_tokenizer_class_from_pretrained_model_name(self):
        """yaml logic: test tokenizer_class.from_pretrained(model_name) to instantiate tokenizer"""
        tokenizer = GPT2Tokenizer.from_pretrained(self.gpt_model_name)
        assert tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.gpt_id_output

    def test_mf_experimental_from_pretrained_repo_id(self):
        """json logic: test from_pretrained(repo_id) to instantiate tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(self.gpt_repo_id)
        assert tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.gpt_id_output

    @pytest.mark.level2
    def test_mf_experimental_save_pretrained(self):
        """json: test save_pretrained(save_json=True)"""
        self.gpt_tokenizer.save_pretrained(self.gpt_save_pretrained_experimental_dir, save_json=True)

    @pytest.mark.level2
    def test_mf_experimental_from_pretrained_dir(self):
        """json logic: test from_pretrained(directory from save_pretrained) to instantiate tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(self.gpt_save_pretrained_experimental_dir, tokenizer_type="gpt2",
                                                  use_fast=False)
        assert tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.gpt_id_output

    @pytest.mark.level2
    def test_mf_experimental_not_through_tokenizer_config(self):
        """json logic: test not from tokenizer_config.json to get tokenizer_class"""
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

    def test_mf_experimental_tokenizer_class_from_pretrained_repo_id(self):
        """json logic: test tokenizer_class.from_pretrained(repo_id) to instantiate tokenizer"""
        tokenizer = GPT2Tokenizer.from_pretrained(self.gpt_repo_id)
        assert tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.gpt_id_output

    @pytest.mark.level0
    def test_plugin_experimental_from_pretrained_repo_id(self):
        """plugin tokenizer-jsonlogic: test from_pretrained(repo_id) to instantiate tokenizer"""
        self.baichuan_tokenizer = AutoTokenizer.from_pretrained(self.baichuan_repo_id, trust_remote_code=True)
        assert self.baichuan_tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == \
               self.baichuan_id_output

    @pytest.mark.level1
    def test_plugin_experimental_save_pretrained(self):
        """plugin tokenizer-json logic: test save_pretrained(save_json=True)"""
        self.baichuan_tokenizer.save_pretrained(self.baichuan_save_pretrained_experimental_dir, save_json=True)

    @pytest.mark.level1
    def test_plugin_save_vocabulary(self):
        """plugin tokenizer: test save_vocabulary"""
        if not os.path.exists(self.baichuan_save_vocabulary_dir):
            os.mkdir(self.baichuan_save_vocabulary_dir)
        self.baichuan_tokenizer.save_vocabulary(self.baichuan_save_vocabulary_dir)

    @pytest.mark.level2
    def test_plugin_experimental_from_pretrained_dir(self):
        """plugin tokenizer-json logic: test from_pretrained(directory from save_pretrained) to instantiate tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(self.baichuan_save_pretrained_experimental_dir,
                                                  trust_remote_code=True)
        assert tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.baichuan_id_output

        try:
            AutoTokenizer.from_pretrained(self.baichuan_save_pretrained_experimental_dir)
        except ValueError as e:
            # when tokenizer class not in mindformers, you can not instantiate it through yaml
            print(e)

    @pytest.mark.level1
    def test_plugin_origin_save_pretrained(self):
        """plugin tokenizer-yaml logic: test save_pretrained(save_json=default or False)"""
        self.baichuan_tokenizer.save_pretrained(self.baichuan_save_pretrained_origin_dir)

    @pytest.mark.level2
    def test_plugin_origin_from_pretrained_dir(self):
        """plugin tokenizer-yaml logic: test from_pretrained(directory from save_pretrained) to instantiate tokenizer"""
        try:
            Baichuan2Tokenizer.from_pretrained(self.baichuan_save_pretrained_origin_dir)
        except ValueError as e:
            # if it has used save_pretrained(save_json=False) to save plugin tokenizer,
            # you cannot use from_pretrained() to get tokenizer instance.
            print(e)

    @pytest.mark.level2
    def test_plugin_origin_tokenizer_class_from_pretrained_vocab_dir(self):
        """plugin tokenizer-yaml logic: test tokenizer_class.from_pretrained(directory from save_vocabulary)
        to instantiate tokenizer"""
        tokenizer = Baichuan2Tokenizer.from_pretrained(self.baichuan_save_vocabulary_dir)
        assert tokenizer(self.str_input, add_special_tokens=False)["input_ids"] == self.baichuan_id_output

    def test_mf_tokenizer_class_from_pretrained_single_vocab_file(self):
        """hub logic: test tokenizer_class.from_pretrained(single vocab file) to instantiate tokenizer"""
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
