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
"""test auto_class.py"""
import os
import tempfile
import yaml
import pytest
import numpy as np
import mindspore as ms
from mindformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, GPT2Processor
from mindformers.auto_class import AutoModel, AutoConfig, AutoTokenizer, AutoProcessor
from tests.st.test_ut.test_tokenizers.get_vocab_model import get_bbpe_vocab_model


tmp_dir = tempfile.TemporaryDirectory()
tmp_path = tmp_dir.name
target_path = os.path.join(tmp_path, "gpt2")
os.makedirs(target_path, exist_ok=True)
yaml_path = os.path.join(target_path, "gpt2.yaml")
ckpt_path = os.path.join(target_path, "gpt2.ckpt")
get_bbpe_vocab_model("gpt", target_path)
vocab_path = os.path.join(target_path, "gpt_vocab.json")
merges_path = os.path.join(target_path, "gpt_merges.txt")


def mock_yaml():
    """mock gpt2 yaml file"""
    useless_names = ["_name_or_path", "tokenizer_class", "architectures", "is_encoder_decoder",
                     "is_sample_acceleration", "parallel_config", "moe_config"]
    gpt2_ori_config = GPT2Config().to_dict()
    for name in useless_names:
        gpt2_ori_config.pop(name, None)
    gpt2_ori_config["num_layers"] = 1
    gpt2_ori_config["type"] = "GPT2Config"
    gpt2_config = {"model": {"arch": {"type": "GPT2LMHeadModel"}, "model_config": gpt2_ori_config},
                   "processor": {"return_tensors": "ms",
                                 "tokenizer": {
                                     "unk_token": "<|endoftext|>",
                                     "bos_token": "<|endoftext|>",
                                     "eos_token": "<|endoftext|>",
                                     "pad_token": "<|endoftext|>",
                                     "vocab_file": vocab_path,
                                     "merges_file": merges_path,
                                     "type": "GPT2Tokenizer"
                                 },
                                 "type": "GPT2Processor"
                                 }
                   }
    gpt2_config["model"]["model_config"]["checkpoint_name_or_path"] = "gpt2"
    with open(yaml_path, "w", encoding="utf-8") as w:
        yaml.dump(gpt2_config, w, default_flow_style=False)


def mock_ckpt():
    """mock gpt2 one layer ckpt"""
    state_dict = {
        "backbone.blocks.0.layernorm1.gamma": ms.Tensor(np.random.random((768,)).astype(np.float16)),
        "backbone.blocks.0.layernorm1.beta": ms.Tensor(np.random.random((768,)).astype(np.float16)),
        "backbone.blocks.0.layernorm2.gamma": ms.Tensor(np.random.random((768,)).astype(np.float16)),
        "backbone.blocks.0.layernorm2.beta": ms.Tensor(np.random.random((768,)).astype(np.float16)),
        "backbone.blocks.0.attention.projection.weight": ms.Tensor(np.random.random((768, 768)).astype(np.float16)),
        "backbone.blocks.0.attention.projection.bias": ms.Tensor(np.random.random((768,)).astype(np.float16)),
        "backbone.blocks.0.attention.dense1.weight": ms.Tensor(np.random.random((768, 768)).astype(np.float16)),
        "backbone.blocks.0.attention.dense1.bias": ms.Tensor(np.random.random((768,)).astype(np.float16)),
        "backbone.blocks.0.attention.dense2.weight": ms.Tensor(np.random.random((768, 768)).astype(np.float16)),
        "backbone.blocks.0.attention.dense2.bias": ms.Tensor(np.random.random((768,)).astype(np.float16)),
        "backbone.blocks.0.attention.dense3.weight": ms.Tensor(np.random.random((768, 768)).astype(np.float16)),
        "backbone.blocks.0.attention.dense3.bias": ms.Tensor(np.random.random((768,)).astype(np.float16)),
        "backbone.blocks.0.output.mapping.weight": ms.Tensor(np.random.random((768, 3072)).astype(np.float16)),
        "backbone.blocks.0.output.mapping.bias": ms.Tensor(np.random.random((3072,)).astype(np.float16)),
        "backbone.blocks.0.output.projection.weight": ms.Tensor(np.random.random((3072, 768)).astype(np.float16)),
        "backbone.blocks.0.output.projection.bias": ms.Tensor(np.random.random((768,)).astype(np.float16)),
        "backbone.layernorm.gamma": ms.Tensor(np.random.random((768,)).astype(np.float16)),
        "backbone.layernorm.gambetama": ms.Tensor(np.random.random((768,)).astype(np.float16)),
        "backbone.embedding.word_embedding.embedding_table":
            ms.Tensor(np.random.random((50257, 768)).astype(np.float16)),
        "backbone.embedding.position_embedding.embedding_table":
            ms.Tensor(np.random.random((1024, 768)).astype(np.float16))
    }
    params = []
    for k, v in state_dict.items():
        params.append({"name": k, "data": v})
    ms.save_checkpoint(params, ckpt_path)


mock_yaml()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_auto_config():
    """
    Feature: auto_class.AutoConfig
    Description: test AutoConfig function
    Expectation: success
    """
    config = AutoConfig.from_pretrained(yaml_path)
    assert isinstance(config, GPT2Config)
    config = AutoConfig.from_pretrained("gpt2")
    assert isinstance(config, GPT2Config)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_auto_model():
    """
    Feature: auto_class.AutoModel
    Description: test AutoModel function
    Expectation: success
    """
    mock_ckpt()
    model = AutoModel.from_pretrained(target_path)
    assert isinstance(model, GPT2LMHeadModel)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_auto_tokenizer():
    """
    Feature: auto_class.AutoTokenizer
    Description: test AutoTokenizer function
    Expectation: success
    """
    tokenizer = AutoTokenizer.from_pretrained(target_path)
    assert isinstance(tokenizer, GPT2Tokenizer)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_auto_processor():
    """
    Feature: auto_class.AutoProcessor
    Description: test AutoProcessor function
    Expectation: success
    """
    processor = AutoProcessor.from_pretrained(target_path)
    assert isinstance(processor, GPT2Processor)
    processor = AutoProcessor.from_pretrained(yaml_path)
    assert isinstance(processor, GPT2Processor)
