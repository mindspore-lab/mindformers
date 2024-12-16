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
Test module for testing the interface used for mindformers.
How to run this:
pytest tests/st/test_convert_reserved
"""
import os.path
import tempfile
from unittest.mock import patch
import pytest
import torch

import mindspore as ms

from mindformers.models.swin.convert_reversed import convert_ms_to_pt as swin_convert_ms_to_pt
from mindformers.models.vit.convert_reversed import convert_ms_to_pt as vit_convert_ms_to_pt
from mindformers.models.bloom.convert_reversed import convert_ms_to_pt as bloom_convert_ms_to_pt
from mindformers.models.gpt2.convert_reversed import convert_ms_to_pt as gpt2_convert_ms_to_pt
from mindformers.models.llama.convert_reversed import convert_ms_to_pt as llama_convert_ms_to_pt
from mindformers.models.blip2.convert_reversed import convert_ms_to_pt as blip2_convert_ms_to_pt
from mindformers.models.glm.convert_reversed import convert_ms_to_pt as glm_convert_ms_to_pt
from mindformers.models.glm2.convert_reversed import convert_ms_to_pt as glm2_convert_ms_to_pt
from mindformers.models.whisper.convert_reversed import convert_ms_to_pt as whisper_convert_ms_to_pt

temp_dir = tempfile.TemporaryDirectory()
path = temp_dir.name

class ModelDict:
    """A mock dict class for testing convert_reserved."""
    def __init__(self):
        self._dict = {}

    def state_dict(self):
        return self._dict

    def items(self):
        return self._dict

    def half(self):
        return self

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("mindspore.load_checkpoint")
def test_swin_convert_reversed(mock_get):
    """
    Feature: swin convert_reversed.
    Description: Test basic function of swin convert_reversed.
    Expectation: success
    """
    model_dict = ModelDict()
    model_dict.state_dict()[".q.k.v"] = ms.Parameter([1])
    model_dict.state_dict()[(".mlp.projection.weight.norm.beta.gamma"
                             ".relative_position_bias.relative_positionpatch_embed.decoder")] = ms.Parameter([2])
    mock_get.return_value = model_dict.state_dict()
    swin_convert_ms_to_pt(path, os.path.join(path, "test_swin.safetensors"), is_pretrain=True)
    test_dict = torch.load(os.path.join(path, "test_swin.safetensors"))["model"]
    assert test_dict is not None

    test_tensor = test_dict[
        ".mlp.fc2.weight.norm.bias.weight.relative_positionpatch_embed.decoder.0"]
    assert test_tensor is not None
    assert test_tensor.numpy() == [2]

    swin_convert_ms_to_pt(path, os.path.join(path, "test_swin_.safetensors"), is_pretrain=False)
    test_dict = torch.load(os.path.join(path, "test_swin_.safetensors"))["model"]
    assert test_dict is not None

    test_tensor = test_dict[
        ".mlp.fc2.weight.norm.bias.weight.relative_positionpatch_embed.decoder"]
    assert test_tensor is not None
    assert test_tensor.numpy() == [2]

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("mindspore.load_checkpoint")
def test_vit_convert_ms_to_pt(mock_get):
    """
    Feature: vit convert_reversed.
    Description: Test basic function of vit convert_reversed.
    Expectation: success
    """
    model_dict = ModelDict()
    model_dict.state_dict()[
        "attention.projection.weight.output.mapping.beta.gamma.layernorm.head.cls_tokens.dense1"] = ms.Parameter([1])
    model_dict.state_dict()[
        "attention.projection.weight.output.mapping.beta.gamma.layernorm.head.cls_tokens.dense2"] = ms.Parameter([2])
    model_dict.state_dict()[
        "attention.projection.weight.output.mapping.beta.gamma.layernorm.head.cls_tokens.dense3"] = ms.Parameter([3])
    mock_get.return_value = model_dict.state_dict()
    vit_convert_ms_to_pt(path, os.path.join(path, "test_vit.safetensors"))
    test_dict = torch.load(os.path.join(path, "test_vit.safetensors"))["model"]
    assert test_dict is not None
    test_tensor = test_dict[
        "attn.proj.weight.mlp.fc1.bias.gamma.norm.head.cls_tokens.qkv"]
    assert test_tensor is not None

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("mindspore.load_checkpoint")
def test_bloom_convert_ms_to_pt(mock_get):
    """
    Feature: bloom convert_reversed.
    Description: Test basic function of bloom convert_reversed.
    Expectation: success
    """
    model_dict = ModelDict()
    model_dict.state_dict()["projection.1.attention.dense1.weight"] = ms.Parameter([[[[1, 2, 3]]]])
    model_dict.state_dict()["projection.1.attention.dense2.weight"] = ms.Parameter([[[[1, 2, 3]]]])
    model_dict.state_dict()["projection.1.attention.dense3.weight"] = ms.Parameter([[[[1, 2, 3]]]])
    mock_get.return_value = model_dict.state_dict()

    bloom_convert_ms_to_pt(path, os.path.join(path, "test_bloom.safetensors"), n_head=1, hidden_size=1)
    test_dict = torch.load(os.path.join(path, "test_bloom.safetensors"))
    assert test_dict is not None
    test_tensor = test_dict[
        "projection.1.self_attention.query_key_value.weight"]
    assert test_tensor is not None

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("mindspore.load_checkpoint")
def test_gpt2_convert_ms_to_pt(mock_get):
    """
    Feature: gpt2 convert_reversed.
    Description: Test basic function of gpt2 convert_reversed.
    Expectation: success
    """
    model_dict = ModelDict()
    model_dict.state_dict()["backbone.blocks.0.attention.dense1.weight"] = ms.Parameter([[1, 2]])
    model_dict.state_dict()["backbone.blocks.0.attention.dense2.weight"] = ms.Parameter([[1, 2]])
    model_dict.state_dict()["backbone.blocks.0.attention.dense3.weight"] = ms.Parameter([[1, 2]])
    model_dict.state_dict()["backbone.layernorm.gamma"] = ms.Parameter([4])
    model_dict.state_dict()["backbone.layernorm.beta"] = ms.Parameter([5])
    model_dict.state_dict()["backbone.embedding.word_embedding.embedding_table"] = ms.Parameter([6])
    model_dict.state_dict()["backbone.embedding.position_embedding.embedding_table"] = ms.Parameter([7])
    mock_get.return_value = model_dict.state_dict()

    gpt2_convert_ms_to_pt(path, os.path.join(path, "test_gpt2.safetensors"))
    test_dict = torch.load(os.path.join(path, "test_gpt2.safetensors"))
    assert test_dict is not None
    test_tensor = test_dict["ln_f.weight"]
    assert test_tensor is not None
    assert test_tensor.numpy() == [4]
    test_tensor = test_dict["ln_f.bias"]
    assert test_tensor is not None
    assert test_tensor.numpy() == [5]
    test_tensor = test_dict["wte.weight"]
    assert test_tensor is not None
    assert test_tensor.numpy() == [6]
    test_tensor = test_dict["wpe.weight"]
    assert test_tensor is not None
    assert test_tensor.numpy() == [7]
    test_tensor = test_dict["h.0.attn.c_attn.weight"]
    assert test_tensor is not None

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("mindspore.load_checkpoint")
def test_llama_convert_ms_to_pt(mock_get):
    """
    Feature: llama convert_reversed.
    Description: Test basic function of llama convert_reversed.
    Expectation: success
    """
    model_dict = ModelDict()
    model_dict.state_dict()["lora.test"] = ms.Parameter([1])
    mock_get.return_value = model_dict.state_dict()
    llama_convert_ms_to_pt(path, os.path.join(path, "test_llama.safetensors"))
    test_dict = torch.load(os.path.join(path, "test_llama.safetensors"))
    assert test_dict is not None

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("mindspore.load_checkpoint")
def test_blip2_convert_ms_to_pt(mock_get):
    """
    Feature: blip2 convert_reversed.
    Description: Test basic function of blip2 convert_reversed.
    Expectation: success
    """
    model_dict = ModelDict()
    model_dict.state_dict()["qkv.weight.attention.dense1"] = ms.Parameter([1])
    model_dict.state_dict()["qkv.weight.attention.dense2"] = ms.Parameter([2])
    model_dict.state_dict()["qkv.weight.attention.dense3"] = ms.Parameter([3])
    model_dict.state_dict()["attention.dense2.bias"] = ms.Parameter([4])
    model_dict.state_dict()["output.mapping.weight"] = ms.Parameter([5])
    mock_get.return_value = model_dict.state_dict()
    blip2_convert_ms_to_pt(path, os.path.join(path, "test_blip2.safetensors"))
    test_dict = torch.load(os.path.join(path, "test_blip2.safetensors"))["model"]
    assert test_dict is not None
    assert isinstance(test_dict, dict)
    test_tensor = test_dict["mlp.fc2.weight"]
    assert test_tensor is not None
    assert test_tensor.numpy() == [5]
    test_tensor = test_dict["qkv.weight.attn.qkv"]
    assert test_tensor is not None

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("mindspore.load_checkpoint")
def test_glm_convert_ms_to_pt(mock_get):
    """
    Feature: glm convert_reversed.
    Description: Test basic function of glm convert_reversed.
    Expectation: success
    """
    model_dict = ModelDict()
    model_dict.state_dict()["post_attention_layernorm"] = ms.Parameter([1])
    model_dict.state_dict()["word_embeddings.embedding_table"] = ms.Parameter([2])
    model_dict.state_dict()["tk_delta_lora_a"] = ms.Parameter([3])
    mock_get.return_value = model_dict.state_dict()
    glm_convert_ms_to_pt(path, os.path.join(path, "test_glm.safetensors"))
    test_dict = torch.load(os.path.join(path, "test_glm.safetensors"))
    assert test_dict is not None
    test_tensor = test_dict["post_attention_layernorm"]
    assert test_tensor is not None
    assert test_tensor.numpy() == [1]
    test_tensor = test_dict["word_embeddings.weight"]
    assert test_tensor is not None
    assert test_tensor.numpy() == [2]
    test_tensor = test_dict["lora_A.weight"]
    assert test_tensor is not None
    assert test_tensor.numpy() == [3]

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("mindspore.load_checkpoint")
def test_glm2_convert_ms_to_pt(mock_get):
    """
    Feature: glm2 convert_reversed.
    Description: Test basic function of glm2 convert_reversed.
    Expectation: success
    """
    model_dict = ModelDict()
    model_dict.state_dict()["embedding_table"] = ms.Parameter([1])
    model_dict.state_dict()["tk_delta_lora_a"] = ms.Parameter([2])
    mock_get.return_value = model_dict.state_dict()
    glm2_convert_ms_to_pt(path, os.path.join(path, "test_glm2.safetensors"))
    test_dict = torch.load(os.path.join(path, "test_glm2.safetensors"))
    assert test_dict is not None
    test_tensor = test_dict["word_embeddings.weight"]
    assert test_tensor is not None
    assert test_tensor.numpy() == [1]
    test_tensor = test_dict["tk_delta_lora_a"]
    assert test_tensor is not None
    assert test_tensor.numpy() == [2]

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("mindspore.load_checkpoint")
def test_whisper_convert_ms_to_pt(mock_get):
    """
    Feature: whisper convert_reversed.
    Description: Test basic function of whisper convert_reversed.
    Expectation: success
    """
    model_dict = ModelDict()
    model_dict.state_dict()["embed_tokens.embedding_weight"] = ms.Parameter([1])
    model_dict.state_dict()["embed_positions.embedding_weight"] = ms.Parameter([2])
    model_dict.state_dict()["layer_norm"] = ms.Parameter([3])
    mock_get.return_value = model_dict.state_dict()
    whisper_convert_ms_to_pt(path, os.path.join(path, "test_whisper.safetensors"), dtype=torch.float32)
    test_dict = torch.load(os.path.join(path, "test_whisper.safetensors"))
    assert test_dict is not None
    test_tensor = test_dict["embed_tokens.weight"]
    assert test_tensor is not None
    assert test_tensor.numpy() == [1]
    test_tensor = test_dict["embed_positions.weight"]
    assert test_tensor is not None
    assert test_tensor.numpy() == [2]
    test_tensor = test_dict["layer_norm"]
    assert test_tensor is not None
    assert test_tensor.numpy() == [3]
