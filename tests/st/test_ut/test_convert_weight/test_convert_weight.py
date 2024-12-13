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
pytest tests/st/test_convert_weight
"""
import os
import json
import tempfile
from unittest.mock import patch
import pytest
from safetensors.torch import save_file
import torch

import mindspore as ms
from mindspore import save_checkpoint

from mindformers.models.glm2.convert_weight import convert_pt_to_ms as glm2_convert_pt_to_ms
from mindformers.models.glm.convert_weight import convert_pt_to_ms as glm_convert_pt_to_ms
from mindformers.models.whisper.convert_weight import convert_pt_to_ms as whisper_convert_pt_to_ms
from mindformers.models.vit.convert_weight import convert_pt_to_ms as vit_convert_pt_to_ms
from mindformers.models.clip.convert_weight import convert_weight as clip_convert_pt_to_ms
from mindformers.models.codegeex2.convert_weight import convert_pt_to_ms as codegeex2_convert_pt_to_ms
from mindformers.models.blip2.convert_weight import convert_pt_to_ms as blip2_convert_pt_to_ms
from mindformers.models.bloom.convert_weight import convert_pt_to_ms as bloom_convert_pt_to_ms
from mindformers.models.swin.convert_weight import convert_pt_to_ms as swin_convert_pt_to_ms
from mindformers.models.gpt2.convert_weight import convert_pt_to_ms as gpt2_convert_pt_to_ms
from mindformers.models.bert.convert_weight import (split_torch_attention as bert_split_torch_attention,
                                                    get_converted_ckpt as bert_get_converted_ckpt,
                                                    generate_params_dict as bert_generate_params_dict)
from mindformers.models.t5.convert_weight import generate_params_dict as t5_generate_params_dict, get_converted_ckpt as t5_get_converted_ckpt
from mindformers.models.cogvlm2.convert_weight import convert_pt_to_ms as cogvlm2_convert_pt_to_ms
from mindformers.models.llama.convert_weight import (convert_to_qkv_concat as llama_convert_to_qkv_concat,
                                                     convert_to_new_ckpt as llama_convert_to_new_ckpt,
                                                     convert_pt_to_ms as llama_convert_pt_to_ms,
                                                     convert_meta_torch_ckpt as llama_convert_meta_torch_ckpt,
                                                     convert_megatron_to_ms as llama_convert_megatron_to_ms)

temp_dir = tempfile.TemporaryDirectory()
path = temp_dir.name
os.mkdir(os.path.join(path, "test"))

class ModelDict:
    """A mock dict class for testing convert_weight."""
    def __init__(self):
        self._dict = {}
        self.list = ["self_attn.rotary_emb.inv_freq",
                     "model.vision.patch_embedding",
                     "model.vision",
                     "model.norm.weight",
                     "lm_head.weight",
                     "model.embed_tokens",
                     "self_attn.language_expert_query_key_value.weight", "test"]

    def state_dict(self):
        return self._dict

    def items(self):
        return self._dict

    def half(self):
        return self

    def keys(self):
        return self.list

    def get_tensor(self, key):
        return self._dict[key]

class MockModel:
    """A mock dict class for testing convert_weight."""
    def __init__(self):
        self.checkpoint_name_or_path = ""

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("transformers.AutoModel.from_pretrained")
def test_glm2_convert_pt_to_ms(mock_get):
    """
    Feature: glm2 convert_weight.
    Description: Test basic function of glm2 convert_weight.
    Expectation: success
    """
    model_dict = ModelDict()
    model_dict.state_dict()["word_embeddings.weight"] = torch.tensor([1])
    mock_get.return_value = model_dict

    glm2_convert_pt_to_ms(path, os.path.join(path, "test_glm2.ckpt"))
    test_dict = ms.load_checkpoint(os.path.join(path, "test_glm2.ckpt"))
    assert test_dict is not None
    test_tensor = test_dict["embedding_weight"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [1]

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("transformers.AutoModel.from_pretrained")
def test_glm_convert_pt_to_ms(mock_get):
    """
    Feature: glm convert_weight.
    Description: Test basic function of glm convert_weight.
    Expectation: success
    """
    model_dict = ModelDict()
    model_dict.state_dict()["word_embeddings.weight"] = torch.tensor([1.0])
    model_dict.state_dict()["post_attention_layernorm"] = torch.tensor([2.0])
    model_dict.state_dict()["input_layernorm"] = torch.tensor([3.0], dtype=torch.float16)
    mock_get.return_value = model_dict

    glm_convert_pt_to_ms(path, os.path.join(path, "test_glm.ckpt"))
    test_dict = ms.load_checkpoint(os.path.join(path, "test_glm.ckpt"))
    assert test_dict is not None

    test_tensor = test_dict["word_embeddings.embedding_table"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [1]
    test_tensor = test_dict["post_attention_layernorm"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [2]
    test_tensor = test_dict["input_layernorm"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [3]

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("torch.load")
def test_whisper_convert_pt_to_ms(mock_get):
    """
    Feature: whisper convert_weight.
    Description: Test basic function of whisper convert_weight.
    Expectation: success
    """
    model_dict = ModelDict()
    model_dict.state_dict()["embed_tokens.weight"] = torch.tensor([1])
    model_dict.state_dict()["embed_positions.weight"] = torch.tensor([2])
    model_dict.state_dict()["layer_norm"] = torch.tensor([3])
    model_dict.state_dict()["conv1.weight"] = torch.tensor([[1], [2], [3]])
    mock_get.return_value = model_dict.state_dict()

    whisper_convert_pt_to_ms(path, os.path.join(path, "test_whisper.ckpt"))

    test_dict = ms.load_checkpoint(os.path.join(path, "test_whisper.ckpt"))
    assert test_dict is not None

    test_tensor = test_dict["embed_tokens.embedding_weight"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [1]
    test_tensor = test_dict["embed_positions.embedding_weight"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [2]
    test_tensor = test_dict["layer_norm"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [3]
    test_tensor = test_dict["conv1.weight"]
    assert test_tensor is not None

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("torch.load")
def test_vit_convert_pt_to_ms(mock_get):
    """
    Feature: vit convert_weight.
    Description: Test basic function of vit convert_weight.
    Expectation: success
    """
    model_dict = ModelDict()
    model_dict_ = dict()
    model_dict_["cls_token"] = torch.tensor([1])
    model_dict_["norm"] = torch.tensor([2])
    model_dict_["mlp.fc1.weight"] = torch.tensor([3])
    model_dict_["attn.proj.weight"] = torch.tensor([4])
    model_dict_["fc_norm.bias"] = torch.tensor([6])
    model_dict_["norm.weight"] = torch.tensor([7])
    model_dict_["mlp.fc2.weight"] = torch.tensor([8])
    model_dict.state_dict()["model"] = model_dict_

    mock_get.return_value = model_dict.state_dict()
    vit_convert_pt_to_ms(path, os.path.join(path, "test_vit.ckpt"))

    test_dict = ms.load_checkpoint(os.path.join(path, "test_vit.ckpt"))
    assert test_dict is not None

    test_tensor = test_dict["vit.cls_tokens"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [1]
    test_tensor = test_dict["vit.layernorm"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [2]
    test_tensor = test_dict["vit.output.mapping.weight"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [3]
    test_tensor = test_dict["vit.attention.projection.weight"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [4]
    test_tensor = test_dict["vit.fc_norm.beta"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [6]
    test_tensor = test_dict["vit.layernorm.gamma"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [7]
    test_tensor = test_dict["vit.output.projection.weight"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [8]

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("torch.load")
def test_clip_convert_pt_to_ms(mock_get):
    """
    Feature: clip convert_weight.
    Description: Test basic function of clip convert_weight.
    Expectation: success
    """
    model_dict = ModelDict()
    model_dict.state_dict()["ln_pre.weight"] = torch.tensor([1])
    model_dict.state_dict()["ln_pre.bias"] = torch.tensor([2])
    model_dict.state_dict()["in_proj_weight"] = torch.tensor([3])
    model_dict.state_dict()["in_proj_bias"] = torch.tensor([4])
    model_dict.state_dict()["token_embedding.weight"] = torch.tensor([5])
    model_dict.state_dict()["test"] = torch.tensor([6])
    mock_get.return_value = model_dict.state_dict()

    clip_convert_pt_to_ms(path, os.path.join(path, "test_clip.ckpt"))
    test_dict = ms.load_checkpoint(os.path.join(path, "test_clip.ckpt"))
    assert test_dict is not None

    test_tensor = test_dict["ln_pre.gamma"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [1]
    test_tensor = test_dict["ln_pre.beta"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [2]
    test_tensor = test_dict["in_proj.weight"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [3]
    test_tensor = test_dict["in_proj.bias"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [4]
    test_tensor = test_dict["token_embedding.embedding_table"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [5]
    test_tensor = test_dict["test"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [6]

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("transformers.AutoModel.from_pretrained")
def test_codegeex2_convert_pt_to_ms(mock_get):
    """
    Feature: codegeex2 convert_weight.
    Description: Test basic function of codegeex2 convert_weight.
    Expectation: success
    """
    model_dict = ModelDict()
    model_dict.state_dict()["word_embeddings.weight"] = torch.tensor([1])
    mock_get.return_value = model_dict

    codegeex2_convert_pt_to_ms(path, os.path.join(path, "test_codegeex2.ckpt"))
    test_dict = ms.load_checkpoint(os.path.join(path, "test_codegeex2.ckpt"))
    assert test_dict is not None
    test_tensor = test_dict["embedding_table"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [1]

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("mindspore.load_checkpoint")
def test_llama_convert_to_new_ckpt(mock_get):
    """
    Feature: llama convert_weight.convert_to_new_ckpt.
    Description: Test basic function of llama convert_weight.convert_to_new_ckpt.
    Expectation: success
    """
    data = {
        "n_heads": 1,
        "dim": 2
    }
    with open(os.path.join(path, "test_config.json"), "w") as file:
        json.dump(data, file)
    model_dict = ModelDict()
    model_dict.state_dict()["model.layers.0.attention.wq.weight"] = ms.Parameter([[[[1, 2, 3, 4]]]])
    mock_get.return_value = model_dict.state_dict()
    llama_convert_to_new_ckpt(path, os.path.join(path, "test_config.json"))
    test_dict = ms.load_checkpoint(os.path.join(path, "_hf"))
    assert test_dict is not None
    test_tensor = test_dict["model.layers.0.attention.wq.weight"]
    assert test_tensor is not None

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("torch.load")
def test_llama_convert_meta_torch_ckpt(mock_get):
    """
    Feature: llama convert_weight.convert_meta_torch_ckpt.
    Description: Test basic function of llama convert_weight.convert_meta_torch_ckpt.
    Expectation: success
    """
    model_dict = ModelDict()
    data = {
        "n_heads": 1,
        "dim": 2
    }
    with open(os.path.join(path, "params.json"), "w") as file:
        json.dump(data, file)
    model_dict.state_dict()["norm.weight"] = torch.tensor([1])
    model_dict.state_dict()["output.weight"] = torch.tensor([2])
    model_dict.state_dict()["model.weight.rope.freqs"] = torch.tensor([3])
    mock_get.return_value = model_dict.state_dict()
    assert llama_convert_meta_torch_ckpt(path, os.path.join(path, "test_llama_convert_meta_torch_ckpt.ckpt")) is False

    torch.save(model_dict.state_dict(), os.path.join(path, "test_model.pth"))
    llama_convert_meta_torch_ckpt(path, os.path.join(path, "test_llama_convert_meta_torch_ckpt.ckpt"))
    test_dict = ms.load_checkpoint(os.path.join(path, "test_llama_convert_meta_torch_ckpt.ckpt"))
    assert test_dict is not None
    test_tensor = test_dict["model.norm_out.weight"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [1]
    test_tensor = test_dict["lm_head.weight"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [2]

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("torch.load")
def test_llama_convert_megatron_to_ms(mock_get):
    """
    Feature: llama convert_weight.convert_megatron_to_ms.
    Description: Test basic function of llama convert_weight.convert_megatron_to_ms.
    Expectation: success
    """
    model_dict = ModelDict()
    model_dict.state_dict()["model"] = dict()
    model_dict.state_dict()["model"]["test_extra_state"] = torch.tensor([1])
    model_dict.state_dict()["model"]["dense_h_to_4h"] = torch.tensor([2])
    mock_get.return_value = model_dict.state_dict()
    llama_convert_megatron_to_ms(path, os.path.join(path, "test_llama_convert_megatron_to_ms.ckpt"))
    test_dict = ms.load_checkpoint(os.path.join(path, "test_llama_convert_megatron_to_ms.ckpt"))
    assert test_dict is not None


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("mindspore.load_checkpoint")
def test_llama_convert_to_qkv_concat(mock_get):
    """
    Feature: llama convert_weight.convert_to_qkv_concat.
    Description: Test basic function of llama convert_weight.convert_to_qkv_concat.
    Expectation: success
    """
    model_dict = ModelDict()

    model_dict.state_dict()["model.layers.0.attention.wq.weight"] = ms.Parameter([1])
    model_dict.state_dict()["model.layers.0.attention.wk.weight"] = ms.Parameter([2])
    model_dict.state_dict()["model.layers.0.attention.wv.weight"] = ms.Parameter([3])
    model_dict.state_dict()["model.layers.0.attention.w_qkv.weight"] = ms.Parameter([4])
    model_dict.state_dict()["model.layers.0.feed_forward.w1.weight"] = ms.Parameter([5])
    model_dict.state_dict()["model.layers.0.feed_forward.w3.weight"] = ms.Parameter([6])
    model_dict.state_dict()["model.layers.0.feed_forward.w_gate_hidden.weight"] = ms.Parameter([7])
    model_dict.state_dict()["model.layers.0.attention.wq.bias"] = ms.Parameter([8])
    model_dict.state_dict()["model.layers.0.attention.wk.bias"] = ms.Parameter([9])
    model_dict.state_dict()["model.layers.0.attention.wv.bias"] = ms.Parameter([10])
    model_dict.state_dict()["model.layers.0.attention.w_qkv.bias"] = ms.Parameter([11])
    mock_get.return_value = model_dict.state_dict()

    llama_convert_to_qkv_concat(os.path.join(path, "test"), os.path.join(path, "test_llama_convert_to_qkv_concat.ckpt"))
    test_dict = ms.load_checkpoint(os.path.join(path, "test_llama_convert_to_qkv_concat.ckpt"))
    assert test_dict is not None

    test_tensor = test_dict["model.layers.0.attention.w_qkv.weight"]
    assert test_tensor is not None

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("transformers.LlamaForCausalLM.from_pretrained")
def test_llama_convert_pt_to_ms(mock_get):
    """
    Feature: llama convert_weight.convert_pt_to_ms.
    Description: Test basic function of llama convert_weight.convert_pt_to_ms.
    Expectation: success
    """
    model_dict = ModelDict()
    model_dict.state_dict()["norm.weight"] = torch.tensor([0])
    model_dict.state_dict()["layers.test"] = torch.tensor([1])
    mock_get.return_value = model_dict

    llama_convert_pt_to_ms(path, os.path.join(path, "test_llama_.ckpt"))
    test_dict = ms.load_checkpoint(os.path.join(path, "test_llama_.ckpt"))
    assert test_dict is not None
    test_tensor = test_dict["norm_out.weight"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [0]
    test_tensor = test_dict["test"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [1]

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("torch.load")
def test_blip2_convert_pt_to_ms(mock_get):
    """
    Feature: blip2 convert_weight.
    Description: Test basic function of blip2 convert_weight.
    Expectation: success
    """
    model_dict = ModelDict()
    model = dict()
    model["attn.qkv.weight"] = torch.tensor([1, 2, 3])
    model_dict.state_dict()["model"] = model
    mock_get.return_value = model_dict.state_dict()

    blip2_convert_pt_to_ms(path, os.path.join(path, "test_blip2.ckpt"))
    test_dict = ms.load_checkpoint(os.path.join(path, "test_blip2.ckpt"))
    assert test_dict is not None
    test_tensor = test_dict["attention.dense1.weight"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [1]
    test_tensor = test_dict["attention.dense2.weight"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [2]
    test_tensor = test_dict["attention.dense3.weight"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [3]

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("torch.load")
def test_bloom_convert_pt_to_ms(mock_get):
    """
    Feature: bloom convert_weight.
    Description: Test basic function of bloom convert_weight.
    Expectation: success
    """
    model_dict = ModelDict()
    model_dict.state_dict()["transformer.lm_head.bias"] = torch.tensor([1])
    model_dict.state_dict()["weight.1.self_attention.query_key_value.bias"] = torch.tensor([1, 2, 3])
    model_dict.state_dict()["w.2.self_attention.query_key_value.bias"] = torch.tensor([1, 2, 3])
    mock_get.return_value = model_dict.state_dict()
    bloom_convert_pt_to_ms(path, os.path.join(path, "test_bloom.ckpt"), n_head=1, hidden_size=1)
    test_dict = ms.load_checkpoint(os.path.join(path, "test_bloom.ckpt"))
    assert test_dict is not None
    test_tensor = test_dict["transformer.head.bias"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [1]

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("torch.load")
def test_swin_convert_pt_to_ms(mock_get):
    """
    Feature: swin convert_weight.
    Description: Test basic function of swin convert_weight.
    Expectation: success
    """
    model_dict = ModelDict()
    model = dict()
    model["decoder.0.patch_embed.relative_position.norm.weight.bias.mlp.fc1"] = torch.tensor([1])
    model["weight.mlp.fc2.qkv"] = torch.tensor([2])

    model_dict.state_dict()["model"] = model
    mock_get.return_value = model_dict.state_dict()
    swin_convert_pt_to_ms(path, os.path.join(path, "test_swin.ckpt"), is_pretrain=True)
    test_dict = ms.load_checkpoint(os.path.join(path, "test_swin.ckpt"))
    assert test_dict is not None
    test_tensor = test_dict["decoder.patch_embed.relative_position_bias.relative_position.norm.gamma.beta.mlp.mapping"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [1]
    test_tensor = test_dict["weight.mlp.projection.v"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [2]

    model_dict_ = ModelDict()
    model_ = dict()
    model_["head.patch_embed.relative_position.norm.weight.bias.mlp.fc1"] = torch.tensor([1])
    model_["weight.mlp.fc2.qkv"] = torch.tensor([2])
    model_dict_.state_dict()["model"] = model_
    mock_get.return_value = model_dict_.state_dict()
    swin_convert_pt_to_ms(path, os.path.join(path, "test_swin_.ckpt"), is_pretrain=False)
    test_dict = ms.load_checkpoint(os.path.join(path, "test_swin_.ckpt"))
    assert test_dict is not None
    test_tensor = test_dict["head.patch_embed.relative_position.norm.weight.bias.mlp.fc1"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [1]
    test_tensor = test_dict["encoder.weight.mlp.projection.v"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [2]

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cogvlm2_convert_pt_to_ms():
    """
    Feature: cogvlm2 convert_weight.
    Description: Test basic function of cogvlm2 convert_weight.
    Expectation: success
    """
    model_dict = dict()
    model_dict["test"] = torch.tensor([0])
    model_dict["self_attn.rotary_emb.inv_freq"] = torch.tensor([1])
    model_dict["model.vision.patch_embedding"] = torch.tensor([2])
    model_dict["model.vision"] = torch.tensor([3])
    model_dict["model.norm.weight"] = torch.tensor([4])
    model_dict["lm_head.weight"] = torch.tensor([5])
    model_dict["model.embed_tokens"] = torch.tensor([6])
    save_file(model_dict, os.path.join(path, "test.safetensors"))
    cogvlm2_convert_pt_to_ms(path, os.path.join(path, "test_cogvlm2.ckpt"))

    test_dict = ms.load_checkpoint(os.path.join(path, "test_cogvlm2.ckpt"))
    assert test_dict is not None

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("torch.load")
def test_gpt2_convert_pt_to_ms(mock_get):
    """
    Feature: gpt2 convert_weight.
    Description: Test basic function of gpt2 convert_weight.
    Expectation: success
    """
    model_dict = ModelDict()
    model_dict.state_dict()["attn.c_attn.weight"] = torch.tensor([0, 1, 2])
    model_dict.state_dict()["h.0.ln_1.weight"] = torch.tensor([1])
    mock_get.return_value = model_dict.state_dict()
    gpt2_convert_pt_to_ms(path, os.path.join(path, "test_gpt2.ckpt"))

    test_dict = ms.load_checkpoint(os.path.join(path, "test_gpt2.ckpt"))
    assert test_dict is not None
    test_tensor = test_dict["backbone.blocks.0.layernorm1.gamma"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [1]

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bert_convert_weight():
    """
    Feature: bert convert_weight.
    Description: Test basic function of bert convert_weight.
    Expectation: success
    """
    model_dict = ModelDict()
    model_dict.state_dict()["attn.c_attn.weight"] = torch.tensor([0, 1, 2])
    model_dict.state_dict()["bert.encoder.layer.0.attention.self.query.weight"] = torch.tensor([1])
    model_dict.state_dict()["bert.embeddings.word_embeddings.weight"] = torch.tensor([2])
    bert_split_torch_attention(model_dict.state_dict())
    ms_name = [
        "bert.bert_encoder.encoder.blocks.{}.attention.dense1.weight"
    ]

    torch_name = [
        "bert.encoder.layer.{}.attention.self.query.weight"
    ]

    addition_mindspore = [
        "bert.word_embedding.embedding_table"
    ]

    addition_torch = [
        "bert.embeddings.word_embeddings.weight",
    ]
    mapped_param = bert_generate_params_dict(total_layers=1,
                                             mindspore_params_per_layer=ms_name,
                                             torch_params_per_layer=torch_name,
                                             mindspore_additional_params=addition_mindspore,
                                             torch_additional_params=addition_torch)
    new_ckpt = bert_get_converted_ckpt(mapped_param, model_dict.state_dict())
    save_checkpoint(new_ckpt, os.path.join(path, "test_bert.ckpt"))
    test_dict = ms.load_checkpoint(os.path.join(path, "test_bert.ckpt"))
    assert test_dict is not None
    test_tensor = test_dict["bert.bert_encoder.encoder.blocks.0.attention.dense1.weight"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [1]
    test_tensor = test_dict["bert.word_embedding.embedding_table"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [2]

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_t5_convert_weight():
    """
    Feature: t5 convert_weight.
    Description: Test basic function of t5 convert_weight.
    Expectation: success
    """
    ms_name = [
        "t5_model.tfm_encoder.blocks.{}.attention.dense3.weight"
    ]

    torch_name = [
        "encoder.block.{}.layer.0.SelfAttention.o.weight"
    ]

    addition_mindspore = [
        "t5_model.encoder_layernorm.gamma"
    ]

    addition_torch = [
        "encoder.final_layer_norm.weight"
    ]
    mapped_param = t5_generate_params_dict(total_layers=1,
                                           mindspore_params_per_layer=ms_name,
                                           torch_params_per_layer=torch_name,
                                           mindspore_additional_params=addition_mindspore,
                                           torch_additional_params=addition_torch)
    model_dict = ModelDict()
    model_dict.state_dict()["encoder.block.0.layer.0.SelfAttention.o.weight"] = torch.tensor([[1], [2]])
    model_dict.state_dict()["encoder.final_layer_norm.weight"] = torch.tensor([0])
    new_ckpt = t5_get_converted_ckpt(mapped_param, model_dict.state_dict())
    save_checkpoint(new_ckpt, os.path.join(path, "test_t5.ckpt"))
    test_dict = ms.load_checkpoint(os.path.join(path, "test_t5.ckpt"))
    assert test_dict is not None
    test_tensor = test_dict["t5_model.tfm_encoder.blocks.0.attention.dense3.weight"]
    assert test_tensor is not None
    test_tensor = test_dict["t5_model.encoder_layernorm.gamma"]
    assert test_tensor is not None
    assert test_tensor.asnumpy() == [0]
