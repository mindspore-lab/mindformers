# Copyright 2025 Huawei Technologies Co., Ltd
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
Test module for testing the /research/qwen2_5/convert_weight.py lora convert used for mindformers.
How to run this:
pytest tests/st/test_model/test_qwen2_5_model/test_convert_weight_lora.py
"""

import json
import os
import tempfile

import pytest

import numpy as np
import mindspore as ms
from safetensors.numpy import save_file
from research.qwen2_5.convert_weight import convert_lora_to_ms

class TestLoraConvert:
    """A test class for testing convert lora safetensors."""
    def setup_class(self):
        """set fake lora attention config"""

        # create fake safetensors directory
        self.work_dir = tempfile.mkdtemp()
        self.src_path = os.path.join(self.work_dir, "torch_ckpt_dir")
        os.makedirs(self.src_path)
        self.dst_path = os.path.join(self.work_dir, "mindspore_ckpt_path")
        os.makedirs(self.dst_path)

        # create fake weights value
        self.rank = 8
        self.hidden_size = 3584
        self.intermediate_size = 18944
        # wq
        self.fake_weight1_lora_a = np.random.rand(self.rank, self.hidden_size).astype(np.float16)
        self.fake_weight1_lora_b = np.random.rand(self.hidden_size, self.rank).astype(np.float16)
        # wk
        self.fake_weight2_lora_a = np.random.rand(self.rank, self.hidden_size).astype(np.float16)
        self.fake_weight2_lora_b = np.random.rand(512, self.rank).astype(np.float16)
        # wv
        self.fake_weight3_lora_a = np.random.rand(self.rank, self.hidden_size).astype(np.float16)
        self.fake_weight3_lora_b = np.random.rand(512, self.rank).astype(np.float16)
        # wo
        self.fake_weight4_lora_a = np.random.rand(self.rank, self.hidden_size).astype(np.float16)
        self.fake_weight4_lora_b = np.random.rand(self.hidden_size, self.rank).astype(np.float16)
        # w1
        self.fake_weight5_lora_a = np.random.rand(self.rank, self.hidden_size).astype(np.float16)
        self.fake_weight5_lora_b = np.random.rand(self.intermediate_size, self.rank).astype(np.float16)
        # w2
        self.fake_weight6_lora_a = np.random.rand(self.rank, self.intermediate_size).astype(np.float16)
        self.fake_weight6_lora_b = np.random.rand(self.hidden_size, self.rank).astype(np.float16)
        # w3
        self.fake_weight7_lora_a = np.random.rand(self.hidden_size, self.hidden_size).astype(np.float16)
        self.fake_weight7_lora_b = np.random.rand(self.intermediate_size, self.rank).astype(np.float16)

        # prepare fake weight dict and map
        fake_adapter_model_dict = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": self.fake_weight1_lora_a,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": self.fake_weight1_lora_b,
            "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight": self.fake_weight2_lora_a,
            "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight": self.fake_weight2_lora_b,
            "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": self.fake_weight3_lora_a,
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": self.fake_weight3_lora_b,
            "base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight": self.fake_weight4_lora_a,
            "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight": self.fake_weight4_lora_b,
            "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight": self.fake_weight5_lora_a,
            "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight": self.fake_weight5_lora_b,
            "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": self.fake_weight6_lora_a,
            "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": self.fake_weight6_lora_b,
            "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight": self.fake_weight7_lora_a,
            "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight": self.fake_weight7_lora_b
        }
        fake_adapter_config_dict = {
            "target_modules": [
                "up_proj",
                "gate_proj",
                "k_proj",
                "q_proj",
                "down_proj",
                "o_proj",
                "v_proj"
            ],
        }
        fake_adapter_model_file = os.path.join(self.src_path, "adapter_model.safetensors")

        # make fake safetensors file
        save_file(tensor_dict=fake_adapter_model_dict, filename=fake_adapter_model_file)
        os.chmod(fake_adapter_model_file, 0o750)

        # make fake json file
        adapter_config_file = os.path.join(self.src_path, "adapter_config.json")
        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(adapter_config_file, flags_, 0o750), 'w') as f:
            json.dump(fake_adapter_config_dict, f)

    @pytest.mark.run(order=1)
    def test_convert_lora_to_ms(self):
        """test convert unified HuggingFace lora safetensors to MindSpore lora safetensors"""
        adapter_mopdel_file = os.path.join(self.dst_path, "adapter_model.ckpt")
        adapter_config_file = os.path.join(self.src_path, "adapter_config.json")
        convert_lora_to_ms(self.src_path, adapter_mopdel_file, dtype=ms.float16, align_rank=True)

        # check whether safetensors and map file exists
        assert os.path.exists(adapter_mopdel_file), \
            "adapter_model.safetensors does not exist, conversion not completed."
        assert os.path.exists(adapter_config_file), \
            "adapter_config.json does not exist, conversion not completed."

        # check whether the weight in the safetensors file is correct
        adapter_model_dict = ms.load_checkpoint(adapter_mopdel_file)
        # assert wq
        assert "model.layers.0.attention.wq.lora_a" in adapter_model_dict.keys(), \
            "adapter_model.safetensors does not have key 'model.layers.0.attention.wq.lora_a'."
        assert np.array_equal(adapter_model_dict.get("model.layers.0.attention.wq.lora_a").value(), \
                              self.fake_weight1_lora_a), \
            f"The value of 'model.layers.0.attention.wq.lora_a' should be {self.fake_weight1_lora_a}, " \
            f"but got {adapter_model_dict.get('model.layers.0.attention.wq.lora_a')}."
        assert "model.layers.0.attention.wq.lora_b" in adapter_model_dict.keys(), \
            "adapter_model.safetensors does not have key 'model.layers.0.attention.wq.lora_b'."
        assert np.array_equal(adapter_model_dict.get("model.layers.0.attention.wq.lora_b").value(), \
                              self.fake_weight1_lora_b), \
            f"The value of 'model.layers.0.attention.wq.lora_b' should be {self.fake_weight1_lora_b}, " \
            f"but got {adapter_model_dict.get('model.layers.0.attention.wq.lora_b')}."

        # assert wk
        assert "model.layers.0.attention.wk.lora_a" in adapter_model_dict.keys(), \
            "adapter_model.safetensors does not have key 'model.layers.0.attention.wk.lora_a'."
        assert np.array_equal(adapter_model_dict.get("model.layers.0.attention.wk.lora_a").value(), \
                              self.fake_weight2_lora_a), \
            f"The value of 'model.layers.0.attention.wk.lora_a' should be {self.fake_weight2_lora_a}, " \
            f"but got {adapter_model_dict.get('model.layers.0.attention.wk.lora_a')}."
        assert "model.layers.0.attention.wk.lora_b" in adapter_model_dict.keys(), \
            "adapter_model.safetensors does not have key 'model.layers.0.attention.wk.lora_b'."
        assert np.array_equal(adapter_model_dict.get("model.layers.0.attention.wk.lora_b").value(), \
                              self.fake_weight2_lora_b), \
            f"The value of 'model.layers.0.attention.wk.lora_b' should be {self.fake_weight2_lora_b}, " \
            f"but got {adapter_model_dict.get('model.layers.0.attention.wk.lora_b')}."

        # assert wv
        assert "model.layers.0.attention.wv.lora_a" in adapter_model_dict.keys(), \
            "adapter_model.safetensors does not have key 'model.layers.0.attention.wv.lora_a'."
        assert np.array_equal(adapter_model_dict.get("model.layers.0.attention.wv.lora_a").value(), \
                              self.fake_weight3_lora_a), \
            f"The value of 'model.layers.0.attention.wv.lora_a' should be {self.fake_weight3_lora_a}, " \
            f"but got {adapter_model_dict.get('model.layers.0.attention.wv.lora_a')}."
        assert "model.layers.0.attention.wv.lora_b" in adapter_model_dict.keys(), \
            "adapter_model.safetensors does not have key 'model.layers.0.attention.wv.lora_b'."
        assert np.array_equal(adapter_model_dict.get("model.layers.0.attention.wv.lora_b").value(), \
                              self.fake_weight3_lora_b), \
            f"The value of 'model.layers.0.attention.wv.lora_b' should be {self.fake_weight3_lora_b}, " \
            f"but got {adapter_model_dict.get('model.layers.0.attention.wv.lora_b')}."

        # assert wo
        assert "model.layers.0.attention.wo.lora_a" in adapter_model_dict.keys(), \
            "adapter_model.safetensors does not have key 'model.layers.0.attention.wo.lora_a'."
        assert np.array_equal(adapter_model_dict.get("model.layers.0.attention.wo.lora_a").value(), \
                              self.fake_weight4_lora_a), \
            f"The value of 'model.layers.0.attention.wo.lora_a' should be {self.fake_weight4_lora_a}, " \
            f"but got {adapter_model_dict.get('model.layers.0.attention.wo.lora_a')}."
        assert "model.layers.0.attention.wo.lora_b" in adapter_model_dict.keys(), \
            "adapter_model.safetensors does not have key 'model.layers.0.attention.wo.lora_b'."
        assert np.array_equal(adapter_model_dict.get("model.layers.0.attention.wo.lora_b").value(), \
                              self.fake_weight4_lora_b), \
            f"The value of 'model.layers.0.attention.wo.lora_b' should be {self.fake_weight4_lora_b}, " \
            f"but got {adapter_model_dict.get('model.layers.0.attention.wo.lora_b')}."

        # assert w1
        assert "model.layers.0.feed_forward.w1.lora_a" in adapter_model_dict.keys(), \
            "adapter_model.safetensors does not have key 'model.layers.0.feed_forward.w1.lora_a'."
        assert np.array_equal(adapter_model_dict.get("model.layers.0.feed_forward.w1.lora_a").value(), \
                              self.fake_weight5_lora_a), \
            f"The value of 'model.layers.0.feed_forward.w1.lora_a' should be {self.fake_weight5_lora_a}, " \
            f"but got {adapter_model_dict.get('model.layers.0.feed_forward.w1.lora_a')}."
        assert "model.layers.0.feed_forward.w1.lora_b" in adapter_model_dict.keys(), \
            "adapter_model.safetensors does not have key 'model.layers.0.feed_forward.w1.lora_b'."
        assert np.array_equal(adapter_model_dict.get("model.layers.0.feed_forward.w1.lora_b").value(), \
                              self.fake_weight5_lora_b), \
            f"The value of 'model.layers.0.feed_forward.w1.lora_b' should be {self.fake_weight5_lora_b}, " \
            f"but got {adapter_model_dict.get('model.layers.0.feed_forward.w1.lora_b')}."

        # assert w2
        assert "model.layers.0.feed_forward.w2.lora_a" in adapter_model_dict.keys(), \
            "adapter_model.safetensors does not have key 'model.layers.0.feed_forward.w2.lora_a'."
        assert np.array_equal(adapter_model_dict.get("model.layers.0.feed_forward.w2.lora_a").value(), \
                              self.fake_weight6_lora_a), \
            f"The value of 'model.layers.0.feed_forward.w2.lora_a' should be {self.fake_weight6_lora_a}, " \
            f"but got {adapter_model_dict.get('model.layers.0.feed_forward.w2.lora_a')}."
        assert "model.layers.0.feed_forward.w2.lora_b" in adapter_model_dict.keys(), \
            "adapter_model.safetensors does not have key 'model.layers.0.feed_forward.w2.lora_b'."
        assert np.array_equal(adapter_model_dict.get("model.layers.0.feed_forward.w2.lora_b").value(), \
                              self.fake_weight6_lora_b), \
            f"The value of 'model.layers.0.feed_forward.w2.lora_b' should be {self.fake_weight6_lora_b}, " \
            f"but got {adapter_model_dict.get('model.layers.0.feed_forward.w2.lora_b')}."

        # assert w3
        assert "model.layers.0.feed_forward.w3.lora_a" in adapter_model_dict.keys(), \
            "adapter_model.safetensors does not have key 'model.layers.0.feed_forward.w3.lora_a'."
        assert np.array_equal(adapter_model_dict.get("model.layers.0.feed_forward.w3.lora_a").value(), \
                              self.fake_weight7_lora_a), \
            f"The value of 'model.layers.0.feed_forward.w3.lora_a' should be {self.fake_weight7_lora_a}, " \
            f"but got {adapter_model_dict.get('model.layers.0.feed_forward.w3.lora_a')}."
        assert "model.layers.0.feed_forward.w3.lora_b" in adapter_model_dict.keys(), \
            "adapter_model.safetensors does not have key 'model.layers.0.feed_forward.w3.lora_b'."
        assert np.array_equal(adapter_model_dict.get("model.layers.0.feed_forward.w3.lora_b").value(), \
                              self.fake_weight7_lora_b), \
            f"The value of 'model.layers.0.feed_forward.w3.lora_b' should be {self.fake_weight7_lora_b}, " \
            f"but got {adapter_model_dict.get('model.layers.0.feed_forward.w3.lora_b')}."

        with open(adapter_config_file, 'r', encoding='utf-8') as f:
            adapter_config_dict = json.load(f)
        # assert target_modules
        assert "target_modules" in adapter_config_dict.keys(), \
            "adapter_config.json does not have key 'target_modules'."
        target_modules = adapter_config_dict.get("target_modules")
        expected_modules = {"w3", "w1", "wk", "wq", "w2", "wo", "wv"}
        assert isinstance(target_modules, list), f"'target_modules' should be a list, but got {type(target_modules)}."
        assert set(target_modules) == expected_modules, \
            f"The value of 'target_modules' should contain exactly {expected_modules}, but got {target_modules}."
