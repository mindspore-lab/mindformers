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
"""test sft map functions"""
import os
import copy
import tempfile
import pytest
from mindformers import LlamaTokenizer
from mindformers.dataset.dataloader.sft_map_functions import \
    (
        _prepare_for_model,
        default_map_fn,
        alpaca_map_fn,
        advertisegen_map_fn,
        cola_map_fn,
        imdb_map_fn,
        sst2_map_fn,
        agnwes_map_fn,
        tnews_map_fn,
        multi_round_chat_map_fn,
        multi_instruct_dyn_map_fn,
        multi_round_chat_dyn_map_fn,
        multi_round_chat_dyn_map_fn_alpaca,
    )
from tests.st.test_ut.test_tokenizers.get_vocab_model import get_sp_vocab_model


string = "An increasing sequence: one, two, three."
temp_dir = tempfile.TemporaryDirectory()
path = temp_dir.name
get_sp_vocab_model("llama", path)
tokenizer_model_path = os.path.join(path, "llama_tokenizer.model")
tokenizer = LlamaTokenizer(vocab_file=tokenizer_model_path)


class Example:
    """mock example"""
    def __init__(self):
        self.string = string
        self.label = "100"
        self.usr_def = {"input": string, "instruction": string, "output": ""}

    def get(self, input_str):
        if input_str == "label":
            return self.label
        if input_str in ["summary"]:
            return ""
        if input_str == "conversations":
            return self.usr_def
        return string

    def values(self):
        return self.string

    def __getitem__(self, item):
        return [self.get(item)]


example = Example()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_prepare_for_model():
    """
    Feature: sft_map_functions._prepare_for_model
    Description: test _prepare_for_model function
    Expectation: success
    """
    res = _prepare_for_model(tokenizer=tokenizer, max_length=15, prompt=string)
    assert hasattr(res, "data")
    assert res.data["input_ids"] == [1, 48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 0, 0, 0, 0]
    assert res.data["attention_mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_default_map_fn():
    """
    Feature: sft_map_functions.default_map_fn
    Description: test default_map_fn function
    Expectation: success
    """
    tmp_example = copy.deepcopy(example)
    tmp_example.string = [string]
    res1 = default_map_fn(tmp_example, tokenizer=tokenizer, max_length=15)
    tmp_example.string = [string, ""]
    res2 = default_map_fn(tmp_example, tokenizer=tokenizer, max_length=15)
    assert res1 == res2
    assert res1["input_ids"] == [1, 48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 0, 0, 0, 0]
    assert res1["attention_mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_alpaca_map_fn():
    """
    Feature: sft_map_functions.alpaca_map_fn
    Description: test alpaca_map_fn function
    Expectation: success
    """
    res = alpaca_map_fn(example.usr_def, tokenizer=tokenizer, max_length=15)
    assert res["input_ids"] == [1, 134, 0, 136, 0, 142, 165, 113, 140, 134, 159, 137, 50, 140, 144]
    assert res["attention_mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    usr_def = {"input": "", "instruction": string, "output": ""}
    res = alpaca_map_fn(usr_def, tokenizer=tokenizer, max_length=15)
    assert res["input_ids"] == [1, 134, 0, 136, 0, 142, 165, 113, 140, 134, 159, 137, 50, 140, 144]
    assert res["attention_mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_advertisegen_map_fn():
    """
    Feature: sft_map_functions.advertisegen_map_fn
    Description: test advertisegen_map_fn function
    Expectation: success
    """
    res = advertisegen_map_fn(example, tokenizer=tokenizer, max_length=15)
    assert res["input_ids"] == [1, 48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 0, 0, 0, 0]
    assert res["attention_mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cola_map_fn():
    """
    Feature: sft_map_functions.cola_map_fn
    Description: test cola_map_fn function
    Expectation: success
    """
    tmp_example = copy.deepcopy(example)
    tmp_example.string = ["", "100", "", string]
    res = cola_map_fn(tmp_example, tokenizer=tokenizer, max_length=15)
    assert res["input_ids"] == [1, 48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 0, 0, 0, 0]
    assert res["attention_mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    assert res["labels"] == "100"


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_imdb_map_fn():
    """
    Feature: sft_map_functions.imdb_map_fn
    Description: test imdb_map_fn function
    Expectation: success
    """
    res = imdb_map_fn(example, tokenizer=tokenizer, max_length=15)
    assert res["input_ids"] == [1, 48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 0, 0, 0, 0]
    assert res["attention_mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    assert res["labels"] == 0


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sst2_map_fn():
    """
    Feature: sft_map_functions.sst2_map_fn
    Description: test sst2_map_fn function
    Expectation: success
    """
    res = sst2_map_fn(example, tokenizer=tokenizer, max_length=15)
    assert res["input_ids"] == [1, 48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 0, 0, 0, 0]
    assert res["attention_mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    assert res["labels"] == "100"


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_agnwes_map_fn():
    """
    Feature: sft_map_functions.agnwes_map_fn
    Description: test agnwes_map_fn function
    Expectation: success
    """
    res = agnwes_map_fn(example, tokenizer=tokenizer, max_length=15)
    assert res["input_ids"] == [1, 48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 0, 0, 0, 0]
    assert res["attention_mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    assert res["labels"] == "100"


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tnews_map_fn():
    """
    Feature: sft_map_functions.tnews_map_fn
    Description: test tnews_map_fn function
    Expectation: success
    """
    res = tnews_map_fn(example, tokenizer=tokenizer, max_length=15)
    assert res["input_ids"] == [1, 48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 0, 0, 0, 0]
    assert res["attention_mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    assert res["labels"] == 0


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_multi_round_chat_map_fn():
    """
    Feature: sft_map_functions.multi_round_chat_map_fn
    Description: test multi_round_chat_map_fn function
    Expectation: success
    """
    tmp_example = copy.deepcopy(example)
    tmp_example.usr_def = {"from": "human", "value": string}
    res = multi_round_chat_map_fn(tmp_example, tokenizer=tokenizer, max_length=15)
    assert res["input_ids"] == [48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 2, 0, 0, 0, 0]
    assert res["attention_mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    assert res["labels"] == [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 2, -100, -100, -100, -100]
    tmp_example.usr_def = {"from": "gpt", "value": string}
    res = multi_round_chat_map_fn(tmp_example, tokenizer=tokenizer, max_length=15)
    assert res["input_ids"] == [48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 2, 0, 0, 0, 0]
    assert res["attention_mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    assert res["labels"] == [48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 2, -100, -100, -100, -100]
    res = multi_round_chat_map_fn(tmp_example, tokenizer=tokenizer, max_length=5)
    assert res["input_ids"] == [48, 87, 85, 157, 65]
    assert res["attention_mask"] == [1, 1, 1, 1, 1]
    assert res["labels"] == [48, 87, 85, 157, 65]
    tmp_example.usr_def = {"from": "mock", "value": string}
    with pytest.raises(ValueError):
        assert multi_round_chat_map_fn(tmp_example, tokenizer=tokenizer, max_length=15)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_multi_round_chat_dyn_map_fn():
    """
    Feature: sft_map_functions.multi_round_chat_dyn_map_fn
    Description: test multi_round_chat_dyn_map_fn function
    Expectation: success
    """
    tmp_example = copy.deepcopy(example)
    tmp_example.usr_def = {"from": "human", "value": string}
    res = multi_round_chat_dyn_map_fn(tmp_example, tokenizer=tokenizer, max_length=15)
    assert res["input_ids"] == [1, 48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 2]
    assert res["attention_mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert res["labels"] == [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 2]
    tmp_example.usr_def = {"from": "gpt", "value": string}
    res = multi_round_chat_dyn_map_fn(tmp_example, tokenizer=tokenizer, max_length=15)
    assert res["input_ids"] == [48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 2]
    assert res["attention_mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert res["labels"] == [48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 2]
    res = multi_round_chat_dyn_map_fn(tmp_example, tokenizer=tokenizer, max_length=5)
    assert res["input_ids"] == [48, 87, 85, 157, 65]
    assert res["attention_mask"] == [1, 1, 1, 1, 1]
    assert res["labels"] == [48, 87, 85, 157, 65]
    tmp_example.usr_def = {"from": "mock", "value": string}
    with pytest.raises(ValueError):
        assert multi_round_chat_dyn_map_fn(tmp_example, tokenizer=tokenizer, max_length=15)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_multi_instruct_dyn_map_fn():
    """
    Feature: sft_map_functions.multi_instruct_dyn_map_fn
    Description: test multi_instruct_dyn_map_fn function
    Expectation: success
    """
    tmp_example = copy.deepcopy(example)
    res = multi_instruct_dyn_map_fn(tmp_example.usr_def, tokenizer=tokenizer, max_length=15)
    assert res["input_ids"] == [48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 16, 87, 85, 157, 65]
    assert res["attention_mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert res["labels"] == [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
    res = multi_instruct_dyn_map_fn(tmp_example.usr_def, tokenizer=tokenizer, max_length=30)
    assert res["input_ids"] == \
           [48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 16, 87, 85, 157, 65, 135, 67, 135, 80, 150, 2]
    assert res["attention_mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert res["labels"] == [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                             -100, -100, -100, -100, -100, 2]


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_multi_round_chat_dyn_map_fn_alpaca():
    """
    Feature: sft_map_functions.multi_round_chat_dyn_map_fn_alpaca
    Description: test multi_round_chat_dyn_map_fn_alpaca function
    Expectation: success
    """
    tmp_example = copy.deepcopy(example)
    tmp_example.usr_def = {"from": "human", "value": string}
    res = multi_round_chat_dyn_map_fn_alpaca(tmp_example, tokenizer=tokenizer, max_length=15)
    assert res["input_ids"] == [134, 0, 139, 0, 140, 144, 159, 143, 144, 0, 139, 0, 6, 0, 134]
    assert res["labels"] == [134, 0, 139, 0, 140, 144, 159, 143, 144, 0, 139, 0, 6, 0, 134]
    tmp_example.usr_def = {"from": "gpt", "value": string}
    res = multi_round_chat_dyn_map_fn_alpaca(tmp_example, tokenizer=tokenizer, max_length=15)
    assert res["input_ids"] == [48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 0, 139, 0, 6, 0]
    assert res["labels"] == [48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 0, 139, 0, 6, 0]
    tmp_example.usr_def = {"from": "gpt", "value": string}
    res = multi_round_chat_dyn_map_fn_alpaca(tmp_example, tokenizer=tokenizer, max_length=55)
    assert res["input_ids"] == [48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 0, 139, 0, 6, 0, 2]
    assert res["labels"] == [48, 87, 85, 157, 65, 135, 67, 135, 80, 150, 0, 139, 0, 6, 0, 2]
    tmp_example.usr_def = {"from": "mock", "value": string}
    with pytest.raises(ValueError):
        assert multi_round_chat_dyn_map_fn_alpaca(tmp_example, tokenizer=tokenizer, max_length=15)
