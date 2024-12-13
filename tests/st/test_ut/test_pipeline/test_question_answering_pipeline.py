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
Test module for testing pipeline function.
How to run this:
pytest tests/st/test_pipeline/test_pipeline.py
"""
import os
import yaml
import pytest

import mindspore as ms


from mindformers import AutoTokenizer, BertForQuestionAnswering
from mindformers.pipeline import QuestionAnsweringPipeline
from mindformers import BertConfig

ms.set_context(mode=0)

class Result:
    """A mock class for testing postprocess."""
    def __init__(self):
        self.unique_id = "test6699"
        self.start_logits = {0: 0.0}
        self.end_logits = {0: 0.0}


class MyDict(dict):
    """A mock dict class for testing postprocess."""
    def __init__(self, *args, **kwargs):
        super(MyDict, self).__init__(*args, **kwargs)
        self.example_index = 0
        self.unique_id = "test6699"
        self.tokens = ["test"]
        self.token_to_orig_map = {0: 0.0}
        self.token_is_max_context = {0: "test"}

    def add_item(self, key, value):
        self[key] = value

def mock_yaml():
    """A mock yaml function for testing question_answering_pipeline."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(cur_dir.split("tests")[0], "configs", "qa")
    os.makedirs(yaml_path, exist_ok=True)
    bert_yaml_path = os.path.join(yaml_path, "run_qa_bert_base_uncased.yaml")
    useless_names = ["_name_or_path", "tokenizer_class", "architectures", "is_encoder_decoder",
                     "is_sample_acceleration", "parallel_config", "moe_config"]
    bert_ori_config = BertConfig().to_dict()
    for name in useless_names:
        bert_ori_config.pop(name, None)
    bert_ori_config["num_layers"] = 1
    bert_ori_config["type"] = "BertConfig"
    bert_config = {"model": {"arch": {"type": "BertForQuestionAnswering"}, "model_config": bert_ori_config},
                   "processor": {"return_tensors": "ms",
                                 "tokenizer": {
                                     "cls_token": "[CLS]",
                                     "mask_token": "[MASK]",
                                     "unk_token": "[UNK]",
                                     "pad_token": "[PAD]",
                                     "sep_token": "[SEP]",
                                     "do_basic_tokenize": True,
                                     "do_lower_case": True,
                                     "type": "BertTokenizer"
                                 },
                                 "type": "BertProcessor"
                                 }
                   }
    bert_config["model"]["model_config"]["checkpoint_name_or_path"] = "qa_bert_base_uncased_squad"
    with open(bert_yaml_path, "w", encoding="utf-8") as w:
        yaml.dump(bert_config, w, default_flow_style=False)
    with open("checkpoint_download/qa/vocab.txt", "w") as file:
        file.write("test")

mock_yaml()
tokenizer = AutoTokenizer.from_pretrained('qa_bert_base_uncased_squad')
qa_squad_config = BertConfig(seq_length=384)
model = BertForQuestionAnswering(qa_squad_config)
qa_pipeline = QuestionAnsweringPipeline(task='question_answering',
                                        model=model,
                                        tokenizer=tokenizer)
@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_question_answering_pipeline():
    """
    Feature: question_answering_pipeline interface.
    Description: Test basic function of question_answering_pipeline api.
    Expectation: success
    """
    input_data = ["My name is Wolfgang and I live in llama2_7b - Where do I live?"]

    output = qa_pipeline(input_data)
    assert isinstance(output[0], dict)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_postprocess():
    """
    Feature: postprocess interface.
    Description: Test basic function of postprocess api.
    Expectation: success
    """
    result = Result()
    model_outputs = {result: "test"}
    my_dict = MyDict()
    my_dict.add_item("test", "tset")
    qa_pipeline.features = [my_dict]
    qa_pipeline.examples = [my_dict]
    output = qa_pipeline.postprocess(model_outputs=model_outputs, top_k=1, n_best_size=1, max_answer_len=128)
    assert isinstance(output[0], dict)
