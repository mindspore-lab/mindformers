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
Test Module for classification function of
TokenClassificationPipeline

How to run this:
windows:
pytest .\\tests\\st\\test_pipeline\\test_token_classification_pipeline.py
linux:
pytest ./tests/st/test_pipeline/test_token_classification_pipeline.py
"""
# import pytest

from mindformers.pipeline import TokenClassificationPipeline
from mindformers import AutoTokenizer, BertForTokenClassification, AutoConfig
from mindformers.dataset.labels import cluener_labels


# @pytest.mark.level0
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.env_onecard
def test_token_classification_pipeline():
    """
    Feature: TokenClassificationPipeline class
    Description: Test the pipeline functions
    Expectation: NotImplementedError, ValueError
    """

    input_data = ["表身刻有代表日内瓦钟表匠freresoltramare的“fo”字样。",
                  "结果上周六他们主场0：3惨败给了中游球队瓦拉多利德，近7个多月以来西甲首次输球。"]

    id2label = {label_id: label for label_id, label in enumerate(cluener_labels)}

    tokenizer = AutoTokenizer.from_pretrained('tokcls_bert_base_chinese_cluener')
    tokcls_cluener_config = AutoConfig.from_pretrained('tokcls_bert_base_chinese_cluener')

    # This is a known issue, you need to specify batch size equal to 1 when creating bert model.
    tokcls_cluener_config.batch_size = 1

    model = BertForTokenClassification(tokcls_cluener_config)
    tokcls_pipeline = TokenClassificationPipeline(task='token_classification',
                                                  model=model,
                                                  id2label=id2label,
                                                  tokenizer=tokenizer,
                                                  max_length=model.config.seq_length,
                                                  padding="max_length")

    results = tokcls_pipeline(input_data)
    assert isinstance(results, list)
    assert len(results) == 2
    assert isinstance(results[0], list)
    assert isinstance(results[0][0], dict)
