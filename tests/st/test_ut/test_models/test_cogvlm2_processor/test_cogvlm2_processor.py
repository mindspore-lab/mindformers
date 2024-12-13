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
"""test cogvlm2 processor."""
import unittest
import numpy as np
import pytest

import mindspore as ms
import mindspore.numpy as mnp

from mindformers import AutoTokenizer
from mindformers.models.cogvlm2.cogvlm2_processor import CogVLM2ImageProcessor, CogVLM2ContentTransformTemplate
from mindformers.models.multi_modal.utils import DataRecord


# pylint: disable=W0212
class TestCogvlm2ImageProcessor(unittest.TestCase):
    """ A test class for testing cogvlm2 processor."""

    @classmethod
    def setUpClass(cls):
        cls.processor = CogVLM2ImageProcessor()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_preprocess(self):
        images = np.ones((2, 224, 224, 3))
        res = self.processor.preprocess(images)
        res_list = res[0][0][0].asnumpy().tolist()
        assert res_list[0] == -1.777664065361023
        assert res_list[-1] == -1.777664065361023

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_bhwc_check(self):
        """test bhwc check."""
        image_batch = np.ones((2, 2, 2, 3))
        res = CogVLM2ImageProcessor._bhwc_check(image_batch)
        assert res
        image_batch = mnp.rand(224, 224, 3)
        res = CogVLM2ImageProcessor._bhwc_check(image_batch)
        assert res
        res = CogVLM2ImageProcessor._bhwc_check("not all")
        assert not res


class TestCogVLM2ContentTransformTemplate(unittest.TestCase):
    """ A test class for testing cogvlm2 content transform template."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("llama2_7b")
        cls.content_transform_template = CogVLM2ContentTransformTemplate(output_columns=None, tokenizer=cls.tokenizer)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_conversation_input_text(self):
        """test build conversion input text."""
        raw_inputs = ["How are you?", "I am fine."]
        res = self.content_transform_template.build_conversation_input_text(raw_inputs, result_recorder=DataRecord())
        assert res == "How are you?I am fine."
        self.content_transform_template.signal_type = "vqa"
        res = self.content_transform_template.build_conversation_input_text(raw_inputs, result_recorder=DataRecord())
        assert res == "Question: How are you?I am fine. Short answer:"
        self.content_transform_template.signal_type = "chat"
        res = self.content_transform_template.build_conversation_input_text(raw_inputs, result_recorder=DataRecord())
        assert res == "Question: How are you?I am fine. Answer:"
        raw_inputs = [
            ('login', 'user'),
            ('view', 'profile')
        ]
        self.content_transform_template.mode = "train"
        res = self.content_transform_template.build_conversation_input_text(raw_inputs, result_recorder=DataRecord())
        assert res[-1] == '<|end_of_text|>'
        self.content_transform_template.signal_type = "str"
        with pytest.raises(ValueError):
            assert self.content_transform_template.build_conversation_input_text(raw_inputs,
                                                                                 result_recorder=DataRecord())

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_need_update_output_items(self):
        """test get need update output items."""
        self.content_transform_template.mode = "predict"
        record = DataRecord()
        record.put('input_ids', [1, 2, 3])
        res = self.content_transform_template.get_need_update_output_items(record).get(
            'position_ids').asnumpy().tolist()
        assert res == [0, 1, 2]
        self.content_transform_template.mode = "train"
        result = DataRecord()
        result.put('position_ids', [1, 2, 4])
        result.put('input_ids', [1, 2, 3])
        result.put('images', ms.Tensor(np.array([1, 2, 3])))
        res = self.content_transform_template.get_need_update_output_items(result).get('position_ids').tolist()
        assert res[0] == 1
        assert res[2] == 4

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_post_process(self):
        """test post process."""
        output_ids = ["1", "2", "3"]
        res = self.content_transform_template.post_process(output_ids)
        assert res == ['', '', '\x00']
        self.content_transform_template.signal_type = "vqa"
        res = self.content_transform_template.post_process(output_ids)
        assert res == ['', '', '\x00']
        self.content_transform_template.signal_type = "chat"
        res = self.content_transform_template.post_process(output_ids)
        assert res == ['', '', '\x00']
