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
Test module for testing the batch infer for text generator.
How to run this:
pytest tests/st/test_text_generator/test_batch_generate.py
"""
import numpy as np

import mindspore as ms
from mindspore import set_seed

from mindformers import LlamaConfig, LlamaForCausalLM

ms.set_context(mode=0)


class TestBeamSearchGenerate:
    """A test class for testing text generate features."""

    def setup_method(self):
        """setup method."""
        set_seed(0)
        np.random.seed(0)

        vocab_size = 1024
        input_length = 8
        seq_length = 16

        model_config = LlamaConfig(batch_size=1, vocab_size=1024, num_layers=2, seq_length=seq_length,
                                   checkpoint_name_or_path="")
        self.model = LlamaForCausalLM(model_config)

        self.input_ids = np.pad(np.random.randint(low=0, high=vocab_size, size=(1, input_length)),
                                ((0, 0), (0, seq_length - input_length)), mode='constant',
                                constant_values=0).tolist()

        self.output_ids_std = np.array([[115, 976, 755, 709, 1022, 847, 431, 448, 251, 735, 735,
                                         735, 735, 735, 735, 735]]).astype(np.int32)

    def test_generate(self):
        """
        Feature: batch generate.
        Description: Test correctness of batch generate.
        Expectation: TypeError, ValueError, RuntimeError
        """
        output_ids = self.model.generate(self.input_ids, num_beams=3)

        assert np.array_equal(output_ids, self.output_ids_std), \
            f"output_ids: {output_ids} is not equal to output_ids_std: {self.output_ids_std}"
