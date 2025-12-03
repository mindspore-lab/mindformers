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
"""UTs for Glm4 modeling API."""
import os
import pytest

from mindformers.models.glm4.configuration_glm4 import Glm4Config
from mindformers.models.glm4.modeling_glm4 import Glm4ForCausalLM
from mindformers.models.glm4.modeling_glm4_infer import InferenceGlm4ForCausalLM


class TestGlm4ForCausalLM:
    """Ensure Glm4ForCausalLM routes to the proper implementation."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_init_model_in_predict_mode(self):
        """When RUN_MODE is unset/predict, the inference model should be instantiated."""
        os.environ['RUN_MODE'] = "predict"
        config = Glm4Config()

        model = Glm4ForCausalLM(config)

        assert isinstance(model, InferenceGlm4ForCausalLM)
        assert model.config is config

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_init_model_in_train_mode(self):
        """RUN_MODE=train should raise an explicit NotImplementedError."""
        os.environ['RUN_MODE'] = "train"

        with pytest.raises(NotImplementedError):
            Glm4ForCausalLM(Glm4Config())
