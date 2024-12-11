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
"""test quantization_config"""
import os.path
import tempfile
from unittest.mock import patch
import pytest
from mindspore_gs.ptq import PTQMode
from mindspore_gs.common import BackendTarget
from mindformers.utils.quantization_config import (
    RtnConfig, PtqConfig, SmoothQuantConfig, QuantizationMethod
)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_config():
    """
    Feature: quantization_config: RtnConfig, PtqConfig, SmoothQuantConfig
    Description: test quantization_config logic
    Expectation: Success
    """
    type_list = ["rtn", "ptq", "smooth"]
    for type_name in type_list:
        if type_name == "rtn":
            config = RtnConfig(quant_method=QuantizationMethod.RTN, mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND,
                               opname_blacklist=['layer0'], outliers_suppression="None")
        elif type_name == "ptq":
            config = PtqConfig(quant_method=QuantizationMethod.PTQ, mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND,
                               opname_blacklist=['layer0'], algo_args={}, outliers_suppression="None")
        else:
            config = SmoothQuantConfig(quant_method=QuantizationMethod.SMOOTH_QUANT, mode=PTQMode.DEPLOY,
                                       backend=BackendTarget.ASCEND, opname_blacklist=['layer0'], algo_args={},
                                       outliers_suppression="smooth", weight_dtype="int8", kvcache_dtype="None",
                                       activation_dtype="int8")
        mode = config.mode
        config.mode = "mock"
        with pytest.raises(ValueError):
            assert config.init_check()
        config.mode = mode

        backend = config.backend
        config.backend = "mock"
        with pytest.raises(ValueError):
            assert config.init_check()
        config.backend = backend

        weight_quant_dtype = config.weight_quant_dtype
        config.weight_quant_dtype = "mock"
        with pytest.raises(ValueError):
            assert config.init_check()
        config.weight_quant_dtype = weight_quant_dtype

        act_quant_dtype = config.act_quant_dtype
        config.act_quant_dtype = "mock"
        with pytest.raises(ValueError):
            assert config.init_check()
        config.act_quant_dtype = act_quant_dtype

        kvcache_quant_dtype = config.kvcache_quant_dtype
        config.kvcache_quant_dtype = "mock"
        with pytest.raises(ValueError):
            assert config.init_check()
        config.kvcache_quant_dtype = kvcache_quant_dtype

        outliers_suppression = config.outliers_suppression
        config.outliers_suppression = "mock"
        with pytest.raises(ValueError):
            assert config.init_check()
        config.outliers_suppression = outliers_suppression


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("copy.deepcopy")
def test_quantization_config_mixin(mock_get):
    """
    Feature: quantization_config.QuantizationConfigMixin
    Description: test QuantizationConfigMixin logic
    Expectation: Success
    """
    config = RtnConfig(quant_method=QuantizationMethod.RTN, mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND,
                       opname_blacklist=['layer0'], outliers_suppression="None")
    config, kwargs = config.from_dict(return_unused_kwargs=True,
                                      config_dict={"quant_method": QuantizationMethod.RTN,
                                                   "backend": BackendTarget.ASCEND, "opname_blacklist": ['layer0'],
                                                   "outliers_suppression": "None"}, mode=PTQMode.DEPLOY)
    assert isinstance(config, RtnConfig)
    assert not kwargs
    assert not config.update(outliers_suppression=None)
    mock_get.return_value = {"opname_blacklist": ['layer0'], "outliers_suppression": "None"}
    assert "opname_blacklist" in config.to_json_string(use_diff=False)
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path = tmp_dir.name
    config.to_json_file(os.path.join(tmp_path, "mock.json"))
    assert os.path.exists(os.path.join(tmp_path, "mock.json"))
