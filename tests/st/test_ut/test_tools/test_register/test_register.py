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
"""test register.py"""
from mindformers.core.context.build_context import build_context, set_context
from mindformers.tools.register.register import MindFormerRegister, MindFormerModuleType
import pytest
from .model_class import MyModel
from .model_class_legacy import MyModel as MyModelNew


class TestMindFormerRegister:
    """test MindFormerRegister"""
    @classmethod
    def setup_class(cls):
        build_context({"use_legacy": True})

    def test_register_decorator_and_is_exist_case(self):
        """
        Test registering with decorator (legacy=True), check existence and internal keys.
        Input: MyModel registered with legacy=True.
        Output: MyModel is found in registry, both 'MyModel' and 'mcore_MyModel' keys exist.
        Expected: get_cls returns MyModel, keys present.
        """
        assert MindFormerRegister.is_exist(MindFormerModuleType.MODELS, "MyModel")
        assert MindFormerRegister.get_cls(MindFormerModuleType.MODELS, "MyModel") is MyModel
        keys = list(MindFormerRegister.registry[MindFormerModuleType.MODELS].keys())
        assert "MyModel" in keys
        assert "mcore_MyModel" in keys

    def test_register_decorator_legacy_false_case(self):
        """
        Test registering with decorator (legacy=False), check existence and internal keys.
        Input: MyModel registered with legacy=False, context set to use_legacy=False.
        Output: MyModel is found in registry, both 'MyModel' and 'mcore_MyModel' keys exist.
        Expected: get_cls returns MyModelNew, keys present.
        """
        set_context(use_legacy=False)
        assert MindFormerRegister.is_exist(MindFormerModuleType.MODELS, "MyModel")
        assert MindFormerRegister.get_cls(MindFormerModuleType.MODELS, "MyModel") is MyModelNew
        keys = list(MindFormerRegister.registry[MindFormerModuleType.MODELS].keys())
        assert "mcore_MyModel" in keys
        assert "MyModel" in keys

    def test_legacy_switching_case(self):
        """
        Test switching between legacy=True and legacy=False, check correct class is returned.
        Input: Switch context between use_legacy True/False.
        Output: get_cls returns correct class for each context.
        Expected: MyModel for legacy=True, MyModelNew for legacy=False.
        """
        set_context(use_legacy=True)
        assert MindFormerRegister.is_exist(MindFormerModuleType.MODELS, "MyModel")
        assert MindFormerRegister.get_cls(MindFormerModuleType.MODELS, "MyModel") is MyModel

        set_context(use_legacy=False)
        assert MindFormerRegister.is_exist(MindFormerModuleType.MODELS, "MyModel")
        assert MindFormerRegister.get_cls(MindFormerModuleType.MODELS, "MyModel") is MyModelNew

        set_context(use_legacy=True)
        assert MindFormerRegister.get_cls(MindFormerModuleType.MODELS, "MyModel") is MyModel

    def test_register_cls_manual_case(self):
        """
        Test manually registering classes with legacy=True and legacy=False, check existence and keys.
        Input: Register ManualLegacy (legacy=True), ManualNew (legacy=False).
        Output: Both classes are found in registry, correct keys exist.
        Expected: get_cls returns correct class, keys present.
        """
        class ManualLegacy:
            pass
        MindFormerRegister.register_cls(ManualLegacy, MindFormerModuleType.MODELS, legacy=True)
        set_context(use_legacy=True)
        assert MindFormerRegister.is_exist(MindFormerModuleType.MODELS, "ManualLegacy")
        assert MindFormerRegister.get_cls(MindFormerModuleType.MODELS, "ManualLegacy") is ManualLegacy

        class ManualNew:
            pass
        MindFormerRegister.register_cls(ManualNew, MindFormerModuleType.MODELS, legacy=False)
        set_context(use_legacy=False)
        assert MindFormerRegister.is_exist(MindFormerModuleType.MODELS, "ManualNew")
        assert MindFormerRegister.get_cls(MindFormerModuleType.MODELS, "ManualNew") is ManualNew

        keys = list(MindFormerRegister.registry[MindFormerModuleType.MODELS].keys())
        assert "mcore_ManualNew" in keys
        assert "ManualLegacy" in keys

    def test_get_cls_not_exist_case(self):
        """
        Test querying a non-existent class, should raise ValueError.
        Input: Query NotExistModel.
        Output: ValueError is raised.
        Expected: Exception is thrown.
        """
        set_context(use_legacy=True)
        with pytest.raises(ValueError):
            MindFormerRegister.get_cls(MindFormerModuleType.MODELS, "NotExistModel")

    def test_is_exist_without_class_name_case(self):
        """
        Test checking if the type exists in registry.
        Input: Only module_type provided.
        Output: Returns True if type exists.
        Expected: is_exist returns True.
        """
        assert MindFormerRegister.is_exist(MindFormerModuleType.MODELS)
