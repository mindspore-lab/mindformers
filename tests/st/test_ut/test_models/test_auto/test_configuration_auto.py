#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 Huawei Technologies
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
"""Unit tests for mindformers.models.auto.configuration_auto."""
from types import SimpleNamespace

import pytest

import mindformers.models.auto.configuration_auto as auto_cfg
from mindformers.models.auto.configuration_auto import (
    AutoConfig,
    CONFIG_MAPPING,
    config_class_to_model_type,
    _LazyConfigMapping,
    _list_model_options,
    replace_list_option_in_docstrings,
)
from mindformers.models.configuration_utils import PretrainedConfig


class DummyMindFormerConfig(dict):
    """Stub MindFormerConfig for unit tests."""

    def __init__(self, use_legacy=True, has_pretrained=False, has_generation=False):
        super().__init__()
        self._use_legacy = use_legacy
        self.model = SimpleNamespace(
            model_config={"type": "DemoConfig"},
            arch=SimpleNamespace(type="demo_arch"),
        )
        if has_pretrained:
            self["pretrained_model_dir"] = "pretrained_dir"
            self.pretrained_model_dir = "pretrained_dir"
        else:
            self.pretrained_model_dir = None
        if has_generation:
            self["generation_config"] = {"gen": True}
            self.generation_config = {"gen": True}
        else:
            self.generation_config = None

    def get_value(self, key, default=None):
        """Get value from config."""
        if key == "use_legacy":
            return self._use_legacy
        return default


@pytest.fixture(autouse=True)
def restore_extra_content():
    """Ensure CONFIG_MAPPING extra registrations are restored between tests."""
    # Accessing protected member for test cleanup is intentional
    original = CONFIG_MAPPING._extra_content.copy()  # pylint: disable=W0212,protected-access
    yield
    CONFIG_MAPPING._extra_content = original  # pylint: disable=W0212,protected-access


class TestConfigurationAuto:
    """Test class for mindformers.models.auto.configuration_auto."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_config_class_to_model_type_core_and_extra(self, monkeypatch):
        """config_class_to_model_type should inspect default and extra registries."""
        assert config_class_to_model_type("LlamaConfig") == "llama"
        dummy_class = type("NewConfig", (), {})
        # Accessing protected member for test setup is intentional
        monkeypatch.setitem(CONFIG_MAPPING._extra_content, "custom", dummy_class)  # pylint: disable=W0212,protected-access
        assert config_class_to_model_type("NewConfig") == "custom"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_lazy_config_mapping_register_and_getitem(self, monkeypatch):
        """_LazyConfigMapping should import modules lazily and honor register."""
        module = SimpleNamespace(MockConfig="sentinel")
        monkeypatch.setattr(auto_cfg.importlib, "import_module", lambda name, package=None: module)
        mapping = _LazyConfigMapping({"mock": "MockConfig"})
        assert mapping["mock"] == "sentinel"
        mapping.register("extra", "ExtraConfig", exist_ok=True)
        assert mapping["extra"] == "ExtraConfig"
        with pytest.raises(ValueError):
            mapping.register("mock", "OtherConfig")
        with pytest.raises(KeyError):
            _ = mapping["missing"]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_list_model_options_and_docstring_replacement(self):
        """_list_model_options and decorator should update docstrings or raise errors."""
        doc = _list_model_options("  ", {"llama": ["LlamaConfig"]}, use_model_types=False)
        assert "LlamaConfig" in doc

        @replace_list_option_in_docstrings({"llama": ["LlamaConfig"]})
        def sample():
            """List options"""

        assert "llama" in sample.__doc__

        def broken():
            """no placeholder"""

        decorator = replace_list_option_in_docstrings({"llama": ["LlamaConfig"]})
        with pytest.raises(ValueError):
            decorator(broken)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_autoconfig_invalid_yaml_name_branches(self, monkeypatch):
        """AutoConfig.invalid_yaml_name should validate against support list."""
        monkeypatch.setattr(AutoConfig, "_support_list",
                            {"llama": ["llama_7b"], "glm": {"9b": ["glm_9b"]}})
        assert AutoConfig.invalid_yaml_name("unknown_model")
        assert not AutoConfig.invalid_yaml_name("llama_7b")
        assert not AutoConfig.invalid_yaml_name("glm_9b")
        with pytest.raises(ValueError):
            AutoConfig.invalid_yaml_name("glm_bad")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_autoconfig_for_model_and_error(self):
        """AutoConfig.for_model should instantiate registered configs or raise."""
        class DummyConfig(PretrainedConfig):
            """Dummy config for unit tests."""
            model_type = "dummy_key"

            def __init__(self, value=None):
                super().__init__()
                self.value = value

        CONFIG_MAPPING.register("dummy_key", DummyConfig, exist_ok=True)
        result = AutoConfig.for_model("dummy_key", value=3)
        assert isinstance(result, DummyConfig) and result.value == 3
        with pytest.raises(ValueError):
            AutoConfig.for_model("missing")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_from_pretrained_switches_modes(self, monkeypatch):
        """AutoConfig.from_pretrained should delegate based on experimental flag."""
        monkeypatch.setattr(auto_cfg, "is_experimental_mode", lambda _: False)
        monkeypatch.setattr(AutoConfig, "get_config_origin_mode",
                            classmethod(lambda cls, name, **_: ("origin", name)))
        res = AutoConfig.from_pretrained("path/model.yaml", pretrained_model_name_or_path="override")
        assert res == ("origin", "override")
        monkeypatch.setattr(auto_cfg, "is_experimental_mode", lambda _: True)
        monkeypatch.setattr(AutoConfig, "get_config_experimental_mode",
                            classmethod(lambda cls, name, **_: ("exp", name)))
        assert AutoConfig.from_pretrained("path/model.yaml") == ("exp", "path/model.yaml")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_config_origin_mode_type_and_extension_errors(self, tmp_path):
        """get_config_origin_mode should validate input types and extensions."""
        with pytest.raises(TypeError):
            AutoConfig.get_config_origin_mode(123)
        bad_file = tmp_path / "not_yaml.txt"
        bad_file.write_text("content", encoding="utf-8")
        with pytest.raises(ValueError):
            AutoConfig.get_config_origin_mode(str(bad_file))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_config_origin_mode_invalid_yaml_name(self, monkeypatch):
        """Non-existing yaml names should raise ValueError."""
        monkeypatch.setattr(AutoConfig, "invalid_yaml_name", classmethod(lambda cls, _: True))
        with pytest.raises(ValueError):
            AutoConfig.get_config_origin_mode("unknown_name")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_config_origin_mode_legacy_flow(self, monkeypatch, tmp_path):
        """Legacy pathway should build configs and update auxiliary fields."""
        dummy = DummyMindFormerConfig(use_legacy=True, has_pretrained=True, has_generation=True)
        monkeypatch.setattr(auto_cfg, "MindFormerConfig", lambda *_: dummy)
        built = {}
        monkeypatch.setattr(auto_cfg, "build_model_config",
                            lambda cfg: built.setdefault("config", cfg) or "legacy")
        monkeypatch.setattr(auto_cfg.MindFormerBook, "set_model_config_to_name",
                            lambda *args, **kwargs: built.setdefault("mark", args))
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("model: {}", encoding="utf-8")
        AutoConfig.get_config_origin_mode(str(yaml_file), hidden_size=128)
        assert dummy.model.model_config["hidden_size"] == 128
        assert dummy.model.pretrained_model_dir == "pretrained_dir"
        assert dummy.model.generation_config == {"gen": True}
        assert built["config"] == dummy.model.model_config

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_config_origin_mode_nonlegacy_flow(self, monkeypatch, tmp_path):
        """Non-legacy pathway should use get_model_config without calling builder."""
        dummy = DummyMindFormerConfig(use_legacy=False)
        monkeypatch.setattr(auto_cfg, "MindFormerConfig", lambda *_: dummy)
        marker = {}
        monkeypatch.setattr(auto_cfg, "build_model_config",
                            lambda *_: marker.setdefault("should_not_call", True))
        monkeypatch.setattr(auto_cfg, "get_model_config",
                            lambda model: marker.setdefault("model", model) or "new_config")
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("model: {}", encoding="utf-8")
        AutoConfig.get_config_origin_mode(str(yaml_file), dropout=0.1)
        assert dummy.model.model_config["dropout"] == 0.1
        assert marker["model"] == dummy.model
        assert "should_not_call" not in marker

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_config_experimental_mode_remote_code(self, monkeypatch):
        """Remote code configs should be loaded via dynamic modules when trusted."""
        monkeypatch.setattr(auto_cfg.PretrainedConfig, "get_config_dict",
                            classmethod(lambda cls, name,
                                        **kwargs: ({"auto_map": {"AutoConfig": "mod.Class"}}, {})))
        monkeypatch.setattr(auto_cfg, "resolve_trust_remote_code", lambda trust, *args, **kwargs: True)

        class RemoteConfig:
            """Remote config for unit tests."""
            @staticmethod
            def register_for_auto_class():
                """Register for auto class."""
                RemoteConfig.registered = True

            @staticmethod
            def from_pretrained(name, **kwargs):
                """From pretrained."""
                return {"name": name, "kwargs": kwargs}

        monkeypatch.setattr(auto_cfg, "get_class_from_dynamic_module", lambda *args, **kwargs: RemoteConfig)
        monkeypatch.setattr(auto_cfg.os.path, "isdir", lambda _: True)
        result = AutoConfig.get_config_experimental_mode("remote_repo", trust_remote_code=True)
        assert result["name"] == "remote_repo"
        assert RemoteConfig.registered is True

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_config_experimental_mode_local_config(self, monkeypatch):
        """Local configs with model_type should resolve via CONFIG_MAPPING."""
        class LocalConfig(PretrainedConfig):
            """LocalConfig for tests."""
            model_type = "custom_dummy"

            @classmethod
            def from_dict(cls, config_dict, **kwargs):
                return {"config": config_dict, "extra": kwargs}

        CONFIG_MAPPING.register("custom_dummy", LocalConfig, exist_ok=True)
        monkeypatch.setattr(auto_cfg.PretrainedConfig, "get_config_dict",
                            classmethod(lambda cls, name,
                                        **kwargs: ({"model_type": "custom_dummy", "value": 1}, {"unused": True})))
        result = AutoConfig.get_config_experimental_mode("local_repo")
        assert result["config"]["value"] == 1
        assert result["extra"]["unused"] is True
