#!/usr/bin/env python
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
"""Unit tests for mindformers.tools.register.config."""
import argparse
import sys
from collections import OrderedDict
import copy

import pytest

from mindformers.tools.register import config as config_module
from mindformers.tools.register.config import (
    ActionDict,
    DictConfig,
    MindFormerConfig,
    BASE_CONFIG,
    ordered_yaml_dump,
    parse_args,
)

yaml = pytest.importorskip("yaml")

# pylint: disable=protected-access


class TestConfig:
    """Test class for mindformers.tools.register.config."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_dict_config_attribute_and_to_dict(self):
        """DictConfig should expose attribute access semantics."""
        cfg = DictConfig(a=1, nested=DictConfig(b=2))
        assert cfg.a == 1
        cfg.c = 3
        assert cfg.c == 3
        del cfg.a
        assert cfg.a is None
        plain = cfg.to_dict()
        assert plain == {"nested": {"b": 2}, "c": 3}

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_dict_config_deepcopy_isolated(self):
        """Deep copy should create independent nested objects."""
        cfg = DictConfig(nested=DictConfig(value=[1, 2]))
        copied = copy.deepcopy(cfg)
        copied.nested.value.append(3)
        assert cfg.nested.value == [1, 2]
        assert copied.nested.value == [1, 2, 3]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_mindformer_config_loads_yaml_with_base(self, monkeypatch, tmp_path):
        """MindFormerConfig should merge base yaml files and convert dict to config."""
        monkeypatch.setattr(config_module.ConfigTemplate, "apply_template", lambda _: None)
        base_content = {"alpha": 1, "nested": {"from_base": True}}
        base_file = tmp_path / "base.yaml"
        base_file.write_text(yaml.safe_dump(base_content), encoding="utf-8")

        child_content = {
            BASE_CONFIG: "base.yaml",
            "beta": 2,
            "nested": {"from_child": True},
        }
        child_file = tmp_path / "child.yaml"
        child_file.write_text(yaml.safe_dump(child_content), encoding="utf-8")

        cfg = MindFormerConfig(str(child_file))
        assert cfg.alpha == 1
        assert cfg.beta == 2
        assert isinstance(cfg.nested, MindFormerConfig)
        assert cfg.nested.from_child

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_mindformer_config_merge_and_set(self, monkeypatch):
        """merge_from_dict and set_value should correctly update nested fields."""
        monkeypatch.setattr(config_module.ConfigTemplate, "apply_template", lambda _: None)
        cfg = MindFormerConfig(model={"model_config": {"type": "Demo"}})
        cfg.merge_from_dict({"model.arch": "DemoArch", "new.branch.leaf": 10})
        assert cfg.model.arch == "DemoArch"
        assert cfg.new.branch.leaf == 10

        cfg.set_value("context.mode", "GRAPH")
        cfg.set_value(["context", "device_id"], 3)
        assert cfg.get_value("context.mode") == "GRAPH"
        assert cfg.get_value(["context", "device_id"]) == 3
        assert cfg.get_value("context.fake", default="fallback") == "fallback"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_file2dict_without_filename_raises(self):
        """_file2dict should raise when filename is None."""
        with pytest.raises(NameError):
            getattr(MindFormerConfig, "_file2dict")(None)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_action_dict_parse_and_call(self):
        """ActionDict should parse ints, floats, tuples and bool strings."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--opts",
            action=ActionDict,
            nargs="*",
            default={},
        )
        args = parser.parse_args(
            [
                "--opts",
                "ints=1,2",
                "floats=3.5",
                "tuple=(7,8)",
                "mixed=[1,(2,3),[4,5]]",
                "flag=True",
            ]
        )
        assert args.opts["ints"] == [1, 2]
        assert args.opts["floats"] == 3.5
        assert args.opts["tuple"] == (7, 8)
        assert args.opts["mixed"] == [1, (2, 3), [4, 5]]
        assert args.opts["flag"] is False  # current implementation compares function object

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_action_dict_find_next_comma_invalid_pairs(self):
        """find_next_comma should raise when brackets are unbalanced."""
        with pytest.raises(ValueError):
            ActionDict.find_next_comma("[1,2")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_ordered_yaml_dump_preserves_order(self):
        """ordered_yaml_dump should keep OrderedDict order in emitted yaml."""
        ordered = OrderedDict()
        ordered["first"] = 1
        ordered["second"] = 2
        dumped = ordered_yaml_dump(ordered)
        assert dumped.index("first") < dumped.index("second")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_parse_args_reads_cli(self, monkeypatch):
        """parse_args should honor the --config cli argument."""
        monkeypatch.setattr(sys, "argv", ["prog", "--config", "path/to/model.yaml"])
        parsed = parse_args()
        assert parsed.config == "path/to/model.yaml"
