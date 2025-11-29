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
"""test tokenization auto."""
import os
import shutil
import json
import sys
from unittest.mock import patch, MagicMock
import yaml
import pytest
import sentencepiece as spm
from mindformers.models.auto.tokenization_auto import AutoTokenizer, is_experimental_mode
from mindformers.models.auto.configuration_auto import AutoConfig
from mindformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
from mindformers.models.auto.tokenization_auto import tokenizer_class_from_name

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class TestAutoTokenizer:
    """ test auto tokenizer """
    @classmethod
    def setup_class(cls):
        """ create test directory """
        cls.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_tokenizer_test_coverage")
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        os.makedirs(cls.test_dir)

        cls.vocab_file = os.path.join(cls.test_dir, "tokenizer.model")

        # Create dummy sentencepiece model
        corpus_path = os.path.join(cls.test_dir, "corpus.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write("This is a test corpus for sentencepiece training. One Two Three.")

        model_prefix = os.path.join(cls.test_dir, "tokenizer")
        spm.SentencePieceTrainer.Train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=100,  # Increased vocab size to avoid "smaller than required_chars" error
            model_type='bpe',
            character_coverage=1.0,
            user_defined_symbols=['<pad>']
        )

    @classmethod
    def teardown_class(cls):
        """ Clean up """
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_is_experimental_mode(self):
        """
        Coverage for `is_experimental_mode`
        """
        # 1. Directory with .yaml -> False (Origin Mode)
        yaml_dir = os.path.join(self.test_dir, "mode_yaml")
        os.makedirs(yaml_dir, exist_ok=True)
        with open(os.path.join(yaml_dir, "model.yaml"), 'w', encoding="utf-8") as f:
            f.write("key: value")
        assert not is_experimental_mode(yaml_dir)

        # 2. Directory without .yaml -> True (Experimental/HF Mode)
        json_dir = os.path.join(self.test_dir, "mode_json")
        os.makedirs(json_dir, exist_ok=True)
        # Assuming no yaml here
        assert is_experimental_mode(json_dir)

        # 3. Supported model name logic
        # We can't easily modify global TOKENIZER_SUPPORT_LIST safely, but we can test unknown string
        assert is_experimental_mode("unknown_model_string")

        # 4. Path exists but is a file (and unsupported string) -> True
        dummy_file = os.path.join(self.test_dir, "dummy_file.txt")
        with open(dummy_file, 'w', encoding="utf-8") as f:
            f.write("content")
        assert is_experimental_mode(dummy_file)

    @patch("mindformers.tools.MindFormerRegister")
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_from_pretrained_origin_mode(self, mock_register):
        """
        Test `from_pretrained` using Origin Mode (YAML detection).
        This simulates loading from a directory containing a YAML file.
        """
        yaml_dir = os.path.join(self.test_dir, "origin_mode_load")
        os.makedirs(yaml_dir, exist_ok=True)

        # Prepare environment: valid yaml + vocab file
        shutil.copy(self.vocab_file, os.path.join(yaml_dir, "tokenizer.model"))

        config_data = {
            "processor": {
                "tokenizer": {
                    "type": "LlamaTokenizer",
                    "vocab_file": "tokenizer.model"
                }
            }
        }
        with open(os.path.join(yaml_dir, "mindspore_model.yaml"), 'w', encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Setup Mock
        mock_tokenizer_cls = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance
        # When MindFormerRegister.get_cls is called, return our mock class
        mock_register.get_cls.return_value = mock_tokenizer_cls

        # Call SUT
        tokenizer = AutoTokenizer.from_pretrained(yaml_dir)

        # Verifications
        # 1. Should detect yaml -> origin mode
        # 2. Origin mode calls MindFormerRegister.get_cls(..., class_name='LlamaTokenizer')
        mock_register.get_cls.assert_called_with(module_type='tokenizer', class_name='LlamaTokenizer')
        # 3. Should instantiate and call from_pretrained on the retrieved class
        mock_tokenizer_cls.from_pretrained.assert_called()
        assert tokenizer == mock_tokenizer_instance

    @patch("mindformers.models.auto.tokenization_auto.tokenizer_class_from_name")
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_from_pretrained_experimental_mode(self, mock_class_from_name):
        """
        Test `from_pretrained` using Experimental Mode (JSON/HF style).
        This simulates loading from a directory with tokenizer_config.json
        """
        json_dir = os.path.join(self.test_dir, "exp_mode_load")
        os.makedirs(json_dir, exist_ok=True)

        shutil.copy(self.vocab_file, os.path.join(json_dir, "tokenizer.model"))

        # Create tokenizer_config.json indicating the class
        config_data = {
            "tokenizer_class": "LlamaTokenizer",
            "vocab_file": "tokenizer.model"
        }
        with open(os.path.join(json_dir, "tokenizer_config.json"), 'w', encoding="utf-8") as f:
            json.dump(config_data, f)

        # Setup Mock
        mock_tokenizer_cls = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance
        mock_class_from_name.return_value = mock_tokenizer_cls

        # Call SUT
        tokenizer = AutoTokenizer.from_pretrained(json_dir)
        mock_class_from_name.assert_called_with("LlamaTokenizerFast")
        mock_tokenizer_cls.from_pretrained.assert_called_with(json_dir, _from_auto=True, _commit_hash=None)
        assert tokenizer == mock_tokenizer_instance

    @patch("mindformers.models.auto.tokenization_auto.tokenizer_class_from_name")
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_from_pretrained_with_explicit_type(self, mock_class_from_name):
        """
        Test `from_pretrained` when `tokenizer_type` arg is explicitly provided.
        """
        mock_tokenizer_cls = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance
        mock_class_from_name.return_value = mock_tokenizer_cls

        # 'llama' is a known key in TOKENIZER_MAPPING_NAMES
        # Origin or Exp mode check happens first, but tokenizer_type argument logic inside AutoTokenizer
        # usually shortcuts or guides the class selection.

        dummy_path = "dummy_path_not_exist"
        # This triggers experimental mode because it doesn't exist locally (usually) and not in support list

        tokenizer = AutoTokenizer.from_pretrained(dummy_path, tokenizer_type="llama")

        # Verify "llama" -> "LlamaTokenizerFast" (default preference) mapping usage
        mock_class_from_name.assert_called_with("LlamaTokenizerFast")
        assert tokenizer == mock_tokenizer_instance

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_invalid_yaml_name(self):
        """
        Test the `invalid_yaml_name` method for filtering/validating model names.
        """
        # "invalid_name" should return True (invalid)
        assert AutoTokenizer.invalid_yaml_name("invalid_name")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_register_custom_tokenizer(self):
        """
        Test `AutoTokenizer.register` API.
        """

        class MyConfig(AutoConfig):
            pass

        class MyTokenizer:
            pass

        # Register
        AutoTokenizer.register(MyConfig, slow_tokenizer_class=MyTokenizer, exist_ok=True)

        # Check registration
        assert MyConfig in TOKENIZER_MAPPING
        assert TOKENIZER_MAPPING[MyConfig][0] == MyTokenizer

        # Cleanup
        # TOKENIZER_MAPPING is a _LazyAutoMapping object, not a dict.
        # It stores extra content in _extra_content.
        # pylint: disable=W0212
        if MyConfig in TOKENIZER_MAPPING._extra_content:
            del TOKENIZER_MAPPING._extra_content[MyConfig]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_tokenizer_class_from_name_helper_real(self):
        """Test the helper function for loading class (real execution)"""
        res = tokenizer_class_from_name("UnknownTokenizerClassXYZ")
        assert res is None

    # Correct the mock structure: {'type': {'name_suffix': ['full_name']}}
    # glm4_9b -> type: glm4, suffix: 9b.
    @patch("mindformers.models.auto.tokenization_auto.TOKENIZER_SUPPORT_LIST", {'glm4': {'9b': ['glm4_9b']}})
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_invalid_yaml_name_logic(self):
        """Test extended logic of invalid_yaml_name"""
        # Valid cases
        # With corrected mock, these should return False (not invalid) and NOT raise ValueError
        assert not AutoTokenizer.invalid_yaml_name("glm4_9b")
        assert not AutoTokenizer.invalid_yaml_name("mindspore/glm4_9b")

        # Invalid cases
        # unknown_model is not in support list keys, so returns True immediately (no exception)
        assert AutoTokenizer.invalid_yaml_name("unknown_model")

        # "glm4_unknown" starts with a known prefix "glm4" but fails specific model check, raising ValueError
        with pytest.raises(ValueError):
            AutoTokenizer.invalid_yaml_name("glm4_unknown")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_origin_mode_download_logic(self):
        """Test origin mode triggering download paths (mocked)"""
        # We need to mock MindFormerBook and os.path.exists to simulate download paths
        # Also mock os.makedirs to prevent attempting to create directories for "downloads"
        with patch("mindformers.models.auto.tokenization_auto.MindFormerBook") as mock_book, \
                patch("os.path.exists") as mock_exists, \
                patch("mindformers.models.auto.tokenization_auto.set_default_yaml_file") as mock_set_yaml, \
                patch("mindformers.models.auto.tokenization_auto.AutoTokenizer._get_class_name_from_yaml") as \
                        mock_get_cls, \
                patch("mindformers.tools.MindFormerRegister") as _:
            mock_book.get_xihe_checkpoint_download_folder.return_value = "/tmp/xihe"
            mock_book.get_default_checkpoint_download_folder.return_value = "/tmp/default"
            mock_exists.return_value = False  # Simulate file needs download logic trigger (though code just mkdirs)
            mock_get_cls.return_value = ("LlamaTokenizer", MagicMock(processor=MagicMock(tokenizer={})))

            # Mock set_default_yaml_file to avoid actual file ops or checks
            mock_set_yaml.return_value = None

            # 1. Mindspore prefix
            AutoTokenizer.get_class_from_origin_mode("mindspore/glm4_9b")
            # Check that it tried to access the download folder, confirming logic path
            mock_book.get_xihe_checkpoint_download_folder.assert_called()

            # 2. Default prefix
            with patch("mindformers.models.auto.tokenization_auto.AutoTokenizer.invalid_yaml_name", return_value=False):
                AutoTokenizer.get_class_from_origin_mode("glm4_9b")
                mock_book.get_default_checkpoint_download_folder.assert_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_origin_mode_error_handling(self):
        """Test error paths in get_class_from_origin_mode"""
        # 1. Not a string
        with pytest.raises(TypeError):
            AutoTokenizer.get_class_from_origin_mode(123)

        # 2. Path is dir but no yaml (class_name is None)
        with patch("os.path.isdir", return_value=True), \
                patch("os.path.exists", return_value=True), \
                patch("mindformers.models.auto.tokenization_auto.AutoTokenizer._get_class_name_from_yaml",
                      return_value=(None, None)):
            with pytest.raises(ValueError):
                AutoTokenizer.get_class_from_origin_mode("/tmp/dummy_dir")

        # 3. Unsupported model
        with patch("mindformers.models.auto.tokenization_auto.AutoTokenizer.invalid_yaml_name", return_value=True), \
                patch("os.path.exists", return_value=False), \
                patch("os.path.isdir", return_value=False), \
                patch("os.makedirs") as _:
            with pytest.raises(FileNotFoundError):
                AutoTokenizer.get_class_from_origin_mode("unsupported_model")

    @patch("mindformers.models.auto.tokenization_auto.AutoConfig")
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_experimental_mode_legacy_automap(self, mock_auto_config):
        """Test experimental mode with legacy auto_map format (list/tuple)"""
        json_dir = os.path.join(self.test_dir, "legacy_automap")
        os.makedirs(json_dir, exist_ok=True)

        # 1. Setup tokenizer config with LEGACY auto_map (list)
        tokenizer_config_data = {
            "auto_map": ["AutoTokenizer", "LlamaTokenizer"],
        }

        # 2. Mock AutoConfig to avoid loading real config (which might fail in test env)
        # and to strictly control attributes.
        mock_config_instance = MagicMock()
        mock_config_instance.tokenizer_class = None
        # Ensure auto_map attribute is missing or None to avoid entering the crashing block (lines 532-533)
        del mock_config_instance.auto_map
        mock_config_instance.auto_map = None

        mock_auto_config.from_pretrained.return_value = mock_config_instance

        with open(os.path.join(json_dir, "tokenizer_config.json"), 'w', encoding="utf-8") as f:
            json.dump(tokenizer_config_data, f)

        with patch("mindformers.models.auto.tokenization_auto.get_tokenizer_config",
                   return_value=tokenizer_config_data), \
                patch("mindformers.models.auto.tokenization_auto.resolve_trust_remote_code", return_value=True), \
                patch("mindformers.models.auto.tokenization_auto.get_class_from_dynamic_module") as mock_get_class:
            mock_tokenizer = MagicMock()
            mock_get_class.return_value = mock_tokenizer

            AutoTokenizer.from_pretrained(json_dir, trust_remote_code=True)
            # Verify it tried to load from dynamic module using legacy format
            mock_get_class.assert_called()

    @patch("mindformers.models.auto.tokenization_auto.TOKENIZER_MAPPING")
    @patch("mindformers.models.auto.tokenization_auto.AutoConfig")
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_from_pretrained_value_errors(self, mock_auto_config, mock_mapping):
        """Test ValueError scenarios in from_pretrained"""
        # Mock keys() to return a simple list of mocks with __name__ to avoid iteration errors over LazyMapping
        mock_cls = MagicMock()
        mock_cls.__name__ = "MockConfig"
        mock_mapping.keys.return_value = [mock_cls]

        # 1. Tokenizer type not found
        # ValueError is wrapped in RuntimeError by @experimental_mode_func_checker
        with pytest.raises(RuntimeError):
            AutoTokenizer.from_pretrained("dummy", tokenizer_type="InvalidType")

        # 2. Config class unrecognized
        mock_config_instance = MagicMock()
        mock_config_instance.tokenizer_class = None

        mock_auto_config.from_pretrained.return_value = mock_config_instance

        # ValueError is wrapped in RuntimeError by @experimental_mode_func_checker
        with pytest.raises(RuntimeError):
            with patch("mindformers.models.auto.tokenization_auto.get_tokenizer_config", return_value={}):
                AutoTokenizer.from_pretrained("dummy_path_val_error")
