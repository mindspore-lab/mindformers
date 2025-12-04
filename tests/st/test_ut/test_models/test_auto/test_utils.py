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
"""
test utils.py
"""
from unittest.mock import patch
import pytest

from mindformers.mindformer_book import MindFormerBook
from mindformers.models.auto.utils import get_default_yaml_file, set_default_yaml_file


class TestYamlFileFunctions:
    """ A test class for testing utils."""
    @pytest.fixture
    def mock_trainer_support_list(self):
        """Mock the trainer support task list"""
        return {
            "text_generation": {
                "model1": "/path/to/model1.yaml",
                "model2": "/path/to/model2.yaml"
            }
        }

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_default_yaml_file_found(self, mock_trainer_support_list):
        """Test getting default yaml file when model exists"""
        with patch.object(
                MindFormerBook, 'get_trainer_support_task_list', return_value=mock_trainer_support_list):
            result = get_default_yaml_file("model1")
            assert result == "/path/to/model1.yaml"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_default_yaml_file_not_found(self, mock_trainer_support_list):
        """Test getting default yaml file when model doesn't exist"""
        with patch.object(
                MindFormerBook, 'get_trainer_support_task_list', return_value=mock_trainer_support_list):
            result = get_default_yaml_file("nonexistent_model")
            assert result == ""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_set_default_yaml_file_success(self, mock_trainer_support_list):
        """Test setting default yaml file when source exists and target doesn't"""
        with patch.object(
                MindFormerBook, 'get_trainer_support_task_list', return_value=mock_trainer_support_list), \
                patch('os.path.exists') as mock_exists, \
                patch('os.path.realpath', return_value='/real/path/to/model1.yaml'), \
                patch('shutil.copy') as mock_copy, \
                patch('mindformers.models.auto.utils.logger') as mock_logger:
            mock_exists.side_effect = lambda path: path != '/target/path.yaml'
            set_default_yaml_file("model1", "/target/path.yaml")
            mock_copy.assert_called_once_with("/path/to/model1.yaml", "/target/path.yaml")
            mock_logger.info.assert_called_once_with("default yaml config in %s is used.", "/target/path.yaml")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_set_default_yaml_file_source_not_found(self, mock_trainer_support_list):
        """Test setting default yaml file when source file doesn't exist"""
        with (patch.object(
                MindFormerBook, 'get_trainer_support_task_list', return_value=mock_trainer_support_list), \
                patch('os.path.exists') as mock_exists, patch('os.path.realpath', return_value=''), \
                patch('shutil.copy') as mock_copy):
            mock_exists.side_effect = lambda path: False
            with pytest.raises(
                    FileNotFoundError, match="default yaml file path must be correct, but get /path/to/model1.yaml"):
                set_default_yaml_file("model1", "/target/path.yaml")
            mock_copy.assert_not_called()
