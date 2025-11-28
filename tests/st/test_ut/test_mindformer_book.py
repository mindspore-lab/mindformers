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
test mindformer_book.py
"""
from unittest.mock import patch
import pytest

from mindformers.mindformer_book import MindFormerBook


#pylint: disable=W0212
class TestMindFormerBook:
    """ A test class for testing mindformer_book."""
    def setup_method(self):
        """Execute before each test method: save original data and set up test data"""
        self.original_trainer_list = getattr(MindFormerBook, '_TRAINER_SUPPORT_TASKS_LIST', {})
        self.original_pipeline_list = getattr(MindFormerBook, '_PIPELINE_SUPPORT_TASK_LIST', {})

        MindFormerBook._TRAINER_SUPPORT_TASKS_LIST = {
            "general": {"some_key": "some_value"},
            "text_generation": {
                "common": {"config": "value"},
                "model1": "path1",
                "model2": "path2"
            },
            "text_classification": {
                "common": {"config": "value"},
                "model3": "path3"
            }
        }

        MindFormerBook._PIPELINE_SUPPORT_TASK_LIST = {
            "text_generation": {
                "common": {"config": "value"},
                "model1": "path1",
                "model2": "path2"
            },
            "text_classification": {
                "common": {"config": "value"},
                "model3": "path3"
            },
            "image_classification": {
                "common": {"config": "value"},
                "model4": "path4"
            }
        }

    def teardown_method(self):
        """Execute after each test method: restore original data"""
        MindFormerBook._TRAINER_SUPPORT_TASKS_LIST = self.original_trainer_list
        MindFormerBook._PIPELINE_SUPPORT_TASK_LIST = self.original_pipeline_list

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_show_trainer_support_model_list_without_task(self):
        """Test case when no task is specified"""
        with patch('mindformers.mindformer_book.print_dict') as mock_print_dict, \
                patch('mindformers.mindformer_book.logger') as mock_logger:
            MindFormerBook.show_trainer_support_model_list()
            mock_logger.info.assert_called_with("Trainer support model list of MindFormer is: ")
            mock_print_dict.assert_called_once()
            call_args = mock_print_dict.call_args[0][0]
            assert "text_generation" in call_args
            assert "text_classification" in call_args
            assert "general" not in call_args
            assert call_args["text_generation"] == ["model1", "model2"]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_show_trainer_support_model_list_with_valid_task(self):
        """Test case when a valid task is specified"""
        with patch('mindformers.mindformer_book.print_path_or_list') as mock_print_list, \
                patch('mindformers.mindformer_book.logger') as mock_logger:
            MindFormerBook.show_trainer_support_model_list(task="text_generation")
            mock_logger.info.assert_called_with("Trainer support model list for %s task is: ", "text_generation")
            mock_print_list.assert_called_once()
            call_args = mock_print_list.call_args[0][0]
            assert call_args == ["model1", "model2"]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_show_trainer_support_model_list_with_another_valid_task(self):
        """Test case when another valid task is specified"""
        with patch('mindformers.mindformer_book.print_path_or_list') as mock_print_list, \
                patch('mindformers.mindformer_book.logger') as mock_logger:
            MindFormerBook.show_trainer_support_model_list(task="text_classification")
            mock_logger.info.assert_called_with("Trainer support model list for %s task is: ", "text_classification")
            mock_print_list.assert_called_once()
            call_args = mock_print_list.call_args[0][0]
            assert call_args == ["model3"]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_show_trainer_support_model_list_with_invalid_task(self):
        """Test case when an invalid task is specified"""
        with patch('mindformers.mindformer_book.logger') as mock_logger:
            with pytest.raises(KeyError, match="unsupported task"):
                MindFormerBook.show_trainer_support_model_list(task="invalid_task")
            mock_logger.info.assert_not_called()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_show_pipeline_support_model_list_without_task(self):
        """Test pipeline case when no task is specified"""
        with patch('mindformers.mindformer_book.print_dict') as mock_print_dict, \
                patch('mindformers.mindformer_book.logger') as mock_logger:
            MindFormerBook.show_pipeline_support_model_list()
            mock_logger.info.assert_called_with("Pipeline support model list of MindFormer is: ")
            mock_print_dict.assert_called_once()
            call_args = mock_print_dict.call_args[0][0]
            assert "text_generation" in call_args
            assert "text_classification" in call_args
            assert "image_classification" in call_args
            assert call_args["text_generation"] == ["model1", "model2"]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_show_pipeline_support_model_list_with_valid_task(self):
        """Test pipeline case when a valid task is specified"""
        with patch('mindformers.mindformer_book.print_path_or_list') as mock_print_list, \
                patch('mindformers.mindformer_book.logger') as mock_logger:
            MindFormerBook.show_pipeline_support_model_list(task="text_generation")
            mock_logger.info.assert_called_with("Pipeline support model list for %s task is: ", "text_generation")
            mock_print_list.assert_called_once()
            call_args = mock_print_list.call_args[0][0]
            assert call_args == ["model1", "model2"]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_show_pipeline_support_model_list_with_another_valid_task(self):
        """Test pipeline case when another valid task is specified"""
        with patch('mindformers.mindformer_book.print_path_or_list') as mock_print_list, \
                patch('mindformers.mindformer_book.logger') as mock_logger:
            MindFormerBook.show_pipeline_support_model_list(task="image_classification")
            mock_logger.info.assert_called_with("Pipeline support model list for %s task is: ", "image_classification")
            mock_print_list.assert_called_once()
            call_args = mock_print_list.call_args[0][0]
            assert call_args == ["model4"]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_show_pipeline_support_model_list_with_invalid_task(self):
        """Test pipeline case when an invalid task is specified"""
        with patch('mindformers.mindformer_book.logger') as mock_logger:
            with pytest.raises(KeyError, match="unsupported task"):
                MindFormerBook.show_pipeline_support_model_list(task="invalid_task")
            mock_logger.info.assert_not_called()
