# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Infer task class"""

from .infer_config import InferConfig
from .infers.base_infer import BaseInfer
from .infers.text_generator_infer import TextGeneratorInfer


class InferTask:
    """
    Infer Task Factory.
    """
    task_mapping = {
        "text_generation": TextGeneratorInfer,
    }

    @classmethod
    def get_infer_task(cls, task_type: str, task_config: InferConfig, **kwargs) -> BaseInfer:
        """
        Get a infer task obj. by task_type.

        Args:
            task_type (str): task name
            task_config (InferConfig): config of infer task.

        Returns:
            BaseInfer. infer task obj.
        """
        return InferTask.task_mapping[task_type](task_config, **kwargs)

    @classmethod
    def check_task_valid(cls, task_type: str) -> bool:
        """
        check whether task type is valid or not.

        Args:
            task_type: str. task name

        Returns:
            bool
        """
        return task_type in InferTask.support_list()

    @classmethod
    def support_list(cls):
        """
        get infer task support list.

        Returns:
            List[str]
        """
        return InferTask.task_mapping.keys()
