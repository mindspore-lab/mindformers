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
"""multi cards testcases function utils."""
import enum
from dataclasses import dataclass


class TaskType(enum.Enum):
    """
    Enumeration representing different types of card group tasks.

    Attributes:
        TWO_CARDS_TASK: Represents a task involving two cards.
        FOUR_CARDS_TASK: Represents a task involving four cards.
        EIGHT_CARDS_TASK: Represents a task involving eight cards.
    """
    TWO_CARDS_TASK = 2
    FOUR_CARDS_TASK = 4
    EIGHT_CARDS_TASK = 8


@dataclass
class TaskInfo:
    """
    TaskInfo holds information about a specific task, including its execution time, command, and group type.

    Attributes:
        task_time (int): The time allocated for the task in milliseconds. Defaults to 1000.
        task_command (str): The command associated with the task. Defaults to None.
        group_type (GroupType): The type of group for the task. Defaults to GroupType.EIGHT_CARDS_TASK.
    """
    task_time: int = 1000
    task_command: str = None
    task_type: TaskType = TaskType.EIGHT_CARDS_TASK
