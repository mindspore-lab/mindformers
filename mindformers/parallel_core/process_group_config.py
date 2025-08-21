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
# ======================

"""Dataclass for model communication process groups."""
from dataclasses import dataclass, field, fields
from typing import List, Optional

from mindformers.parallel_core.inference import parallel_state
from mindformers.parallel_core.inference.parallel_state import ProcessGroup


@dataclass
class ModelCommProcessGroups:
    """
    Dataclass to hold model communication groups.

    Args:
        globals (ProcessGroup): World Group.
        tp (ProcessGroup): Tensor Model Parallel Group.
        dp (ProcessGroup): Data Parallel Group.
        pp (ProcessGroup): Pipeline Model Parallel Group.
        moe_tp (ProcessGroup): MoE Tensor Parallel Group.
        moe_ep (ProcessGroup): MoE Expert Parallel Group.
        tpdp (ProcessGroup): Tensor and Data Parallel Group.
    """
    # _TENSOR_MODEL_PARALLEL_GROUP
    globals: ProcessGroup = field(init=False)

    # _TENSOR_MODEL_PARALLEL_GROUP
    tp: ProcessGroup = field(init=False)

    # _DATA_PARALLEL_GROUP
    dp: ProcessGroup = field(init=False)

    # _PIPELINE_MODEL_PARALLEL_GROUP
    pp: ProcessGroup = field(init=False)

    # _MOE_TENSOR_MODEL_PARALLEL_GROUP
    moe_tp: ProcessGroup = field(init=False)

    # _MOE_EXPERT_MODEL_PARALLEL_GROUP
    moe_ep: ProcessGroup = field(init=False)

    # _TENSOR_AND_DATA_PARALLEL_GROUP
    tpdp: ProcessGroup = field(init=False)

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key in [field.name for field in fields(self)]:
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Invalid field name: {key}. Available fields are: {[field.name for field in fields(self)]}")

    @classmethod
    def use_parallel_state_groups(cls, required_groups: Optional[List[str]] = None) -> "ModelCommProcessGroups":
        """
        Use the default parallel state groups to create an instance of ModelCommProcessGroups.
        Args:
            required_groups (Optional[List[str]]): List of group names to initialize.
                If None, all available groups will be used. Available groups are:

                - 'globals': WorldGroup
                - 'tp': Tensor Model Parallel Group
                - 'dp': Data Parallel Group
                - 'pp': Pipeline Model Parallel Group
                - 'moe_tp': MoE Tensor Parallel Group
                - 'moe_ep': MoE Expert Parallel Group
                - 'tpdp': Tensor and Data Parallel Group
        """
        # Get all available groups from the class
        all_groups = {field.name for field in fields(cls)}

        # If no specific groups are required, use all available groups
        if required_groups is None:
            required_groups = list(all_groups)

        # Validate the required groups
        invalid_groups = set(required_groups) - all_groups
        if invalid_groups:
            raise ValueError(f"Invalid group names: {invalid_groups}")

        # Mapping of attribute names to their initialization methods
        group_to_init_method = {
            'globals': parallel_state.get_world_group,
            'tp': parallel_state.get_tensor_model_parallel_group,
            'dp': parallel_state.get_data_parallel_group,
            'pp': parallel_state.get_pipeline_model_parallel_group,
            'moe_tp': parallel_state.get_moe_tensor_parallel_group,
            'moe_ep': parallel_state.get_moe_expert_parallel_group,
            'tpdp': parallel_state.get_tensor_and_data_parallel_group,
        }

        # Create an instance of the class with the initialized groups
        init_dict = {
            group: group_to_init_method[group]() for group in group_to_init_method if group in required_groups
        }

        return cls(**init_dict)

    @classmethod
    def get_default_model_comm_pgs(cls) -> "ModelCommProcessGroups":
        """
        Create an single instance of ModelCommProcessGroups.
        """
        # Mapping of attribute names to their initialization methods
        group_init_method = {
            'globals': ProcessGroup(),
            'tp': ProcessGroup(),
            'dp': ProcessGroup(),
            'pp': ProcessGroup(),
            'moe_tp': ProcessGroup(),
            'moe_ep': ProcessGroup(),
            'tpdp': ProcessGroup(),
        }

        return cls(**group_init_method)


default_model_comm_pgs = ModelCommProcessGroups.get_default_model_comm_pgs()
