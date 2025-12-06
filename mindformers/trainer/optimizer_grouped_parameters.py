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
"""Grouped parameters of optimizer."""

import json
from typing import Optional
from collections import defaultdict
from fnmatch import fnmatch

from mindspore.nn import Cell
from mindspore.nn.learning_rate_schedule import LearningRateSchedule

from mindformers.models import PreTrainedModel
from mindformers.tools.logger import logger
from mindformers.trainer.utils import check_keywords_in_name

# Global list to store grouped parameter names in optimizer
GROUPED_PARAMS = []


def filter_current_stage_parameters(model, model_params):
    """
    Disable gradient updates for parameters that are not included in the
    current training stage (used in PMA training).
    """
    if not model_params:
        raise ValueError(
            "The model got empty trainable parameters, "
            "please check the get_model_parameters method."
        )

    # Iterate over all submodules (cells) and their names
    for _, cell in model.cells_and_names():
        for param in cell.trainable_params():
            if param not in model_params:
                param.requires_grad = False


def _get_gouped_lr_map(model, grouped_lr_scheduler=None):
    """
    Build parameter-to-group and group-to-learning-rate mappings
    based on grouped learning rate scheduler configuration.
    """
    param_group_map = {}
    grouped_lr_map = defaultdict(dict)
    if not grouped_lr_scheduler:
        return param_group_map, grouped_lr_map

    # Map parameter name patterns to group IDs
    group_map = {}
    for group_id, group_dict in enumerate(grouped_lr_scheduler):
        params = group_dict.get('params', None)
        lr_scheduler = group_dict.get('lr_scheduler')
        lr_config = group_dict.get('lr_config')

        # Assign each param pattern to a group
        for param in params:
            group_map[param] = group_id

        # Store LR scheduler instance and its config
        grouped_lr_map[group_id]['instance'] = lr_scheduler
        grouped_lr_map[group_id]['config'] = lr_config

    # Initialize the global grouped parameter tracker
    global GROUPED_PARAMS
    GROUPED_PARAMS = [[] for _ in range(len(grouped_lr_scheduler))]

    # Match actual parameter names to group patterns
    for param in model.trainable_params():
        for grouped_param_name in list(group_map.keys()):
            group_id = group_map.get(grouped_param_name)
            # Match exact or wildcard parameter names
            if grouped_param_name in param.name or fnmatch(param.name, grouped_param_name):
                param_group_map[param.name] = group_id
                GROUPED_PARAMS[group_id].append(param.name)
                break
    for group_id, sub_params in enumerate(GROUPED_PARAMS):
        if not sub_params:
            raise ValueError(
                f"No matched parameters were found for `params` in group {group_id}.")
    return param_group_map, grouped_lr_map


def get_optimizer_grouped_parameters(model: Optional[PreTrainedModel] = None,
                                     weight_decay: float = 0.0,
                                     dynamic_lr_schedule: Optional[LearningRateSchedule] = None,
                                     optimizer_type: str = 'AdamW',
                                     model_params: set = None,
                                     grouped_lr_schedule: dict = None,
                                     layer_scale: bool = False,
                                     layer_decay: float = 1.0,):
    """
    Build optimizer parameter groups with appropriate weight decay, 
    learning rate scheduling, and optional parameter grouping.
    """
    if not isinstance(model, (Cell, PreTrainedModel)):
        raise TypeError(f"model type should be PreTrainedModel, but get {type(model)}")
    if layer_scale:
        raise ValueError("layer_scale is not supported currently.")

    no_wd_params = {}
    no_wd_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        no_wd_params = model.no_weight_decay()
        logger.info(f'Get no weight decay params: {no_wd_params}')
    if hasattr(model, 'no_weight_decay_keywords'):
        no_wd_keywords = model.no_weight_decay_keywords()
        logger.info(f'Get no weight decay keywords: {no_wd_keywords}')

    # set default values if not provided
    if not weight_decay:
        weight_decay = 0.0
    if not layer_decay:
        layer_decay = 1.0

    # PMA optimizer requires filtering of stage-specific parameters
    if optimizer_type in ("PmaAdamW", "FusedPmaAdamW", "Muon"):
        filter_current_stage_parameters(model, model_params)

    # Build mapping from params to LR groups
    param_group_map, lr_scheduler_map = _get_gouped_lr_map(model, grouped_lr_schedule)
    parameter_group_names = {}  # For logging
    parameter_group_vars = {}  # Actual optimizer groups

    # Iterate over trainable parameters and assign them to groups
    for param in model.trainable_params():
        param_name = param.name

        no_wd = (
            len(param.shape) == 1
            or param_name.endswith(".bias")
            or param_name in no_wd_params
            or check_keywords_in_name(param_name, no_wd_keywords)
        )
        if no_wd:
            wd_mul = 0.0
            group_name = 'no_weight_decay'
        else:
            wd_mul = 1.0
            group_name = 'weight_decay'

        group_id = None
        if param_name in param_group_map:
            group_id = param_group_map.get(param_name)
            group_name = f"group_{group_id}_{group_name}"

        # Initialize group if not exists
        if group_name not in parameter_group_names:
            parameter_group_names[group_name] = {
                "weight_decay": wd_mul * weight_decay,
                "params": [],
            }
            parameter_group_vars[group_name] = {
                "weight_decay": wd_mul * weight_decay,
                "params": [],
            }

            # Attach LR scheduler if group-specific
            if group_id is not None:
                cur_lr = lr_scheduler_map[group_id].get('instance')
                cur_lr_config = lr_scheduler_map[group_id].get('config')
                parameter_group_vars[group_name]["lr"] = cur_lr
                parameter_group_names[group_name]["lr_config"] = cur_lr_config

        # Append parameter to its group
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(param.name)
    param_groups = json.dumps(parameter_group_names, indent=2)
    logger.info("Param groups = %s", param_groups)
    return list(parameter_group_vars.values())
