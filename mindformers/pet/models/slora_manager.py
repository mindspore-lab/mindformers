# Copyright 2024 Huawei Technologies Co., Ltd
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
"""LoRA Model Manager."""

import abc
import re
import json
import os
from typing import Tuple

from mindspore import Tensor
from mindspore.nn import Cell

from mindformers.modules.layers import Linear
from mindformers.pet.pet_config import SLoraConfig
from mindformers.pet.tuners.slora_adapter import SLoraLinear
from mindformers.tools.logger import logger


def init_slora_config(adapter_path: str):
    """read S-LoRA config"""
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"The adapter_path must be correct, but get {adapter_path}")
    with open(adapter_path, 'r') as file:
        path_dict = json.load(file)

    num = len(path_dict)
    max_rank = 0
    target_modules_set = set()
    for _, slora_path in path_dict.items():
        config_path = os.path.join(slora_path, "adapter_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"{config_path} does not exist, "
                             f"please pass a valid the slora path in config.adapter_path file.")
        with open(config_path, 'r') as file:
            config = json.load(file)
        max_rank = max(max_rank, int(config["r"]))
        target_modules_set.update(config["target_modules"])
    target_modules_list = [".*" + module for module in list(target_modules_set)]
    target_modules = '|'.join(target_modules_list)

    # read alpha, dropout from the first lora.
    config_path = os.path.join(path_dict[list(path_dict.keys())[0]], "adapter_config.json")
    with open(config_path, 'r') as file:
        config = json.load(file)
    alpha = int(config["lora_alpha"])
    dropout = float(config["lora_dropout"])
    return SLoraConfig(target_modules, num, max_rank, alpha, dropout, lora_names=list(path_dict.keys()))


class SLoraManager(abc.ABC):
    """
    LoRA Model Manager to wrap layers
    """

    def __init__(self, config):
        self.slora_config = init_slora_config(config.model_config.pet_config.adapter_path)
        self.common_lora_cell: dict = {}

    def build(self):
        self.common_lora_cell[Linear] = SLoraLinear

    def apply(self, network: Cell):
        self.build()
        self.process(network, network.adapter_ids)
        network.update_parameters_name()
        network.config.pet_config['lora_names'] = self.slora_config.lora_names

    def process(self, root: Cell, adapter_ids: Tensor, name_prefix: str = "root"):
        """Recursively search for target layers in the network and wrap them"""
        if root is None:
            return root
        for name, cell in root.name_cells().items():
            full_cell_name = f"{name_prefix}.{name}"
            new_cell, is_end_point = self.process_cell(full_cell_name, cell, adapter_ids)
            if new_cell is not cell:
                root.insert_child_to_cell(name, new_cell)
            if not is_end_point:
                _ = self.process(new_cell, adapter_ids, full_cell_name)
        return root

    def process_cell(self, cell_name: str, cell: Cell, adapter_ids: Tensor) -> Tuple[Cell, bool]:
        """Match the layer with target list to get the layer with lora"""
        if not isinstance(cell, Cell):
            return cell, True
        if re.match(self.slora_config.lora_target_modules, cell_name):
            wrap_lora = self.common_lora_cell.get(type(cell))
            if wrap_lora:
                logger.info(f"Apply LoRA to {cell_name}.")
                new_cell = wrap_lora(cell, adapter_ids, self.slora_config)
                new_cell.shard()
                return new_cell, True
        return cell, False
