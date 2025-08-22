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
"""layout config"""
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindspore.parallel import Layout


class LayoutManager:
    """
    Singleton class to manage and provide access to parallel layout configurations.

    This class ensures a single global layout configuration is initialized once and
    shared across all modules. It supports two types of parallel layouts:
    1. DP/CP/TP/EP (Data/Context/Tensor/Expert Parallel)
    2. DP/CP/TP (Data/Context/Tensor Parallel)

    Class Variables:
        _instance (LayoutManager): Singleton instance holder.
        _layout (Layout): Current parallel layout configuration.
        _layout_type (str): Type of active layout ('dp_cp_tp_ep' or 'dp_cp_tp').
        _config (dict): Stored parallel configuration parameters.
    """
    _instance = None
    _layout: Layout = None
    _layout_type: str = None
    _config = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __call__(self, *args, **kwargs):
        """Make the instance callable to return the current layout."""
        comm_group = {
            "cp_tp": ("cp", "tp"),
            "cp_dp": ("cp", "dp"),
            "dp_cp": ("dp", "cp")
        }

        if self._layout is None:
            raise RuntimeError(
                "Layout not initialized. Call `init_dp_cp_tp_layout()` first.")

        if not args:
            return self._layout()

        layout_strategy = tuple(comm_group.get(item, item) for item in args)
        return self._layout(*layout_strategy)

    def get_parallel_config(self, config: TransformerConfig):
        """
        Internal method to get and store parallel configuration parameters.

        Args:
            config (TransformerConfig): Configuration object containing:
                - data_parallel_size (int): Data parallel group size.
                - tensor_model_parallel_size (int): Tensor parallel group size.
                - context_parallel_size (int): Context parallel group size.
                - expert_model_parallel_size (int): Expert parallel group size (optional).

        Returns:
            dict: Stored configuration with keys: 'dp', 'tp', 'cp', 'ep'.
        """
        if self._config is None:
            self._config = {
                'dp': config.data_parallel_size if config.data_parallel_size is not None else 1,
                'tp': config.tensor_model_parallel_size if config.tensor_model_parallel_size is not None else 1,
                'cp': config.context_parallel_size if config.context_parallel_size is not None else 1
            }
        return self._config

    def init_layout(self, config: TransformerConfig, layout_type: str = "dp_cp_tp"):
        """
        Unified initialization entry point
        Args:
            layout_type:
                - "dp_cp_tp" : Data/Context/Tensor Parallel
                - "dp_cp_tp_ep" : Requires MoeLayoutManager
        """
        if layout_type == "dp_cp_tp":
            return self.init_dp_cp_tp_layout(config)

        if layout_type == "dp_cp_tp_ep":
            raise ValueError("Use MoeLayoutManager for moe_layout")

        raise ValueError(f"Unknown layout type: {layout_type}")

    def init_dp_cp_tp_layout(self, config: TransformerConfig):
        """
        Initialize layout with Data/Context/Tensor Parallel (DP/CP/TP).

        Args:
            config (TransformerConfig): Must contain:
                - data_parallel_size (int)
                - tensor_model_parallel_size (int)
                - context_parallel_size (int)

        Returns:
            Layout: Configured layout with device matrix (dp, cp, tp).
        """
        if self._layout is not None and self._layout_type == "dp_cp_tp":
            return self._layout

        parallel_config = self.get_parallel_config(config)

        dev_mat = (parallel_config['dp'], parallel_config['cp'], parallel_config['tp'])
        self._layout = Layout(dev_mat, ("dp", "cp", "tp"))
        self._layout_type = "dp_cp_tp"
        return self._layout

    @staticmethod
    def to_tuple_strategy(layout_):
        """convert layout tensor_map to strategy in tuple"""
        layout_dict = layout_.to_dict()
        device_matrix = layout_dict["device_matrix"]
        tensor_map = layout_dict["tensor_map"]
        alias_name = layout_dict["alias_name"]

        converted = []
        for tensor_map_item in tensor_map:
            if tensor_map_item == -1:
                converted.append(1)
            else:
                index = len(alias_name) - 1 - tensor_map_item
                converted.append(device_matrix[index])
        return tuple(converted)


class MoeLayoutManager(LayoutManager):
    """
    MoE-specific layout manager (inherits from LayoutManager) with expert parallel (EP) support.

    Extended capabilities:
    1. Supports hybrid DP/CP/TP/EP parallelization
    2. Provides MoE-specific communication group mappings
    """
    _instance = None
    _layout: Layout = None
    _layout_type: str = None
    _config = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __call__(self, *args, **kwargs):
        """MoE-specific communication group mapping"""
        moe_comm_group = {
            "cp_tp": ("cp", "tp"),
            "cp_dp": ("cp", "dp_ex_ep", "ep"),
            "dp": ("dp_ex_ep", "ep"),
            "dp_cp": ("dp_ex_ep", "ep", "cp"),
            "dp_cp_tp": ("dp_ex_ep", "cp", "tp"),
            "dp_ep_cp_tp": ("dp_ex_ep", "ep", "cp", "tp")
        }

        if self._layout is None:
            raise RuntimeError("Layout not initialized. Call `init_dp_ep_cp_tp_layout()` first.")

        moe_layout_strategy = tuple(moe_comm_group.get(item, item) for item in args)
        return self._layout(*moe_layout_strategy)

    def get_parallel_config(self, config: TransformerConfig):
        if self._config is None:
            self._config = {
                **super().get_parallel_config(config),
                'ep': config.expert_model_parallel_size if config.expert_model_parallel_size is not None else 1
            }
        return self._config

    def init_layout(self, config: TransformerConfig, layout_type: str = "dp_cp_tp_ep"):
        """
        Unified initialization entry point (with MoE extensions)
        Args:
            layout_type:
                - "dp_cp_tp" : Fallback to base mode
                - "dp_cp_tp_ep" : Default MoE mode
        """
        if layout_type == "dp_cp_tp_ep":
            return self.init_dp_cp_tp_ep_layout(config)

        if layout_type == "dp_cp_tp":
            return super().init_dp_cp_tp_layout(config)

        raise ValueError(f"Unknown layout type: {layout_type}")

    def init_dp_cp_tp_ep_layout(self, config):
        """
        Initialize layout with Data/Context/Tensor/Expert Parallel (DP/CP/TP/EP).

        Args:
            config (TransformerConfig): Must contain:
                - data_parallel_size (int)
                - tensor_model_parallel_size (int)
                - context_parallel_size (int)
                - expert_model_parallel_size (int)

        Returns:
            Layout: Configured layout with device matrix (dp//ep, ep, cp, tp).
        """
        if self._layout is not None and self._layout_type == "dp_cp_tp_ep":
            return self._layout

        parallel_config = self.get_parallel_config(config)
        ep = parallel_config['ep']
        dp_ex_ep = parallel_config['dp'] // ep

        dev_mat = (dp_ex_ep, ep, parallel_config['cp'], parallel_config['tp'])
        self._layout = Layout(dev_mat, ("dp_ex_ep", "ep", "cp", "tp"))
        self._layout_type = "dp_cp_tp_ep"
        return self._layout


layout = LayoutManager()
layout_moe = MoeLayoutManager()
