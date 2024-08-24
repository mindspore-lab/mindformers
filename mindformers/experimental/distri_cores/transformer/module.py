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
"""Transformer Module"""

import copy
from collections import OrderedDict
import mindspore as ms
from mindspore import nn
import mindspore.ops as P
from mindformers.tools import logger
from mindformers.experimental.distri_cores.create_comm import (
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    get_pp_world_size,
    get_pp_rank,
    get_embedding_group,
    is_rank_in_embedding_group
)

# Helper function for handling cell's own params
def get_default_dict_for_module(cell, recurse=False):
    state_dict = {}
    for name, param in cell.parameters_dict(recurse=recurse).items():
        shape = param.shape
        shard = tuple([1] * param.ndim)
        state_dict[name] = {'shape': shape, 'shard': shard,
                            'opt_weight_shard_step': 0, 'opt_weight_shard_size': -1}
    return state_dict

class LayerList:
    """ Layer list for building sub model"""
    def __init__(self):
        super().__init__()
        self._model = []

    def __iter__(self):
        return self._model.__iter__()

    def add(self, layers):
        """ add layer to list """
        if isinstance(layers, (list, tuple)):
            for layer in layers:
                self._model.append(layer)
                layer.pipeline_stage = get_pp_rank()
        else:
            self._model.append(layers)
            layers.pipeline_stage = get_pp_rank()

    def reset(self):
        """ reset layer list """
        self._model = []

    def pop(self, index=None):
        """ pop layer from list """
        if index is None:
            layer = self._model.pop()
        else:
            layer = self._model.pop(index)
        return layer

class Module(nn.Cell):
    """specific extensions of cell with support for pipelining."""
    def __init__(self, config=None, **kwargs):
        super(Module, self).__init__(**kwargs)
        self.untie_embeddings_and_output_weights = True
        self.pp_submodel = LayerList()
        self.shared_weight_name_list = []
        if config is not None:
            self.config = config
            self.untie_embeddings_and_output_weights = config.untie_embeddings_and_output_weights
        try:
            if get_pp_world_size() > 1:
                self.pre_process = is_pipeline_first_stage()
                self.post_process = is_pipeline_last_stage()
        except AssertionError:
            pass

    def get_embedding_or_head_share_weight(self, layers):
        """get embedding or head share weight"""
        if self.pre_process or self.post_process:
            weight = layers.trainable_params()
            shared_weight = []
            count = 0
            for cur_weight in weight:
                if hasattr(cur_weight, 'share') and cur_weight.share:
                    shared_weight.append(cur_weight)
                    count += 1
            if count > 1:
                raise RuntimeError("Now, only one weight can be set 'share' attribute in a pipeline stage, "
                                   "please check if 'add_attr_for_shared_weight' function is used correctly.")
            cur_rank_count = ms.Tensor(count, ms.float32)
            all_rank_count = P.AllReduce(group=get_embedding_group())(cur_rank_count)
            if all_rank_count != 0 and cur_rank_count == all_rank_count:
                raise RuntimeError("There is one weight with 'share' attribute in the model, "
                                   "but parameter sharing requires two weights with 'share' attribute in first stage "
                                   "and last stage respectively. "
                                   "Please check if 'add_attr_for_shared_weight' function is used correctly.")
            if shared_weight:
                self.untie_embeddings_and_output_weights = False
            return shared_weight
        return []

    def shared_embedding_or_output_weight(self):
        """ get share weight in first stage and last stage """
        if not is_rank_in_embedding_group(ignore_virtual=True):
            return None

        # pylint: disable=R1705
        if self.pre_process:
            return self.language_model.embedding.word_embeddings.weight
        else:
            if self.untie_embeddings_and_output_weights:
                raise RuntimeError("When 'shared_embedding_or_output_weight()' is called, "
                                   "'untie_embeddings_and_output_weights' should be False.")
            return self.language_model.output_layer.weight

    # pylint: disable=R1710
    def initialize_word_embeddings(self):
        """ initialize share weight under pipeline parallel """
        if not is_rank_in_embedding_group(ignore_virtual=True):
            return None

        if self.config.untie_embeddings_and_output_weights:
            raise RuntimeError("When calling `initialize_word_embeddings()`, "
                               "please ensure `untie_embeddings_and_output_weights=False`.")

        if get_pp_world_size() == 1:
            logger.warning("When calling `initialize_word_embeddings()`, "
                           "there is no need initialize share weight because `pp_stage=1`.")
            return

        shared_weight = None
        if self.pre_process:
            shared_weight = self.language_model.embedding.word_embeddings.weight
        elif self.post_process:
            shared_weight = self.language_model.output_layer.weight
        else:
            return None
        self.shared_weight_name_list.append(shared_weight.name)
        weight_sum = shared_weight.value().sum()
        if weight_sum != 0.0 and self.post_process:
            raise ValueError(f"When embedding's weight share with head layer in pipeline parallel, "
                             f"the weight of head layer must be set as 'zeros' init method. "
                             f" But the weight {shared_weight.name} do not meet the requirement.")
        shared_weight_value = P.AllReduce(group=get_embedding_group())(shared_weight.value())
        if self.post_process:
            P.assign(shared_weight, shared_weight_value)

    # pylint: disable=W0212
    def slice_transformer_layers(self, layers, start_idx, end_idx):
        """ helper function for slice transformer layers without prefix rename """
        if not isinstance(layers, nn.SequentialCell):
            layers = layers.layers
        cells = OrderedDict(list(layers._cells.items())[start_idx:end_idx])
        layers._cells.clear()
        for name, cell in cells.items():
            layers.insert_child_to_cell(name, cell)
            layers._is_dynamic_name.append(False)
        layers.cell_list = list(layers._cells.values())
        return layers

    def sharded_state_dict(self):
        """iterate over the subcells to construct the total sharded state dict"""
        sharded_state_dict = {}
        # Recurse into subcells
        def update_sharded_dict_for_single_cell(subcell):
            nonlocal sharded_state_dict
            if hasattr(subcell, 'sharded_state_dict'):
                sharded_state_dict.update(subcell.sharded_state_dict())
            else:
                if isinstance(subcell, (nn.SequentialCell, nn.CellList)):
                    for inner_layer in subcell:
                        update_sharded_dict_for_single_cell(inner_layer)
                else:
                    sharded_state_dict.update(get_default_dict_for_module(subcell, recurse=True))
        for subcell in self.cells():
            update_sharded_dict_for_single_cell(subcell)
        # Handle params in the current cell
        sharded_state_dict.update(get_default_dict_for_module(self, recurse=False))

        return sharded_state_dict

    def _get_cell_lora_config(self, config, cell_name):
        """get lora config by cell_name"""
        cell_lora_config = copy.deepcopy(config)
        cell_lora_config.update_lora_config(cell_name)
        cell_lora_config = cell_lora_config.lora_config.lora_module
        return cell_lora_config
