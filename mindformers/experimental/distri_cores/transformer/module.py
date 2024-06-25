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

from mindspore import nn
from mindformers.experimental.distri_cores.create_comm import (
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    get_pp_world_size,
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

class Module(nn.Cell):
    """specific extensions of cell with support for pipelining."""
    def __init__(self, config=None, **kwargs):
        super(Module, self).__init__(**kwargs)
        if config is not None:
            self.config = config
            self.share_embedding_weight = config.share_embedding_weight
            try:
                if get_pp_world_size() > 1:
                    self.first_stage = is_pipeline_first_stage()
                    self.last_stage = is_pipeline_last_stage()
            except AssertionError:
                pass

    def get_embedding_or_head_share_weight(self, layers):
        """get embedding or head share weight"""
        if self.first_stage or self.last_stage:
            weight = layers.trainable_params()
            shared_weight = None
            if len(weight) > 1:
                for cur_weight in weight:
                    if hasattr(cur_weight, 'share') and cur_weight.share:
                        shared_weight = cur_weight
                        break
                if shared_weight is None:
                    raise_err = f"'share_embedding_weight' config is set to True, but all the weight of layer "+\
                                f"'{layers.__class__.__name__}' have no 'share' attribute. "+\
                                f"Please use 'add_attr_for_shared_weight' function to add attribute "+\
                                f"for share weight.\n\n"+\
                                f"For example:\n"+\
                                f"from mindformers.experimental.distri_cores.utils "+\
                                f"import add_attr_for_shared_weight\n"+\
                                f"    self.A = ...\n"+\
                                f"    add_attr_for_shared_weight(self.A)"
                    raise RuntimeError(raise_err)
            else:
                shared_weight = weight[0]
            return shared_weight
        return None

    def sharded_state_dict(self):
        """iterate over the subcells to construct the total sharded state dict"""
        sharded_state_dict = {}
        # Recurse into subcells
        def update_sharded_dict_for_single_cell(subcell):
            nonlocal sharded_state_dict
            if hasattr(subcell, 'sharded_state_dict'):
                sharded_state_dict.update(subcell.sharded_state_dict())
            else:
                if isinstance(subcell, nn.SequentialCell):
                    for inner_layer in subcell:
                        update_sharded_dict_for_single_cell(inner_layer)
                else:
                    sharded_state_dict.update(get_default_dict_for_module(subcell, recurse=True))
        for subcell in self.cells():
            update_sharded_dict_for_single_cell(subcell)
        # Handle params in the current cell
        sharded_state_dict.update(get_default_dict_for_module(self, recurse=False))

        return sharded_state_dict
