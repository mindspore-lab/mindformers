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
"""Pipeline Cell"""

import inspect
from collections import OrderedDict
from types import MethodType
import numpy as np
import mindspore.ops as P
import mindspore.nn as nn
from mindformers.experimental.distri_cores.create_comm import get_pp_rank, get_pp_world_size, \
                                                                  get_embedding_group, is_pipeline_first_stage, \
                                                                  is_pipeline_last_stage
from mindformers.experimental.distri_cores.utils import get_layer_input_signatures
from mindformers.experimental.distri_cores.transformer import Module

class PipelineCell(Module):
    """ pipeline cell """
    def __init__(
            self,
            model,
            map_dict,
            pipeline_offset=None,
            recompute_interval=0,
            model_customize_staged=False,
            model_forward_func=None
        ):
        super().__init__(auto_prefix=False)
        # check valid
        if get_pp_world_size() < 2:
            raise RuntimeError(f"Under pipeline parallel mode, the pp size must be greater than 1. "
                               f"But now, 'pp_size={get_pp_world_size()}', it do not support using 'PipelineCell'.")
        self.model_customize_staged = model_customize_staged
        self._check_inputs_valid(map_dict, pipeline_offset, model_forward_func)

        # get pp info
        self.first_stage = is_pipeline_first_stage()
        self.last_stage = is_pipeline_last_stage()
        self.stage_id = get_pp_rank()

        for key, value in map_dict.items():
            setattr(self, key, value)
        if not model_customize_staged:
            self.num_layers = len(self.transformer_layers)
            self.num_stages = get_pp_world_size()
            self.set_all_layers_input_signatures()
        else:
            self.model_forward = MethodType(model_forward_func, self)

        self.set_ori_model_input_signatures(model)

        # split model
        self.partition_layers(map_dict, pipeline_offset)

        # set recompute attr for selected layers in each stage
        self.recompute(recompute_interval)

        if hasattr(model, 'share_embedding_weight'):
            self.share_embedding_weight = model.share_embedding_weight
        else:
            self.share_embedding_weight = False
        print("PipelineCell self.share_embedding_weight=", self.share_embedding_weight)

        # if share embedding weight with first stage and last stage,
        # the head layer's weight must be init as all zeros parameter tensor,
        # and using all-reduce to sync init value between first stage and last stage.
        if self.share_embedding_weight and (self.first_stage or self.last_stage):
            self.initialize_shared_embedding_weight(model)

    def _is_current_stage(self, layer, staged_set):
        """ check if the layer has the stage tag and meets the pp rank """
        if hasattr(layer, 'pipeline_stage'):
            stage_idx = layer.pipeline_stage
            staged_set.add(stage_idx)
            if stage_idx == get_pp_rank():
                return True
        return False

    def _check_inputs_valid(self, map_dict, pipeline_offset, model_forward_func):
        """ ensure the map_dict/pipeline_offset/model_forward_func valid"""
        print("[WARNING] When using pipeline parallel, "
              "please make sure that all layers in the 'map_dict' contain the complete model forward.")

        self._check_map_dict(map_dict)

        if pipeline_offset is not None:
            if not isinstance(pipeline_offset, (list, tuple)):
                raise TypeError(f"'pipeline_offset' must be 'list' or 'tuple', but got '{type(pipeline_offset)}'")

            if len(pipeline_offset) != get_pp_world_size():
                raise RuntimeError(f"The length of 'pipeline_offset' must be equal to pipeline stage size, "
                                   f"but got length of 'pipeline_offset': {len(pipeline_offset)}, "
                                   f"pipeline stage size: {get_pp_world_size()}.")

            if not self.model_customize_staged:
                correction_value = len(map_dict["transformer_layers"]) % get_pp_world_size()
                if sum(list(pipeline_offset)) != correction_value:
                    raise RuntimeError(f"The sum of 'pipeline_offset' must be equal to {correction_value}, "
                                       f"but got {sum(list(pipeline_offset))}.")

        if self.model_customize_staged:
            params = inspect.signature(model_forward_func).parameters
            last_param_name = list(params)[-1]
            if not isinstance(map_dict, OrderedDict) \
                or last_param_name != 'recv_data' or params[last_param_name].default is not None:
                raise ValueError(f"If model_customize_staged=True, layers 'embedding' and 'head'"
                                 f" and other customized layers should be included in an OrderedDict."
                                 f" And the method model_forward should be overwritten"
                                 f" with the last param recv_data=None.")

    def _check_map_dict(self, map_dict):
        """ check map_dict valid"""
        layers = map_dict.values()
        for layer in layers:
            if not isinstance(layer, nn.Cell):
                raise TypeError(f"The layers of 'map_dict' must be 'mindspore.nn.Cell', "
                                f"but got type '{type(layer)}'.")

        input_keys = set(map_dict.keys())
        if not self.model_customize_staged:
            self.required_keys = {'preprocess', 'embedding', 'transformer_layers', 'final_norm', 'head', 'loss'}
        else:
            self.required_keys = {'embedding', 'head'}
        missed_keys = self.required_keys - input_keys
        unexpected_keys = input_keys - self.required_keys
        have_unexpected_keys = unexpected_keys and not self.model_customize_staged
        if missed_keys or have_unexpected_keys:
            if missed_keys and have_unexpected_keys:
                msg = f"the key: {list(missed_keys)} are missing now, " \
                      f"and got unexpected key: {list(unexpected_keys)}"
            elif missed_keys:
                msg = f"the key: {list(missed_keys)} are missing now"
            elif have_unexpected_keys:
                msg = f"got unexpected key: {list(unexpected_keys)}"
            raise ValueError(f"1.By default, six layers: 'preprocess', 'embedding',"
                             f" 'transformer_layers', 'final_norm', 'head' and 'loss',"
                             f" must be included in the input param 'map_dict'."
                             f" 2.If model_customize_staged=True, layers 'embedding' and 'head'"
                             f" should be included, and the method model_forward should be overwritten."
                             f" 3.According to 'map_dict', {msg}. Please check 'map_dict' are set correctly.")

        if not self.model_customize_staged and not isinstance(map_dict['transformer_layers'], nn.SequentialCell):
            raise TypeError(f"The type of 'transformer_layers' must be mindspore.nn.SequentialCell, "
                            f"but got type '{type(map_dict['transformer_layers'])}'.")

    def set_ori_model_input_signatures(self, model):
        """ set full model's input argument """
        params = inspect.signature(model.construct).parameters
        self.model_args_index = list(params.keys())
        self.model_args_default_value = [value.default for value in params.values()]
        self.key_need_to_be_merged = []

    def set_all_layers_input_signatures(self):
        """ define each layer input argument key for 'model_forward' under pipeline parallel """
        self.preprocess_input_signatures = get_layer_input_signatures(self.preprocess.construct)
        self.embedding_input_signatures = get_layer_input_signatures(self.embedding.construct)
        self.transformer_layer_input_signatures = get_layer_input_signatures(self.transformer_layers[0].construct)
        self.final_norm_input_signatures = get_layer_input_signatures(self.final_norm.construct)
        self.head_input_signatures = get_layer_input_signatures(self.head.construct)
        self.loss_input_signatures = get_layer_input_signatures(self.loss.construct)

    def _get_bounds(self, stage_layers_list):
        if self.stage_id == 0:
            start_idx = 0
            end_idx = stage_layers_list[0]
        else:
            start_idx = np.sum(stage_layers_list[:self.stage_id])
            end_idx = np.sum(stage_layers_list[:self.stage_id+1])
        return (start_idx, end_idx)

    def _set_bounds(self, start=None, stop=None):
        self._local_start = start
        self._local_stop = stop

    def partition_layers(self, map_dict, offset=None):
        """split transformer layers as a partial block """
        if not self.model_customize_staged:
            stage_layers_list = np.array([self.num_layers // self.num_stages] * self.num_stages)
            remain_layer_nums = self.num_layers - np.sum(stage_layers_list)
            for i in range(remain_layer_nums):
                stage_layers_list[-i-2] += 1

            if offset is not None:
                stage_layers_list += np.array(offset)

            self.stage_layers_list = stage_layers_list
            start_idx, end_idx = self._get_bounds(stage_layers_list)
            self._set_bounds(start_idx, end_idx)

            # pylint: disable=W0212
            def split_transformer_layers(seq, start_idx, end_idx):
                """ helper function for splitting transformer layers without prefix rename """
                cells = OrderedDict(list(seq._cells.items())[start_idx:end_idx])
                seq._cells.clear()
                for name, cell in cells.items():
                    seq.insert_child_to_cell(name, cell)
                    seq._is_dynamic_name.append(False)
                seq.cell_list = list(seq._cells.values())
                return seq
            self.transformer_layers = split_transformer_layers(self.transformer_layers, start_idx, end_idx)

        else:
            staged_set = set()
            for key, cur_layer in map_dict.items():
                if isinstance(cur_layer, nn.SequentialCell):
                    filter_layer = nn.SequentialCell()
                    mask = [self._is_current_stage(inner_layer, staged_set) for inner_layer in cur_layer]
                    for inner_layer, is_cur_stage in zip(cur_layer, mask):
                        if is_cur_stage:
                            filter_layer.append(inner_layer)
                    setattr(self, key, filter_layer)
                elif not self._is_current_stage(cur_layer, staged_set):
                    setattr(self, key, None)
            if len(staged_set) != get_pp_world_size():
                missed_stage = set(range(get_pp_world_size())) - staged_set
                raise ValueError(f"'{type(self).__name__}' pipeline stage {list(missed_stage)} is empty unexpectedly. "
                                 f"Check if the pipeline setup is correct and match the pp world size.")
        self.reset_useless_layers_for_current_stage(map_dict)

        # check cur stage layer
        if not self.trainable_params():
            raise RuntimeError("Current stage have no any parameters, "
                               "please check each stage has at least one layer with weight.")

    def reset_useless_layers_for_current_stage(self, map_dict):
        """delete useless layers for current stage"""
        if not self.model_customize_staged:
            if self.first_stage:
                self.final_norm = None
                self.head = None
                self.loss = None
            else:
                self.embedding = None
                if not self.last_stage:
                    self.final_norm = None
                    self.head = None
                    self.loss = None
        map_dict.clear()

    def initialize_shared_embedding_weight(self, model):
        """ if share_embedding_weight=True, initialize embedding/head layer's weight """
        layer = self.embedding if self.first_stage else self.head
        shared_weight = model.get_embedding_or_head_share_weight(layer)
        self.shared_weight_name = shared_weight.name
        weight_sum = shared_weight.value().sum()
        if weight_sum != 0.0 and self.last_stage:
            raise ValueError(f"When embedding layer's weight share with head layer in pipeline,"
                             f" the weight of head layer must be set as 'zeros' init method."
                             f" Here the weight {shared_weight.name} does not meet the requirement.")
        embedding_weight = P.AllReduce(group=get_embedding_group())(shared_weight.value())

        if self.last_stage:
            P.assign(shared_weight, embedding_weight)

    def recompute(self, recompute_interval):
        """ set recompute attr for specific layers """
        if recompute_interval == 0:
            return
        if self.model_customize_staged:
            raise ValueError("Recompute for model_customize_staged mode is not support now.")
        for layer_id in range(self._local_start, self._local_stop, recompute_interval):
            self.transformer_layers[layer_id].recompute()

    def get_layer_inputs(self, layer_input_signatures, public_tensors):
        """ get inputs based on layer signatures, except for 'input_ids' """
        layer_inputs = {}
        if layer_input_signatures is None:
            return layer_inputs

        for key in layer_input_signatures:
            if isinstance(public_tensors, tuple):
                if key in self.model_args_index:
                    idx = self.model_args_index.index(key)
                    layer_inputs[key] = public_tensors[idx]
            else:
                if key in public_tensors:
                    layer_inputs[key] = public_tensors[key]
        return layer_inputs

    def input_tensor_merge(self, public_tensors, inputs):
        """  make sure the public_tensors have fields needed """
        if self.key_need_to_be_merged:
            for key in self.key_need_to_be_merged:
                idx = self.model_args_index.index(key)
                public_tensors[key] = inputs[idx]
            return public_tensors

        for idx, key in enumerate(self.model_args_index):
            if key in public_tensors:
                continue
            self.key_need_to_be_merged.append(key)
            public_tensors[key] = inputs[idx]
        return public_tensors

    # pylint: disable=E0202
    def model_forward(self, *inputs, recv_data=None):
        """ model pipeline forward process """
        if self.model_customize_staged:
            raise ValueError(f"If model_customize_staged=True, the method model_forward should be overwritten"
                             f" with the last param 'recv_data=None'.")
        outputs = None
        # The preprocess layer output must be a dict, which contain full inputs for all of next layers
        preprocess_inputs = self.get_layer_inputs(self.preprocess_input_signatures, inputs)
        public_tensors = self.preprocess(inputs[0], **preprocess_inputs)

        # make sure the public_tensors have fields needed
        public_tensors = self.input_tensor_merge(public_tensors, inputs)

        input_ids = public_tensors.pop('input_ids')

        # first stage layer
        if self.first_stage:
            embedding_inputs = self.get_layer_inputs(self.embedding_input_signatures, public_tensors)
            # pylint: disable=E1102
            outputs, _ = self.embedding(input_ids, **embedding_inputs)

        # transformer layers
        transformer_layer_inputs = self.get_layer_inputs(self.transformer_layer_input_signatures,
                                                         public_tensors)
        # if recv_data if not None, correct the first input of transformer layer
        if recv_data is not None:
            outputs = recv_data

        if self.transformer_layers:
            for layer in self.transformer_layers:
                outputs = layer(outputs, **transformer_layer_inputs)

        # last stage layers
        if self.last_stage:
            # final norm layer
            final_norm_layer_inputs = self.get_layer_inputs(self.final_norm_input_signatures, public_tensors)
            outputs = self.final_norm(outputs, **final_norm_layer_inputs)
            # head layer
            head_layer_inputs = self.get_layer_inputs(self.head_input_signatures, public_tensors)
            outputs = self.head(outputs, **head_layer_inputs)
            # loss
            loss_layer_inputs = self.get_layer_inputs(self.loss_input_signatures, public_tensors)
            outputs = self.loss(outputs, **loss_layer_inputs)

        return outputs

    def construct(self, *inputs, recv_data=None):
        # When using pipeline parallel, the model forward will be replace by this function,
        # which execute different operation or layer in different stage.
        # Now, pipeline process function default to get the gradient of 'recv_data'.
        # Therefore, please ensure 'recv_data' is the last input argument of construct function for bprop correctly.
        # pylint: disable=E1102
        return self.model_forward(*inputs, recv_data=recv_data)
