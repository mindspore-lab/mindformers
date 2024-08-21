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
import numpy as np
import mindspore as ms
import mindspore.ops as P
import mindspore.nn as nn
from mindformers.experimental.distri_cores.create_comm import get_pp_rank, get_pp_world_size, \
                                                                  get_embedding_group, is_pipeline_first_stage, \
                                                                  is_pipeline_last_stage
from mindformers.experimental.distri_cores.transformer import Module

class PipelineCell(Module):
    """ pipeline cell """
    def __init__(
            self,
            model,
            map_dict=None,
            pipeline_offset=None,
            recompute_interval=0,
            model_customize_staged=False
        ):
        super().__init__(auto_prefix=False)
        self.set_ori_model_input_signatures(model)
        self.new_model = hasattr(model, "model_key")
        self.shared_weight_name_list = model.shared_weight_name_list
        self.untie_embeddings_and_output_weights = model.untie_embeddings_and_output_weights

        # if pp_size=1 or the model is imported from the distri_model directory, do not init anything
        if get_pp_world_size() == 1 or self.new_model:
            self.model = model
            return

        # check valid
        self.model_customize_staged = model_customize_staged
        self.check_inputs_valid(map_dict, pipeline_offset)

        # get pp info
        self.first_stage = is_pipeline_first_stage()
        self.last_stage = is_pipeline_last_stage()
        self.stage_id = get_pp_rank()

        if not model_customize_staged:
            for key, value in map_dict.items():
                setattr(self, key, value)
            self.num_layers = len(self.transformer_layers)
            self.num_stages = get_pp_world_size()
            self.set_all_layers_input_signatures()

        # set public layer
        self.get_public_layer(model)

        # split model
        self.partition_layers(map_dict, model, pipeline_offset)

        # set recompute attr for selected layers in each stage
        self.recompute(recompute_interval)

        # if share embedding weight in first stage and last stage,
        # the head layer's weight must be init as all zeros parameter tensor,
        # and using all-reduce to sync init value between first stage and last stage.
        if self.first_stage or self.last_stage:
            self.initialize_shared_embedding_weight()

    def check_inputs_valid(self, map_dict, pipeline_offset):
        """ ensure the map_dict/pipeline_offset valid"""
        print("[WARNING] When using pipeline parallel, "
              "please make sure that all layers in the 'map_dict' contain the complete model forward.")

        if not self.model_customize_staged:
            self.check_map_dict(map_dict)

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

    def check_map_dict(self, map_dict):
        """ check map_dict valid"""
        if map_dict is None or not isinstance(map_dict, dict):
            raise RuntimeError(f"When `model_customize_staged=False`, the argument of 'map_dict' must be dict.")

        layers = map_dict.values()
        for layer in layers:
            if not isinstance(layer, nn.Cell):
                raise TypeError(f"The layers of 'map_dict' must be 'mindspore.nn.Cell', "
                                f"but got type '{type(layer)}'.")

        input_keys = set(map_dict.keys())
        required_keys = {'public_layer', 'embedding', 'transformer_layers', 'final_norm', 'head', 'loss'}
        missed_keys = required_keys - input_keys
        unexpected_keys = input_keys - required_keys
        have_unexpected_keys = unexpected_keys
        if missed_keys or have_unexpected_keys:
            if missed_keys and have_unexpected_keys:
                msg = f"the key: {list(missed_keys)} are missing now, " \
                      f"and got unexpected key: {list(unexpected_keys)}"
            elif missed_keys:
                msg = f"the key: {list(missed_keys)} are missing now"
            elif have_unexpected_keys:
                msg = f"got unexpected key: {list(unexpected_keys)}"
            raise ValueError(f"1.By default, six layers: 'public_layer', 'embedding',"
                             f" 'transformer_layers', 'final_norm', 'head' and 'loss',"
                             f" must be included in the input param 'map_dict'."
                             f" 2.According to 'map_dict', {msg}. Please check 'map_dict' are set correctly.")

        if not isinstance(map_dict['transformer_layers'], nn.SequentialCell):
            raise TypeError(f"The type of 'transformer_layers' must be mindspore.nn.SequentialCell, "
                            f"but got type '{type(map_dict['transformer_layers'])}'.")

    def set_input_tensor(self, input_tensor):
        """ set input tensor for model"""
        self.model.set_input_tensor(input_tensor)

    def set_ori_model_input_signatures(self, model):
        """ set full model's input argument """
        params = inspect.signature(model.construct).parameters
        self.model_args_index = list(params.keys())
        self.model_args_default_value = [value.default for value in params.values()]
        self.key_need_to_be_merged = []

    def set_all_layers_input_signatures(self):
        """ define each layer input argument key for 'model_forward' under pipeline parallel """
        self.public_layer_input_signatures = self.get_layer_input_signatures(self.public_layer.construct, True)
        self.embedding_input_signatures = self.get_layer_input_signatures(self.embedding.construct)
        self.transformer_layer_input_signatures = self.get_layer_input_signatures(self.transformer_layers[0].construct)
        self.final_norm_input_signatures = self.get_layer_input_signatures(self.final_norm.construct)
        self.head_input_signatures = self.get_layer_input_signatures(self.head.construct)
        self.loss_input_signatures = self.get_layer_input_signatures(self.loss.construct)

    def get_bounds(self, stage_layers_list):
        """ get transformer_layer silce index """
        if self.stage_id == 0:
            start_idx = 0
            end_idx = stage_layers_list[0]
        else:
            start_idx = np.sum(stage_layers_list[:self.stage_id])
            end_idx = np.sum(stage_layers_list[:self.stage_id+1])
        return (start_idx, end_idx)

    def partition_layers(self, map_dict, model, offset=None):
        """split transformer layers as a partial block """
        if not self.model_customize_staged:
            stage_layers_list = np.array([self.num_layers // self.num_stages] * self.num_stages)
            remain_layer_nums = self.num_layers - np.sum(stage_layers_list)
            for i in range(remain_layer_nums):
                stage_layers_list[-i-2] += 1

            if offset is not None:
                stage_layers_list += np.array(offset)

            self.stage_layers_list = stage_layers_list
            start_idx, end_idx = self.get_bounds(stage_layers_list)
            self.transformer_layers = self.transformer_layers[start_idx:end_idx]
            self.reset_useless_layers_for_current_stage(map_dict)
        else:
            if not hasattr(model, "build_pipeline_model"):
                raise NotImplementedError(f"When using pipeline parallel, the model '{type(model).__name__}' must "
                                          f"implement the `build_pipeline_model` method for building pipeline model.")
            model.build_pipeline_model()
            self.submodel = nn.CellList(auto_prefix=False)
            for layer in model.pp_submodel:
                self.submodel.append(layer)

        # check current stage
        if not self.trainable_params():
            raise RuntimeError("Current stage have no any parameters, "
                               "please check each stage has at least one layer with weight.")

    def get_public_layer(self, model):
        """ get public layer from model """
        sub_cells_dict = model.__dict__['_cells']
        self.public_layer = None
        for key, layer in sub_cells_dict.items():
            cur_key = 'model.' + key
            if hasattr(layer, "is_public_layer") and layer.is_public_layer:
                # pylint: disable=W0123
                self.public_layer = eval(cur_key)

        if self.public_layer is None:
            raise NotImplementedError(f"Please implement 'public_layer' under pipeline parallel, "
                                      f"or the custom 'public_layer' is not based on the 'BasePublicLayer' class.")

    def reset_useless_layers_for_current_stage(self, map_dict):
        """delete useless layers for current stage"""
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

    def initialize_shared_embedding_weight(self):
        """ if there are share weights, initialize shared layer's weight """
        if self.model_customize_staged:
            layer = self.submodel
        else:
            layer = self.embedding if self.first_stage else self.head
        shared_weight_list = self.get_embedding_or_head_share_weight(layer)
        if shared_weight_list:
            param_non_zeros_count = 0
            param_non_zeros_index = 0
            param_non_zeros_list = []
            self.shared_weight_name_list = []
            for idx, shared_weight in enumerate(shared_weight_list):
                self.shared_weight_name_list.append(shared_weight.name)
                weight_sum = shared_weight.value().sum()
                if weight_sum != 0.0:
                    param_non_zeros_list.append(shared_weight.name)
                    param_non_zeros_count += 1
                    param_non_zeros_index = idx

            if param_non_zeros_count > 1 and self.first_stage:
                raise RuntimeError(f"When there are multiple shared weights in the model,"
                                   f" only one weight that can be not set as 'zeros' method in the first stage."
                                   f" But the weights '{param_non_zeros_list}' does not meet the requirement.")

            if param_non_zeros_count > 0 and self.last_stage:
                raise ValueError(f"When embedding layer's weight share with head layer in pipeline,"
                                 f" the weight of head layer must be set as 'zeros' init method in the last stage."
                                 f" Here the weights '{param_non_zeros_list}' does not meet the requirement.")

            if self.first_stage:
                shared_weight = shared_weight_list.pop(param_non_zeros_index)
                for zeros_param in shared_weight_list:
                    P.assign(zeros_param, shared_weight.value())

            cur_rank_shared_num = ms.Tensor(len(shared_weight_list), ms.float32)
            all_shared_num = P.AllReduce(group=get_embedding_group())(cur_rank_shared_num)
            if self.first_stage:
                last_stage_param_need_be_assign_num = all_shared_num - cur_rank_shared_num
            elif self.last_stage:
                last_stage_param_need_be_assign_num = cur_rank_shared_num

            for i in range(int(last_stage_param_need_be_assign_num.tolist())):
                if self.first_stage:
                    shared_weight_value = shared_weight.value()
                elif self.last_stage:
                    shared_weight = shared_weight_list[i]
                    shared_weight_value = shared_weight.value()
                shared_weight_value = P.AllReduce(group=get_embedding_group())(shared_weight_value)
                if self.last_stage:
                    P.assign(shared_weight, shared_weight_value)

    def recompute(self, recompute_interval):
        """ set recompute attr for specific layers """
        if recompute_interval == 0:
            return
        if self.model_customize_staged:
            raise ValueError("Recompute for model_customize_staged mode is not support now.")
        for layer_id in range(self._local_start, self._local_stop, recompute_interval):
            self.transformer_layers[layer_id].recompute()

    def get_layer_inputs(self, layer_input_signatures, public_tensors):
        """ get inputs based on layer signatures """
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

    def get_layer_input_signatures(self, func, get_input_ids=False):
        """
        if get_input_ids=False:
        except for 'input_ids', get all of input signatures for layer's construct.
        """
        params = inspect.signature(func).parameters
        layer_input_signatures = list(params.keys())
        if not get_input_ids:
            # remove 'input_ids' argument
            layer_input_signatures = layer_input_signatures[1:]
        if not layer_input_signatures:
            layer_input_signatures = None
        return layer_input_signatures

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
        outputs = None
        logits = None
        # The public layer output must be a dict, which contain full inputs for all of next layers
        public_layer_inputs = self.get_layer_inputs(self.public_layer_input_signatures, inputs)
        public_tensors = self.public_layer(**public_layer_inputs)

        # make sure the public_tensors have fields needed
        public_tensors = self.input_tensor_merge(public_tensors, inputs)

        input_ids = public_tensors.pop('input_ids')

        # first stage layer
        if self.first_stage:
            embedding_inputs = self.get_layer_inputs(self.embedding_input_signatures, public_tensors)
            # pylint: disable=E1102
            outputs = self.embedding(input_ids, **embedding_inputs)

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
            if self.final_norm:
                final_norm_layer_inputs = self.get_layer_inputs(self.final_norm_input_signatures, public_tensors)
                outputs = self.final_norm(outputs, **final_norm_layer_inputs)
            # head layer
            head_layer_inputs = self.get_layer_inputs(self.head_input_signatures, public_tensors)
            outputs = self.head(outputs, **head_layer_inputs)
            logits = outputs
            # loss
            loss_layer_inputs = self.get_layer_inputs(self.loss_input_signatures, public_tensors)
            outputs = self.loss(outputs, **loss_layer_inputs)
            return outputs, P.stop_gradient(logits)
        return outputs

    # pylint: disable=E0202
    def model_customize_forward(self, *inputs, recv_data=None):
        """ model pipeline customize forward process """
        outputs = None
        logits = None
        # The public layer output must be a dict, which contain full inputs for all of next layers
        public_layer_input_signatures = self.get_layer_input_signatures(self.public_layer.construct, True)
        public_layer_inputs = self.get_layer_inputs(public_layer_input_signatures, inputs)
        public_tensors = self.public_layer(**public_layer_inputs)

        # make sure the public_tensors have fields needed
        public_tensors = self.input_tensor_merge(public_tensors, inputs)

        outputs = public_tensors.pop('input_ids')

        # if recv_data if not None, correct the first input of transformer layer
        if recv_data is not None:
            outputs = recv_data

        for layer in self.submodel:
            layer_inputs_signatures = self.get_layer_input_signatures(layer.construct)
            layer_inputs = self.get_layer_inputs(layer_inputs_signatures, public_tensors)
            outputs = layer(outputs, **layer_inputs)

            # if 'layer' is not a complete model for current stage, collect 'logits' result from head layer
            if hasattr(layer, "is_head_layer") and layer.is_head_layer:
                logits = outputs

            if not isinstance(outputs, tuple):
                if is_pipeline_last_stage() and len(self.submodel) == 1:
                    raise RuntimeError("The model output should be a tuple which contain loss and logits, "
                                       "but now only get one tensor output.")
            else:
                if len(self.submodel) == 1 and is_pipeline_last_stage():
                    # if 'layer' of self.submodel is a complete model for current stage,
                    # get loss and logits from 'outputs'.
                    outputs, logits = outputs[0], outputs[1]
                else:
                    # if self.submodel contain multiple layers,
                    # only the first output can be pass to next layer now.
                    outputs = outputs[0]
        if is_pipeline_last_stage():
            return outputs, P.stop_gradient(logits)
        return outputs

    def construct(self, *inputs, recv_data=None):
        """
        When using pipeline parallel, the model forward will be replace by this function,
        which execute different operation or layer in different stage.
        Pipeline process function default to get the gradient of 'recv_data'.
        Therefore, please ensure 'recv_data' is the last input argument of construct function for bprop correctly.
        """
        # if pp_size=1, return the original model directly
        if get_pp_world_size() == 1 or self.new_model:
            return self.model(*inputs)

        # pylint: disable=E1102
        # pylint: disable=R1705
        if self.model_customize_staged:
            return self.model_customize_forward(*inputs, recv_data=recv_data)
        else:
            return self.model_forward(*inputs, recv_data=recv_data)
