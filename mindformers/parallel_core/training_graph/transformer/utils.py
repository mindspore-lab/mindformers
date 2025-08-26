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
"""utils"""
__all__ = ["get_attn_mask_func"]

import re
import numpy as np
import mindspore as ms
from mindspore.ops import operations as P
from mindspore import nn, Tensor
from mindformers.tools.logger import logger
from mindformers.modules.transformer import TransformerSwapConfig, TransformerRecomputeConfig
from mindformers.core.context import is_legacy_model

# pylint: disable=W0212
MAX_INT32 = 2147483647


class AttnMaskFill(nn.Cell):
    """Applies a mask to attention scores by filling masked positions with a specified value."""

    def __init__(self, config):
        super(AttnMaskFill, self).__init__()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.tile = P.Tile()
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.mul2 = P.Mul()
        self.add = P.Add()
        self.cast = P.Cast()
        self.shard(config)

    def construct(self, attention_scores: Tensor, attention_mask, fill_value=-10000.0):
        """ Construct function of AttnMaskFill. """
        ori_dtype = attention_scores.dtype
        if attention_mask.ndim == 2:
            bs, _, seq_length0, seq_length1 = self.shape(attention_scores)
            attention_mask = self.reshape(attention_mask, (1, 1, seq_length0, seq_length1))
            attention_mask = self.tile(attention_mask, (bs, 1, 1, 1))

        # socres * lower_triangle + attention_mask * -10000.0
        ones = Tensor([1.0], dtype=attention_mask.dtype)
        lower_triangle = self.sub(ones, attention_mask)
        attention_scores = self.mul(attention_scores, lower_triangle)
        attention_scores = self.add(attention_scores, self.mul2(attention_mask, fill_value))
        attention_scores = self.cast(attention_scores, ori_dtype)
        return attention_scores

    def shard(self, parallel_config):
        """sharding parameters"""
        dp = 1 if parallel_config is None else parallel_config.data_parallel_size
        tp = 1 if parallel_config is None else parallel_config.tensor_model_parallel_size

        self.tile.shard(((dp, 1, 1, 1),))
        self.sub.shard(((1,), (dp, 1, 1, 1)))
        self.mul.shard(((dp, tp, 1, 1), (dp, 1, 1, 1)))
        self.mul2.shard(((dp, 1, 1, 1), ()))
        self.add.shard(((dp, tp, 1, 1), (dp, 1, 1, 1)))


ATTNMASK_FUNC_MAP = {
    "attn_mask_fill": AttnMaskFill,
}


def get_attn_mask_func(mask_func_type):
    r"""
    Get attention mask function.

    Args:
        mask_func_type (str): The attention mask function type.

    Returns:
        Function, the attention mask function.
    """
    if mask_func_type not in ATTNMASK_FUNC_MAP:
        raise KeyError("Invalid attention mask function. Supported attention "
                       "mask function are ['attn_mask_fill'] "
                       ", but got {}.".format(mask_func_type))
    return ATTNMASK_FUNC_MAP[mask_func_type]


class LayerSetting:
    r"""
    Class for setting offset, pipeline stage, swap and select recompute for each transformer layer.
    Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.

        Args:
            num_layers(int) - The total layers in the model.
            offset(Union[int, List[int], List[List[int]]]) - The layer offset for each (mini) stage.
            parallel_config(dict) - Parallel Config.
            pp_interleave_num(int) - The number of pp-interleave. When it is larger than 1,
                each stage will be divided into mini stages by pp_interleave_num.
    """

    def __init__(self, num_layers, offset, parallel_config, pp_interleave_num=1, start_stage=0, stage_num=0):
        self.use_legacy = is_legacy_model()
        if self.use_legacy:
            default_patterns = [r'feed_forward\.mul', r'feed_forward\.w1\.activation\.silu']
            self.swap = parallel_config.swap
            self.recompute = parallel_config.recompute
        else:
            default_patterns = [r'mlp\.shared_experts\.mul', r'mlp\.shared_experts\.activation_func\.silu',
                                r'mlp\.mul', r'mlp\.activation_func\.silu']
            self.swap = TransformerSwapConfig(swap=parallel_config.cpu_offloading,
                                              layer_swap=parallel_config.cpu_offloading_num_layers,
                                              op_swap=parallel_config.op_swap)
            self.recompute = TransformerRecomputeConfig(
                recompute=parallel_config.recompute,
                select_recompute=parallel_config.select_recompute,
                parallel_optimizer_comm_recompute=parallel_config.parallel_optimizer_comm_recompute,
                select_comm_recompute=parallel_config.select_comm_recompute,
                mp_comm_recompute=parallel_config.mp_comm_recompute,
                recompute_slice_activation=parallel_config.recompute_slice_activation,
                select_recompute_exclude=parallel_config.select_recompute_exclude,
                select_comm_recompute_exclude=parallel_config.select_comm_recompute_exclude
            )
        default_comm_patterns = [r'.*\.norm']
        default_off_patterns = []
        default_comm_off_patterns = []
        try:
            use_pp_interleave = ms.get_auto_parallel_context("pipeline_interleave")
        except ValueError:
            logger.warning(f"Current MindSpore version do not pipeline interleave. `pp_interleave_num` is set to 1.")
            use_pp_interleave = False
        self.num_layers = num_layers
        if stage_num != 0:
            self.pp = stage_num
            self.start_stage = start_stage
        else:
            self.start_stage = 0
            if self.use_legacy:
                self.pp = parallel_config.pipeline_stage
            else:
                self.pp = parallel_config.pipeline_model_parallel_size
        self.backward_prefetch = 'backward_prefetch'
        self.layers = 'layers'
        self.gradient_aggregation_group = parallel_config.gradient_aggregation_group
        self.pp_interleave_num = pp_interleave_num if use_pp_interleave else 1
        self.offset = np.array(offset, np.int32)
        self._check_inputs()
        self.offset = np.broadcast_to(self.offset, (self.pp_interleave_num, self.pp))

        self.is_zbv = ms.get_auto_parallel_context("pipeline_scheduler") == "zero_bubble_v"
        avg_layer = self.num_layers // (self.pp * self.pp_interleave_num)
        self.layer_list = np.ones((self.pp_interleave_num, self.pp), np.int32) * avg_layer + self.offset
        if self.is_zbv:
            self.layer_list[1] = self.layer_list[1][::-1]
        interleave_sum = np.insert(np.cumsum(np.sum(self.layer_list, axis=1))[:-1], 0, 0)
        self.layer_accu = np.cumsum(self.layer_list, axis=1) + interleave_sum.reshape(-1, 1)
        self.pp_ids = [np.searchsorted(self.layer_accu.reshape(-1), i + 1) % self.pp for i in range(self.num_layers)]
        self.interleave_ids = [np.where(i < self.layer_accu)[0][0] for i in range(self.num_layers)]
        logger.info(f"num_layers per stage: {self.layer_list.tolist()}")
        logger.info(f"Accumulated num_layers per stage: {self.layer_accu.tolist()}")
        logged_pp_ids = [(pp_id + self.start_stage) for pp_id in self.pp_ids]
        logger.info(f"Pipeline id list with start_stage: {logged_pp_ids}")
        logger.info(f"Interleave id list: {self.interleave_ids}")
        pre_pad = np.array([[0] + [self.layer_accu[i, -1] for i in range(len(self.layer_accu) - 1)]])
        self.layer_accu_mod = np.concatenate((pre_pad.T, self.layer_accu), axis=-1)

        if not isinstance(self.recompute, bool):
            if self.swap.swap:
                self.layer_recompute = self._format_recompute_list_select_layer_index(self.recompute.recompute)
            else:
                self.layer_recompute = self._format_recompute_list(
                    self.recompute.recompute)
            self.select_recompute = self._format_recompute_dict(
                self.recompute.select_recompute, default_patterns, self.swap.swap)
            self.select_comm_recompute = self._format_recompute_dict(
                self.recompute.select_comm_recompute, default_comm_patterns, self.swap.swap)
            self.select_recompute_exclude = self._format_recompute_dict(
                self.recompute.select_recompute_exclude, default_off_patterns)
            self.select_comm_recompute_exclude = self._format_recompute_dict(
                self.recompute.select_comm_recompute_exclude, default_comm_off_patterns)
            logger.info(f"Formative layer_recompute: {self.layer_recompute}")

            logger.info("The configuration of select_recompute_exclude and select_comm_recompute_exclude "
                        "have the highest priority.")
            # check repeat config and remove
            self._check_repeat_recompute(self.select_recompute, self.select_recompute_exclude, is_comm=False)
            self._check_repeat_recompute(self.select_comm_recompute, self.select_comm_recompute_exclude, is_comm=True)

            logger.info(f"Formative select_recompute: {self.select_recompute}")
            logger.info(f"Formative select_comm_recompute: {self.select_comm_recompute}")
            logger.info(f"Formative select_recompute_exclude: {self.select_recompute_exclude}")
            logger.info(f"Formative select_comm_recompute_exclude: {self.select_comm_recompute_exclude}")

        if self.swap.swap:
            self.layer_swap = []
            self.op_swap = dict()
            self.layer_swap = self._initialize_swap_list(self.swap.layer_swap)
            for key in self.swap.op_swap:
                self.op_swap[key] = self._initialize_swap_list(self.swap.op_swap[key])
            logger.info(f"Formative layer swap: {self.layer_swap}")
            logger.info(f"Formative op swap: {self.op_swap}")
            self._check_swap_recompute_conflict()

    @staticmethod
    def _check_repeat_pattern(key, select_recompute):
        """check and return the repeat pattern"""
        repeat_key = None
        key = key.split(r'\.')
        for pattern in select_recompute:
            split_pattern = pattern.split(r'\.')
            if len(key) != len(split_pattern):
                continue
            for p1, p2 in zip(key, split_pattern):
                if not re.fullmatch(p1, p2) and not re.fullmatch(p2, p1):
                    break
            else:
                repeat_key = pattern
                break
        return repeat_key

    def _check_repeat_recompute(self, select_recompute, select_recompute_exclude, is_comm=False):
        "Check if the select_recompute conflicts select_recompute_exclude."
        comm = '_comm' if is_comm else ''
        for key in select_recompute_exclude:
            repeat_key = self._check_repeat_pattern(key, select_recompute)
            if repeat_key:
                select_recompute.pop(repeat_key)
                logger.info(f"The pattern {repeat_key} in select{comm}_recompute conflicts with "
                            f"select{comm}_recompute_exclude and will be removed.")

    def _check_swap_recompute_conflict(self):
        "Check if the layer or operator is enable swap and recompute at the same time."
        if isinstance(self.recompute.recompute, bool) and self.recompute.recompute:
            logger.warning(f"All layers and ops that are enabled swap do not work!\
                           Because recompute = {self.recompute.recompute}.")
        if isinstance(self.recompute.recompute, list) and self.recompute.recompute:
            for layer_idx in self.recompute.recompute:
                self._check_layer_swap_recompute_conflict(layer_idx, self.layer_swap)
            for op_name, _ in self.op_swap.items():
                self._check_op_swap_recompute_conflict(op_name, self.recompute.recompute)

        if isinstance(self.recompute.select_recompute, dict):
            for op_name, recompute_layers in self.recompute.select_recompute.items(): # op_name->str, recompute_info->[]
                if isinstance(recompute_layers, bool) and recompute_layers:
                    logger.warning(f"{op_name} operator in all layers that is enabled swap do not work!\
                                   Because it is enabled recompute.")
                    continue
                for layer_idx in recompute_layers:
                    self._check_layer_swap_recompute_conflict(layer_idx, self.layer_swap, op_name)
                if op_name in self.op_swap.keys():
                    self._check_op_swap_recompute_conflict(op_name, recompute_layers)

    def _check_layer_swap_recompute_conflict(self, layer_idx, layer_list, op_name="All"):
        "Check if the layer is enable swap and recompute at the same time."
        for swap_layer in layer_list:
            if isinstance(swap_layer.get(self.layers), bool) and swap_layer.get(self.layers):
                logger.warning(f"{op_name} operator in layer {layer_idx} that\
                               is enabled swap do not work! Because it is enabled recompute.")
            else:
                if layer_idx in swap_layer.get(self.layers):
                    logger.warning(f"{op_name} operator in layer {layer_idx} that\
                                   is enabled swap do not work! Because it is enabled recompute.")

    def _check_op_swap_recompute_conflict(self, op_name, layer_list):
        "Check if the operator is enable swap and recompute at the same time."
        if isinstance(layer_list, bool) and layer_list:
            logger.warning(f"{op_name} operator that is enabled swap do not work! Because it is enabled recompute.")
        else:
            op_swap_layer_list = self.op_swap[op_name]
            op_swap_layers = set()
            for swap_info in op_swap_layer_list:
                if isinstance(swap_info.get(self.layers), bool) and swap_info.get(self.layers):
                    op_swap_layers.clear()
                    op_swap_layers = True
                    break
                op_swap_layers = op_swap_layers.union(set(swap_info.get(self.layers)))
            for layer_idx in layer_list:
                if isinstance(op_swap_layers, bool) or layer_idx in op_swap_layers:
                    logger.warning(f"{op_name} operator in layer {layer_idx} that\
                                   is enabled swap do not work! Because it is enabled recompute.")

    def _initialize_swap_list(self, swap_list):
        """Initialize the swap list by creating swap configurations for each item."""
        if self.swap.swap and not swap_list:
            return []
        result = []
        for item in swap_list:
            layers = self._format_recompute_list_select_layer_index(item.get(self.layers))
            swap_config = self._create_swap_dict(
                item.get(self.backward_prefetch),
                layers
            )
            result.append(swap_config)
        return result

    @staticmethod
    def _create_swap_dict(backward_prefetch, layers):
        """Create a dictionary for swap configuration with backward_prefetch and layers."""
        return {'backward_prefetch': backward_prefetch, 'layers': layers}

    def set_swap(self, layer, layer_id):
        """Set swap for a specific layer based on its layer_id."""
        if self.swap.swap:
            if self.layer_swap and isinstance(self.layer_swap[0].get(self.layers), bool):
                if self.layer_swap[0].get(self.layers):
                    layer.offload(backward_prefetch=self.layer_swap[0].get(self.backward_prefetch))
                    logger.info(f"Set layer swap at layer {layer_id} \
                                and value is: {self.layer_swap[0].get(self.backward_prefetch)}")
            else:
                self._set_layer_swap(layer, layer_id)
            self._set_op_swap(layer, layer_id)

    def _set_op_swap(self, layer, layer_id):
        """Set swap for operations in the layer based on patterns and layer_id."""
        log_ops = []
        for pattern in self.op_swap:
            for layer_swap in self.op_swap[pattern]:
                layers_id = layer_swap.get(self.layers)
                is_valid_bool = isinstance(layers_id, bool) and layers_id
                is_valid_list = isinstance(layers_id, list) and layer_id in layers_id
                if is_valid_bool or is_valid_list:
                    log = LayerSetting.set_pattern_swap(layer, pattern.split(r'\.'),
                                                        layer_swap.get(self.backward_prefetch))
                    if log:
                        log_ops.append(log)
                    break
        if log_ops:
            logger.info(f"Set op_swap at layer {layer_id}: {', '.join(log_ops)}")

    def _set_layer_swap(self, layer, layer_id):
        """Set swap for the entire layer based on the layer_id."""
        for layer_swap in self.layer_swap:
            if layer_id in layer_swap.get(self.layers):
                layer.offload(backward_prefetch=layer_swap.get(self.backward_prefetch))
                logger.info(f"Set layer swap at layer {layer_id} \
                            and value is: {layer_swap.get(self.backward_prefetch)}")
                break

    @staticmethod
    def set_pattern_swap(layer, p_list, value, info=''):
        """Set swap for operators in the layer based on a pattern list and value."""
        log_list = []
        log = ''
        if p_list:
            p = p_list.pop(0)
        else:
            return info
        if p_list:
            # pylint: disable=W0212
            for name, cell in layer._cells.items():
                if re.fullmatch(p, name):
                    log = LayerSetting.set_pattern_swap(cell, p_list, value, info + f'.{name}')
                    if log:
                        log_list.append(log[1:])
        else:
            for attr in dir(layer):
                if re.fullmatch(p, attr):
                    operator = getattr(layer, attr)
                    if hasattr(operator, '_offload'):
                        operator._offload(backward_prefetch=value)
                        log = f"{info}.{attr}, value={value}"
            # pylint: disable=W0212
            for name, cell in layer._cells.items():
                if re.fullmatch(p, name):
                    cell.offload(backward_prefetch=value)
                    log = f"{info}.{name}, value={value}"
        p_list.insert(0, p)
        if log_list:
            return " " + ", ".join(log_list)
        return log

    def set(self, layer, layer_id):
        """Set pipeline stage and recompute for each layer with a layer_id."""
        pp_id = int(self.pp_ids[layer_id]) + self.start_stage
        if self.is_zbv:
            if self.interleave_ids[layer_id] == 1:
                pp_id = self.pp - 1 - pp_id
                layer.pipeline_segment = 1
            else:
                layer.pipeline_segment = 0
        layer.pipeline_stage = pp_id
        dis = max(int((self.num_layers + 1) / self.gradient_aggregation_group), 1)
        if self.pp > 1:
            layer.set_comm_fusion(2)
        else:
            layer.set_comm_fusion(int((layer_id + self.offset[0, 0]) / dis) + 1)

        if isinstance(self.recompute, bool):
            if self.recompute:
                layer.recompute()
            return

        is_full_recompute = False
        if self.recompute.recompute:
            if isinstance(self.recompute.recompute, bool):
                layer.recompute(recompute_slice_activation=self.recompute.recompute_slice_activation)
                is_full_recompute = True
                logger.info(f"Set full recompute at layer {layer_id}")
            elif self._check_layer_rule(layer_id):
                layer.recompute()
                is_full_recompute = True
                logger.info(f"Set full recompute at layer {layer_id}")
        if not is_full_recompute:
            self._set_select_recompute(layer, layer_id, False, set_on=True)
            self._set_select_recompute(layer, layer_id, True, set_on=True)

        # select recompute exclude
        self._set_select_recompute(layer, layer_id, False, set_on=False)
        self._set_select_recompute(layer, layer_id, True, set_on=False)


    def set_recompute_select_layer_index(self, layer, layer_id):
        """Set swap for specific layer based on its layer_id when."""
        if isinstance(self.recompute, bool):
            if self.recompute:
                layer.recompute()
            return
        if self.recompute.recompute:
            if isinstance(self.recompute.recompute, bool):
                if self.recompute.recompute:
                    layer.recompute(recompute_slice_activation=self.recompute.recompute_slice_activation)
            else:
                if layer_id in self.layer_recompute:
                    layer.recompute()
                    logger.info(f"Set full recompute at layer {layer_id}")
                else:
                    self._set_select_recompute_layer_index(layer, layer_id, False)
                    self._set_select_recompute_layer_index(layer, layer_id, True)
        else:
            self._set_select_recompute_layer_index(layer, layer_id, False)
            self._set_select_recompute_layer_index(layer, layer_id, True)

    def _alloc_recompute_layer(self, select_recompute):
        """Average allocate recompute layer among different interleave."""
        res = np.zeros_like(self.layer_list)
        layer_by_stage = np.sum(self.layer_list, axis=0)
        for j, num in enumerate(select_recompute):
            num = min(num, layer_by_stage[j])
            while num:
                for i in range(self.pp_interleave_num):
                    if not num:
                        break
                    if res[i, j] < self.layer_list[i, j]:
                        res[i, j] += 1
                        num -= 1
        return res

    def _format_recompute_list(self, select_recompute):
        """Format recompute inputs into a list."""
        if isinstance(select_recompute, bool):
            if select_recompute:
                return self.layer_list.tolist()
            return np.zeros_like(self.layer_list).tolist()
        if isinstance(select_recompute, (list, tuple)):
            select_recompute = list(select_recompute)
            if self.is_zbv:
                select_recompute[1] = select_recompute[1][::-1]
            bool_to_value = {False: 0, True: MAX_INT32}
            if all(isinstance(item, (int, bool)) for item in select_recompute):
                for i, item in enumerate(select_recompute):
                    if isinstance(item, bool):
                        select_recompute[i] = bool_to_value[item]
                return self._alloc_recompute_layer(select_recompute).tolist()
            if all(isinstance(item, str) for item in select_recompute):
                return self.layer_list.tolist()
            for i, sub in enumerate(select_recompute):
                if isinstance(sub, bool):
                    select_recompute[i] = self.layer_list[i] if sub else np.zeros_like(self.layer_list[i])
                elif isinstance(sub, (list, tuple)):
                    for j, item in enumerate(sub):
                        if isinstance(item, bool):
                            select_recompute[i][j] = self.layer_list[i][j] if item else 0
                        elif isinstance(item, int):
                            select_recompute[i][j] = min(select_recompute[i][j], self.layer_list[i, j])
                        else:
                            raise ValueError(f"Illegal input list for select_recompute: {select_recompute}")
                else:
                    raise ValueError(f"Illegal input list for select_recompute: {select_recompute}")
            return select_recompute
        raise ValueError(f"Illegal input list for select_recompute: {select_recompute}")

    @staticmethod
    def _format_recompute_list_select_layer_index(select_recompute):
        """Format recompute inputs into a list when using swap."""
        if isinstance(select_recompute, bool):
            if select_recompute:
                return select_recompute
            return []
        if isinstance(select_recompute, list):
            if all(isinstance(item, int) for item in select_recompute):
                return select_recompute
        raise ValueError(f"Illegal input list for select_recompute: {select_recompute}")

    def _format_recompute_dict(self, select_recompute, default_patterns, select_layer_index=False):
        """Format select_recompute inputs into a dict"""
        dic = {}
        if isinstance(select_recompute, (list, tuple)) and all(isinstance(item, str) for item in select_recompute):
            parttern = select_recompute
        elif isinstance(select_recompute, dict):
            parttern = select_recompute.keys()
        else:
            parttern = default_patterns
        for p in parttern:
            value = select_recompute[p] if isinstance(select_recompute, dict) else select_recompute
            if select_layer_index:
                dic[p] = self._format_recompute_list_select_layer_index(value)
            else:
                dic[p] = self._format_recompute_list(value)
        return dic

    def _check_layer_rule(self, layer_id):
        """Check whether a layer should be set full layer recompute."""
        pp_id = int(self.pp_ids[layer_id])
        v_id = int(self.interleave_ids[layer_id])
        if 0 <= layer_id - self.layer_accu_mod[v_id, pp_id] < self.layer_recompute[v_id][pp_id]:
            return True
        return False

    def _set_select_recompute(self, layer, layer_id, add_prim_attr=False, set_on=True):
        """Set select recompute on/off for a layer."""
        if set_on:
            select_recompute = self.select_comm_recompute if add_prim_attr else self.select_recompute
            action = "on"
        else:
            select_recompute = self.select_comm_recompute_exclude if add_prim_attr else self.select_recompute_exclude
            action = "exclude"
        pp_id = int(self.pp_ids[layer_id])
        v_id = int(self.interleave_ids[layer_id])
        log_ops = []
        for pattern, layers_dict in select_recompute.items():
            if 0 <= layer_id - self.layer_accu_mod[v_id, pp_id] < layers_dict[v_id][pp_id]:
                log = LayerSetting.set_pattern_recompute(layer, pattern.split(r'\.'), add_prim_attr, set_on)
                if log:
                    log_ops.append(log[1:])
        log_ops_str = ', '.join(log_ops)
        if log_ops_str:
            comm = 'comm ' if add_prim_attr else ''
            logger.info(f"Set select {comm}recompute {action} at layer {layer_id}: {log_ops_str}")

    def _set_select_recompute_layer_index(self, layer, layer_id, add_prim_attr=False):
        """Set select recompute for a layer when using swap."""
        select_recompute = self.select_comm_recompute if add_prim_attr else self.select_recompute
        log_ops = []
        for pattern, layers_recompute in select_recompute.items():
            if (isinstance(layers_recompute, bool) and layers_recompute) or layer_id in layers_recompute:
                log = LayerSetting.set_pattern_recompute(layer, pattern.split(r'\.'), add_prim_attr)
                if log:
                    log_ops.append(log[1:])
        log_ops_str = ', '.join(log_ops)
        if log_ops_str:
            comm = 'comm ' if add_prim_attr else ''
            logger.info(f"Set select {comm}recompute at layer {layer_id}: {log_ops_str}")

    def _check_inputs(self):
        """Check the inputs of offset."""
        if self.offset.ndim >= 1 and self.offset.shape[-1] != self.pp:
            raise ValueError(f"offset.shape[-1] should equal to `pp` ({self.pp}), "
                             f"but got ({self.offset.shape[-1]}). `offset`: {self.offset}")
        if self.offset.ndim >= 2 and self.offset.shape[-2] != self.pp_interleave_num:
            raise ValueError(f"offset.shape[-2] should equal to `pp_interleave_num` ({self.pp_interleave_num}), "
                             f"but got ({self.offset.shape[-2]}). `offset`: {self.offset}")
        if self.offset.sum() != self.num_layers % (self.pp * self.pp_interleave_num):
            r = self.num_layers % (self.pp * self.pp_interleave_num)
            raise ValueError(f"The sum of `offset` ({self.offset.sum()}) should equal to remainder of `num_layers` "
                             f"({self.num_layers}) % (pp ({self.pp}) * pp_interleave_num ({self.pp_interleave_num})) "
                             f"= {r}")

    @staticmethod
    def set_pattern_recompute(layer, p_list, add_prim_attr=False, set_on=True, info=''):
        """Set an operator recompute status on/off for a given key-value pair in select_recompute dict."""
        log_list = []
        log = ''
        if p_list:
            p = p_list.pop(0)
        else:
            return info
        if p_list:
            # pylint: disable=W0212
            for name, cell in layer._cells.items():
                if re.fullmatch(p, name):
                    log = LayerSetting.set_pattern_recompute(cell, p_list, add_prim_attr, set_on, info + f'.{name}')
                    if log:
                        log_list.append(log[1:])
        else:
            for attr in dir(layer):
                if re.fullmatch(p, attr):
                    operator = getattr(layer, attr)
                    if add_prim_attr:
                        operator.add_prim_attr("recompute_comm_op", set_on)
                        log = f"{info}.{attr}"
                    elif hasattr(operator, 'recompute'):
                        operator.recompute(set_on)
                        log = f"{info}.{attr}"
            # pylint: disable=W0212
            for name, cell in layer._cells.items():
                if re.fullmatch(p, name):
                    if not set_on:
                        logger.info(f"For select recompute/comm_recompute exclude, {info.replace('.', '', 1)}.{name} "
                                    "is expected to be operation but got cell, "
                                    "this configuration will not be effective.")
                        continue
                    if add_prim_attr:
                        logger.info(f"For communication recompute, {info.replace('.', '', 1)}.{name} "
                                    "is expected to be operation but got cell, "
                                    "this configuration will not be effective.")
                        continue
                    cell.recompute()
                    log = f"{info}.{name}"
        p_list.insert(0, p)
        if log_list:
            return " " + ", ".join(log_list)
        return log

    def __call__(self, layer, layer_id):
        if self.swap.swap:
            self.set_recompute_select_layer_index(layer, layer_id)
            self.set_swap(layer, layer_id)
        else:
            self.set(layer, layer_id)
