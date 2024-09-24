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
""" Distributed optimizer wrapper. """
from collections import OrderedDict
import json

import mindspore as ms
import mindspore.ops as ops
from mindspore.common import dtype as mstype
from mindspore.common.initializer import Zero
from mindspore.communication.management import get_group_size, get_rank
from mindspore.communication.comm_func import all_gather_into_tensor

from mindformers.experimental.parallel_core.pynative.distributed import ParamAndGradBuffer

from .optimizer import MixedPrecisionOptimizer

from .utils import _update_adamw_var


def shard_bucket(bucket, group):
    """ shard bucket to dp_size part. """
    dp_world_size = get_group_size(group=group)
    dp_rank = get_rank(group=group)
    bucket_size = bucket.grad_data.numel()

    # each rank processes a slice data of whole bucket
    size_per_dp_rank = bucket_size // dp_world_size

    # bucket range info of slice data processed by this rank
    shard_start = dp_rank * size_per_dp_rank
    shard_end = min(bucket_size, shard_start + size_per_dp_rank)

    return shard_start, shard_end


class DistributedOptimizer(MixedPrecisionOptimizer):
    """
    Distributed optimizer implementation. This class wrap a non-parallel optimizer

    This class build the mapping between parameters' range take charge by this dp rank in
    the optimizer parallelism and their shard ranges in the buffer, sharded bucket, integrated bucket
    and integrated parameters. This mapping is needed for converting between model param indexes
    and main parameter shard indexes. This class also update the non-parallel optimizer attributes
    according to parameter shard information.

    Args:
        optimizer (mindspore.mint.optim): Non-parallel optimizer.
        config (OptimizerConfig): The OptimizerConfig object containing the optimizer related configurations.
        per_model_buffers (List): List of biffers of all model chunks.
        data_parallel_group (str): Data parallel group name.

    Examples:
        >>> from mindformers.experimental.distri_cores.distributed import DistributedDataParallel, \
        >>>     DistributedDataParallelConfig
        >>> from mindformers.experimental.distri_cores.optimizer import DistributedOptimizer
        >>> network = Net(config=model_config)
        >>> ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, use_distributed_optimizer=True)
        >>> network = DistributedDataParallel(config=model_config,
        >>>                                   ddp_config=ddp_config,
        >>>                                   module=network_with_loss)
        >>> optimizer = DistributedOptimizer(optimizer=optimizer,
        >>>                                  config=optimizer_config,
        >>>                                  per_model_buffers=network.buffers,
        >>>                                  data_parallel_group=get_data_parallel_group(with_context_parallel=True))
    """
    @classmethod
    def _build_param_ranges_map(
            cls,
            param_index_map,
            bucket_world_range,
            bucket_offset
        ):
        """ Build parameter range map. """
        # param_range_map
        param_range_map = {}
        bucket_world_start, bucket_world_end = bucket_world_range
        shard_size = bucket_world_end - bucket_world_start

        for param, param_world_indexes in param_index_map.items():
            param_world_start, param_world_end, _ = param_world_indexes
            param_local_start = max(0, param_world_start - bucket_world_start)
            param_local_end = min(param_world_end - bucket_world_start, shard_size)

            # param_range_map only record shard info for parameters
            # in the buffer shard processed by this dp rank
            if param_local_start < param_local_end:
                # range in bucket shard of this param
                range_in_shard = (param_local_start, param_local_end)
                local_size_in_shard = param_local_end - param_local_start
                # range in buffer of this param
                range_in_buffer = (
                    bucket_world_start + param_local_start,
                    bucket_world_start + param_local_start + local_size_in_shard
                )
                # range in bucket of this param
                range_in_whole_bucket = (
                    range_in_buffer[0] - bucket_offset,
                    range_in_buffer[1] - bucket_offset
                )
                # range in integrated param of this param slice
                sub_param_start = max(0, bucket_world_start - param_world_start)
                sub_param_end = sub_param_start + local_size_in_shard
                sub_param_range = (sub_param_start, sub_param_end)
                # build param range map
                param_range_map[param] = {
                    'range_in_buffer': range_in_buffer,
                    'range_in_bucket': range_in_whole_bucket,
                    'range_in_shard': range_in_shard,
                    'range_in_param': sub_param_range
                }

        return param_range_map


    @classmethod
    def _build_bucket_ranges_map(cls, param_and_grad_buffer, bucket_index):
        """ Build parameter range map for bucket. """
        bucket = param_and_grad_buffer.buckets[bucket_index]
        data_parallel_group = param_and_grad_buffer.data_parallel_group
        shard_start, shard_end = shard_bucket(bucket, group=data_parallel_group)

        # buffer range info of slice data processed by this rank
        bucket_world_range = (shard_start + bucket.offset, shard_end + bucket.offset)

        param_range_map = cls._build_param_ranges_map(param_and_grad_buffer.param_index_map,
                                                      bucket_world_range,
                                                      bucket.offset)
        return param_range_map

    @classmethod
    def _build_buffer_ranges_map(cls, param_and_grad_buffer):
        """ Build parameter range map for buffer. """
        return [
            cls._build_bucket_ranges_map(param_and_grad_buffer, bucket_idx)
            for bucket_idx in range(len(param_and_grad_buffer.buckets))
        ]

    @classmethod
    def _get_optimizer_group_ranges(cls, param_groups, param_ranges_map):
        """ Build optimizer group info for distributed optimizer. """
        world_param_group_map = {}
        # build map of parameters to their optimizer group in the original optimizer
        for group_idx, group in enumerate(param_groups):
            for param in group['params']:
                if not param.requires_grad:
                    raise ValueError("param.requires_grad should be True but got False!")
                world_param_group_map[param] = group_idx

        # In distributed optimizer, each dp rank only update a part of parameters,
        # thus only those parameters' group information is required. local_param_group_map mapping
        # parameters to their optimizer group index and its index in the groups' 'param' list.
        local_param_group_map = {}
        sharded_param_groups = [{'params': []} for _ in range(len(param_groups))]
        for all_bucket_range_list in param_ranges_map:
            for param_range_map in all_bucket_range_list:
                for param in param_range_map:
                    group_idx = world_param_group_map[param]
                    sharded_group = sharded_param_groups[group_idx]
                    sharded_group['params'].append(param)
                    local_param_group_map[param] = (group_idx, len(sharded_group['params']) - 1)

        return local_param_group_map, sharded_param_groups

    @classmethod
    def _build_sharded_params_and_grads(
            cls,
            param_ranges_map,
            param_to_bucket_map,
            sharded_param_groups,
            buffers,
        ):
        """ Build shards of param and grad buffer. """
        param_fp16_groups = []
        param_fp32_groups = []
        # sharded_param_fp16_groups and sharded_param_fp32_groups
        # are view tensors on corresponding param_and_grad_buffer.
        # sharded_param_fp32_from_fp16_groups is a float32 copy of parameters,
        # which will apply for a new contiguous memory.
        sharded_param_fp16_groups = []
        sharded_param_fp32_from_fp16_groups = []
        sharded_param_fp32_groups = []

        for sharded_group in sharded_param_groups:
            param_fp16_this_group = []
            param_fp32_this_group = []
            sharded_param_fp16_this_group = []
            sharded_param_fp32_from_fp16_this_group = []
            sharded_param_fp32_this_group = []

            # the param is the integrated parameter object
            for param in sharded_group['params']:
                buffer_idx, bucket_idx = param_to_bucket_map[param]
                param_range = param_ranges_map[buffer_idx][bucket_idx][param]
                param_start_in_buffer, param_end_in_buffer = param_range['range_in_buffer']
                # for float16 and bfloat16 parameters, clone an float32 copy
                if param.dtype in [mstype.float16, mstype.bfloat16]:
                    sharded_param_fp16 = buffers[buffer_idx].param_data[param_start_in_buffer:param_end_in_buffer]
                    sharded_param_fp32_from_fp16 = ops.cast(sharded_param_fp16, mstype.float32)
                    sharded_param_fp16.name = param.name
                    sharded_param_fp32_from_fp16.name = param.name
                    param.main_param = sharded_param_fp32_from_fp16
                    sharded_grad_fp32_from_fp16 = ops.cast(buffers[buffer_idx].grad_data[param_start_in_buffer: \
                                                           param_end_in_buffer],
                                                           mstype.float32)
                    param.grad = sharded_grad_fp32_from_fp16
                    sharded_param_fp32_from_fp16.grad = sharded_grad_fp32_from_fp16

                    param_fp16_this_group.append(param)
                    sharded_param_fp16_this_group.append(sharded_param_fp16)
                    sharded_param_fp32_from_fp16_this_group.append(sharded_param_fp32_from_fp16)

                elif param.dtype == mstype.float32:
                    sharded_param_fp32 = buffers[buffer_idx].param_data[param_start_in_buffer:param_end_in_buffer]
                    sharded_param_fp32.name = param.name
                    param.main_param = sharded_param_fp32
                    sharded_grad_fp32 = buffers[buffer_idx].grad_data[param_start_in_buffer: param_end_in_buffer]
                    param.grad = sharded_grad_fp32
                    sharded_param_fp32.grad = sharded_grad_fp32
                    param_fp32_this_group.append(param)
                    sharded_param_fp32_this_group.append(sharded_param_fp32)

                else:
                    raise TypeError("Invalid parameter dtype. Supported parameter dtypes are"
                                    "`mindspore.float16`, `mindspore.bfloat16` and `mindspore.float32`,"
                                    " but got {}.".format(param.dtype))
            param_fp16_groups.append(param_fp16_this_group)
            param_fp32_groups.append(param_fp32_this_group)
            sharded_param_fp16_groups.append(sharded_param_fp16_this_group)
            sharded_param_fp32_from_fp16_groups.append(sharded_param_fp32_from_fp16_this_group)
            sharded_param_fp32_groups.append(sharded_param_fp32_this_group)

        return (
            param_fp16_groups,
            param_fp32_groups,
            sharded_param_fp16_groups,
            sharded_param_fp32_from_fp16_groups,
            sharded_param_fp32_groups,
        )

    def __init__(
            self,
            optimizer,
            config,
            grad_scaler,
            init_state_fn,
            per_model_buffers,
            data_parallel_group,
        ):
        super().__init__(
            optimizer,
            config,
            grad_scaler,
            init_state_fn
        )

        self.buffers = []
        if isinstance(per_model_buffers[0], ParamAndGradBuffer):
            per_model_buffers = {0: per_model_buffers}
        self.per_model_buffers = per_model_buffers
        self.data_parallel_group = data_parallel_group
        self.update_success = False

        self.buffer_idx_to_model_idex_map = {}
        self.param_to_bucket_map = {}
        buffer_idx = 0

        for model_idx, buffers in self.per_model_buffers.items():
            for buffer in buffers:
                self.buffer_idx_to_model_idex_map[buffer_idx] = model_idx
                self.buffers.append(buffer)
                buffer_idx += 1

        self.param_ranges_map = []
        # build param ranges
        for buffer in self.buffers:
            self.param_ranges_map.append(self._build_buffer_ranges_map(buffer))

        # build param to bucket map
        for buffer_idx, all_bucket_range_list in enumerate(self.param_ranges_map):
            for bucket_index, param_range_map in enumerate(all_bucket_range_list):
                for param, _ in param_range_map.items():
                    if param not in self.param_to_bucket_map:
                        self.param_to_bucket_map[param] = (buffer_idx, bucket_index)
                    else:
                        raise ValueError("Parameter should only belongs to a single bucket.")

        # get optimizer group info
        (
            self.model_param_group_map,
            self.sharded_param_groups
        ) = self._get_optimizer_group_ranges(self.optimizer.param_groups, self.param_ranges_map)

        # group parameters and gradients
        (
            self.param_fp16_groups,
            self.param_fp32_groups,
            self.sharded_param_fp16_groups,
            self.sharded_param_fp32_from_fp16_groups,
            self.sharded_param_fp32_groups,
        ) = self._build_sharded_params_and_grads(self.param_ranges_map,
                                                 self.param_to_bucket_map,
                                                 self.sharded_param_groups,
                                                 self.buffers)

        self.param_buffer_dp_views = self._get_model_param_buffer_dp_views()
        self.overlap_param_gather = False

        # update self.optimizer attributions according to shard info
        self._update_optimizer_attr()

        # build map between parameter name and its index in optimizer.parameters after sharding
        self.param_idx_in_opt = {}
        for idx, param in enumerate(self.optimizer.parameters):
            self.param_idx_in_opt[param.name] = idx

        # build fp32 copy for fp16/bf16 params
        self.reload_model_params()

    def zero_grad(self):
        """ reset grads data. """
        self.grads = []

    def step_with_ready_grads(self):
        """ optimizer update and synchronize updated parameters among dp group. """
        self.update_success = super().step_with_ready_grads()
        if self.update_success:
            # allgather updated buckets' data among dp group
            self._sync_gather_all_model_params()

    def load_state_dict(self, state_dict):
        """ load the state dict. """
        if len(self.param_to_bucket_map) != len(self.optimizer.parameters):
            raise ValueError(
                f"Inconsistent size between "
                f"'self.param_to_bucket_map' ({len(self.param_to_bucket_map)}) and "
                f"'self.optimizer.parameters' ({len(self.optimizer.parameters)})"
            )

        # Build a mapping from sharded param to sharded state
        shard_param_to_state_map = {
            param.name: (exp_avg, exp_avg_sq)
            for param, exp_avg, exp_avg_sq in zip(
                self.optimizer.parameters,
                self.optimizer.exp_avg,
                self.optimizer.exp_avg_sq
            )
        }

        # Load optimizer state from the state dict
        for param, bucket_info in self.param_to_bucket_map.items():
            # Get param range info
            buffer_idx, bucket_idx = bucket_info
            param_range = self.param_ranges_map[buffer_idx][bucket_idx][param]
            param_start, param_end = param_range["range_in_param"]

            # Get sharded param & state
            shard_param = param.main_param
            shard_state = shard_param_to_state_map.get(shard_param.name, tuple())
            if len(shard_state) != 2:
                raise ValueError(f"Fail to get sharded states of '{param.name}'.")
            shard_exp_avg, shard_exp_avg_sq = shard_state

            # Get weight from state dict
            for ele in [shard_param, shard_exp_avg, shard_exp_avg_sq]:
                weight = state_dict.get(ele.name)
                if weight is None:
                    raise ValueError(f"Fail to get weight of '{ele.name}' from state dict.")
                ele.copy_(weight.view(-1)[param_start: param_end])

        if 'state_step' in state_dict.keys():
            self.optimizer.state_step.assign_value(state_dict['state_step'].value())

    def _update_optimizer_attr(self):
        """ update attributes of self.optimizer according to shard information. """
        # reset optimizer.parameter
        self.optimizer.ori_parameters = self.optimizer.parameters
        self.optimizer.parameters = []
        for group_idx, _ in enumerate(self.optimizer.param_groups):
            # update params in this group
            self.optimizer.param_groups[group_idx]['params'] = [
                *self.sharded_param_fp32_groups[group_idx],\
                *self.sharded_param_fp32_from_fp16_groups[group_idx]
            ]
            self.optimizer.group_start_id[group_idx + 1] = self.optimizer.group_start_id[group_idx] + \
                len(self.optimizer.param_groups[group_idx]['params'])
            self.optimizer.lrs[group_idx] = self.optimizer.param_groups[group_idx]['lr']
            self.optimizer.parameters += tuple(self.optimizer.param_groups[group_idx]['params'])

        self.parameters = self.optimizer.ori_parameters
        self.defaults = self.optimizer.defaults

        # update non-parallel optimizer attributes
        _update_adamw_var(self.optimizer)

    def _collect_main_grad_data(self):
        """ collect grads this dp rank takes into account. """
        for param in self.optimizer.parameters:
            self.grads.append(param.grad)

    def _get_model_param_buffer_dp_views(self):
        """ get shard metadata for each bucket among dp group. """
        view_items = []
        for buffer_index, buffer in enumerate(self.buffers):
            for bucket_index, bucket in enumerate(buffer.buckets):
                shard_start, shard_end = shard_bucket(bucket, group=self.data_parallel_group)
                view_items.append((buffer_index, bucket_index, shard_start, shard_end))

        return view_items

    def _copy_model_grads_to_main_grads(self):
        """
        Before distributed optimizer update, copy the the gradient elements this dp rank take charge to sharded gradient
        group. For fp16 gradients, a fp32 copy will be created and optimizer will update using fp32 gradients instead of
        original fp16 gradients.
        """
        def copy_group_grads(model_groups, main_groups):
            for model_group, main_group in zip(model_groups, main_groups):
                for model_param, main_param in zip(model_group, main_group):
                    buffer_idx, bucket_idx = self.param_to_bucket_map.get(model_param)
                    range_map = self.param_ranges_map[buffer_idx][bucket_idx][model_param]
                    param_start, param_end = range_map['range_in_param']
                    main_param.grad.copy_(ops.cast(model_param.main_grad.view(-1)[param_start: param_end],
                                                   mstype.float32))

        copy_group_grads(self.param_fp32_groups, self.sharded_param_fp32_groups)
        copy_group_grads(self.param_fp16_groups, self.sharded_param_fp32_from_fp16_groups)

    def _copy_model_params_to_main_params(self):
        """
        Before distributed optimizer update, copy the the elements this dp rank take charge to sharded param
        group. For fp16 params, a fp32 copy will be created and optimizer will update the fp32 params instead of
        original fp16 params.
        """
        def copy_group_params(model_groups, main_groups):
            for model_group, main_group in zip(model_groups, main_groups):
                for model_param, main_param in zip(model_group, main_group):
                    buffer_idx, bucket_idx = self.param_to_bucket_map.get(model_param)
                    range_map = self.param_ranges_map[buffer_idx][bucket_idx][model_param]
                    param_start, param_end = range_map['range_in_param']
                    main_param.copy_(ops.cast(model_param.view(-1)[param_start:param_end], mstype.float32))

        copy_group_params(self.param_fp32_groups, self.sharded_param_fp32_groups)
        copy_group_params(self.param_fp16_groups, self.sharded_param_fp32_from_fp16_groups)

    def _copy_main_params_to_model_params(self):
        """
        After distributed optimizer update, update result need to be copied back to section in param_data buffer.
        For parameters with fp16 data type, a fp32 copy is used for optimizer update, their update result will be cast
        to original param data type and be copied back to param_data buffer with fp16 data type.
        """
        def copy_group_params(main_groups, model_groups):
            for main_group, model_group in zip(main_groups, model_groups):
                for main_param, model_param in zip(main_group, model_group):
                    buffer_idx, bucket_idx = self.param_to_bucket_map.get(model_param)
                    range_map = self.param_ranges_map[buffer_idx][bucket_idx][model_param]
                    param_start, param_end = range_map['range_in_param']
                    model_param.view(-1)[param_start:param_end].copy_(ops.cast(main_param, model_param.dtype))

        copy_group_params(self.sharded_param_fp32_groups, self.param_fp32_groups)
        copy_group_params(self.sharded_param_fp32_from_fp16_groups, self.param_fp16_groups)

    def _sync_gather_all_model_params(self):
        """
        After distributed optimizer update, only the elements this dp rank take charge has been updated.
        This function conducts all-gather on data parallel group to get all updated parameters.
        """
        if not self.overlap_param_gather:
            for buffer_index, bucket_index, shard_start, shard_end in self.param_buffer_dp_views:
                bucket = self.buffers[buffer_index].buckets[bucket_index]
                param_data_view = bucket.param_data[shard_start:shard_end]
                param_data = all_gather_into_tensor(param_data_view, group=bucket.data_parallel_group)[0].reshape(-1)
                bucket.param_data.copy_(param_data)


    def save_opt_shard_strategy(self, file):
        """
        Save distributed optimizer shard info as json file.

        Inputs:
            file (str): Path to save optimizer shard strategy.
        """
        # only rank 0 of each data parallel group need save strategy file.
        if get_rank(self.data_parallel_group) != 0:
            return
        strategy = {}
        strategy['dp_rank_list'] = [int(s) for s in self.data_parallel_group.split('-') if s.isdigit()]
        # build parameter info
        strategy['param_info'] = {}
        for buffer_idx, buffer in enumerate(self.buffers):
            for param, value in buffer.param_index_map.items():
                start_idx, end_idx, bucket_idx = value
                this_buffer = self.buffers[buffer_idx]
                start_idx, end_idx, bucket_idx = this_buffer.param_index_map[param]
                strategy['param_info'][param.name] = {}
                strategy['param_info'][param.name]['range_map'] = (buffer_idx, bucket_idx, start_idx, end_idx)
                strategy['param_info'][param.name]['shape'] = param.shape
                strategy['param_info'][param.name]['dtype'] = str(param.dtype).lower()
        # build buffer info
        strategy['buffer_info'] = {}
        for buffer_idx, buffer in enumerate(self.buffers):
            strategy['buffer_info'][buffer_idx] = {}
            strategy['buffer_info'][buffer_idx]['buckets'] = {
                bucket_idx: (
                    bucket.grad_data_numel,
                    bucket.numel_unpadded
                ) for bucket_idx, bucket in enumerate(buffer.buckets)
            }
            strategy['buffer_info'][buffer_idx]['buffer_size'] = buffer.numel
            strategy['buffer_info'][buffer_idx]['buffer_numel_unpadded'] = buffer.numel_unpadded
            strategy['buffer_info'][buffer_idx]['bucket_num'] = len(buffer.buckets)
        # save as json file
        with open(file, 'w') as f:
            json.dump(strategy, f, indent=4)

    def parameters_dict(self):
        """ get parameter dict for save checkpoint. """
        param_dict = OrderedDict()

        for buffer_index, bucket_index, shard_start, shard_end in self.param_buffer_dp_views:
            param_range_map_this_bucket = self.param_ranges_map[buffer_index][bucket_index]
            # create dummy tensor for this bucket shard, which will be used to save checkpoint
            param_shard = ms.Tensor(shape=(shard_end-shard_start), dtype=mstype.float32, init=Zero())
            exp_avg_shard = ms.Tensor(shape=(shard_end-shard_start), dtype=mstype.float32, init=Zero())
            exp_avg_sq_shard = ms.Tensor(shape=(shard_end-shard_start), dtype=mstype.float32, init=Zero())
            # copy param data into dummy tensor
            for param, range_map in param_range_map_this_bucket.items():
                start_idx, end_idx = range_map['range_in_shard']
                param_id_in_opt = self.param_idx_in_opt[param.name]
                param_shard[start_idx:end_idx].copy_(self.optimizer.parameters[param_id_in_opt])
                exp_avg_shard[start_idx:end_idx].copy_(self.optimizer.exp_avg[param_id_in_opt])
                exp_avg_sq_shard[start_idx:end_idx].copy_(self.optimizer.exp_avg_sq[param_id_in_opt])
            shard_name = 'buffer_{}_bucket_{}'.format(buffer_index, bucket_index)
            param_dict[shard_name] = ms.Parameter(
                param_shard.copy(),
                name=shard_name,
                requires_grad=False
            )
            param_dict['exp_avg.'+shard_name] = ms.Parameter(
                exp_avg_shard.copy(),
                name='exp_avg.'+shard_name,
                requires_grad=False
            )
            param_dict['exp_avg_sq.'+shard_name] = ms.Parameter(
                exp_avg_sq_shard.copy(),
                name='exp_avg_sq.'+shard_name,
                requires_grad=False
            )

        # add state step to state_dict
        param_dict['state_step'] = self.optimizer.state_step

        return param_dict
