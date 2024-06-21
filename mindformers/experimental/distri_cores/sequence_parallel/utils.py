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
"""Ring Attention utils."""
from mindspore import Tensor
from mindspore.communication import get_group_size, get_rank

from mindformers.experimental.distri_cores.create_comm import get_cp_rank, get_dp_world_size, get_cp_world_size

def get_sp_chuncks(batch, input_layout, enable_dp_shard=True, enable_flash_sp=False):
    """
    Slice batch input along sequence dimension into multiple chunks,
    which are parallelized across NPUs in a sequence parallel group.
    """
    sp_rank = get_cp_rank()
    world_size = get_group_size()
    dp = get_dp_world_size(with_context_parallel=False)
    sp = get_cp_world_size()

    if not isinstance(enable_flash_sp, bool):
        raise TypeError(f"The type of enable_flash_sp must be bool, but got the {type(enable_flash_sp)}")
    if not enable_flash_sp:
        if input_layout == "BSH":
            seq_dim = 1
            batch_dim = 0
        elif input_layout == "BNSD":
            seq_dim = 2
            batch_dim = 0
        elif input_layout == "SBH":
            seq_dim = 0
            batch_dim = 1
        else:
            raise ValueError(f"Only input_layout = 'BSH' or 'BNSD' or 'SBH' is supported")
    else:
        if input_layout == "BSH":
            seq_dim = 1
            batch_dim = 0
        else:
            raise ValueError(f"For FlashSP, only input_layout = 'BSH' is supported")

    if not isinstance(enable_dp_shard, bool):
        raise TypeError(f"The type of enable_dp_shard must be bool, but got the {type(enable_dp_shard)}")

    if dp * sp != world_size:
        raise ValueError(f"The product of dp and sp should be equal to total device number,"
                         f"but got dp = {dp}, sp = {sp} and total device number = {world_size}")

    seq_len = batch.shape[seq_dim]
    if seq_len < 2 * sp:
        raise ValueError(f"The sequence length of input batch should be larger or equal to 2*sp,"
                         f"but got sequence length {seq_len} and sp is {sp}")
    if seq_len % (2 * sp) != 0:
        raise ValueError(f"The sequence length of input batch is not divisible by 2*sp,"
                         f"but got sequence length {seq_len} and sp is {sp}")

    if enable_dp_shard:
        batch_sz = batch.shape[batch_dim]
        if batch_sz % dp != 0:
            raise ValueError(f"The batch size of input batch is not divisible by dp,"
                             f"but got batch_size {batch_sz} and dp is {dp}")
        if dp > 1:
            if batch_dim == 0:
                batch = batch.view(
                    dp,
                    batch.shape[batch_dim] // dp,
                    *batch.shape[(batch_dim + 1):],
                )
            else:
                batch = batch.view(
                    *batch.shape[0:batch_dim],
                    dp,
                    batch.shape[batch_dim] // dp,
                    *batch.shape[(batch_dim + 1):],
                )
            sp_group_index = get_rank() // sp
            sp_group_index = Tensor([sp_group_index])
            batch = batch.index_select(batch_dim, sp_group_index).squeeze(batch_dim)

    if sp > 1:
        if seq_dim == 0:
            batch = batch.view(
                2 * sp,
                batch.shape[seq_dim] // (2 * sp),
                *batch.shape[(seq_dim + 1):],
            )
        else:
            batch = batch.view(
                *batch.shape[0:seq_dim],
                2 * sp,
                batch.shape[seq_dim] // (2 * sp),
                *batch.shape[(seq_dim + 1):],
            )

        if enable_flash_sp:
            index = Tensor([2 * sp_rank, 2 * sp_rank + 1])
        else:
            index = Tensor([sp_rank, (2 * sp - sp_rank - 1)])
        batch = batch.index_select(seq_dim, index)

        if seq_dim == 0:
            batch = batch.view(-1, *batch.shape[(seq_dim + 2):])
        else:
            batch = batch.view(*batch.shape[0:seq_dim], -1, *batch.shape[(seq_dim + 2):])

    return batch
