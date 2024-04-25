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
from mindspore.communication import get_rank, get_group_size, create_group

_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_GLOBAL_RANKS = None
_SEQUENCE_PARALLEL_GROUP_INDEX = None

def init_sp_group(
        sp=1,
) -> None:
    """Initialize parallel groups."""
    world_size = get_group_size()
    if not isinstance(sp, int):
        raise TypeError(f"The input of sp must be int, but get the type of {type(sp)}")
    if sp > world_size:
        raise ValueError(f"The sp must be smaller or equal to total device_num, but got the sp is {sp},"
                         f"the total device_num is {world_size}")
    if sp&(sp-1) != 0:
        raise ValueError(f"The sp value must be power of two, but got sp is {sp}")

    dp = world_size // sp
    # Build the context-parallel groups.
    global _SEQUENCE_PARALLEL_GROUP
    global _SEQUENCE_PARALLEL_GLOBAL_RANKS
    global _SEQUENCE_PARALLEL_GROUP_INDEX
    # assert _SEQUENCE_PARALLEL_GROUP is None, 'sequence parallel group is already initialized'

    for j in range(dp):
        start_rank = j * sp
        end_rank = (j + 1) * sp
        ranks = list(range(start_rank, end_rank))
        cur_rank = get_rank()
        if cur_rank in ranks:
            sp_group = '_'.join(str(n) for n in ranks)
            sp_group_name = "sp_group_" + sp_group
            create_group(sp_group_name, ranks)
            _SEQUENCE_PARALLEL_GROUP = sp_group_name
            _SEQUENCE_PARALLEL_GLOBAL_RANKS = ranks
            _SEQUENCE_PARALLEL_GROUP_INDEX = j


def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    return _SEQUENCE_PARALLEL_GROUP

def get_sequence_parallel_global_ranks():
    """Get all global ranks of the sequence parallel group that the caller rank belongs to."""
    return _SEQUENCE_PARALLEL_GLOBAL_RANKS

def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    return get_group_size(group=get_sequence_parallel_group())

def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    return get_rank(group=get_sequence_parallel_group())

def get_sequence_parallel_group_index():
    """Get the sequence parallel group index the caller rank belongs to."""
    return _SEQUENCE_PARALLEL_GROUP_INDEX

def get_sp_chuncks(batch, dp, sp, seq_dim=0, batch_dim=1, enable_dp_shard=True):
    """
    Slice batch input along sequence dimension into multiple chunks,
    which are parallelized across NPUs in a sequence parallel group.
    """
    sp_rank = get_sequence_parallel_rank()
    world_size = get_group_size()
    if not isinstance(dp, int):
        raise TypeError(f"The type of dp must be int, but got the {type(dp)}")
    if not isinstance(sp, int):
        raise TypeError(f"The type of sp must be int, but got the {type(sp)}")
    if not isinstance(seq_dim, int):
        raise TypeError(f"The type of seq_dim must be int, but got the {type(seq_dim)}")
    if not isinstance(batch_dim, int):
        raise TypeError(f"The type of batch_dim must be int, but got the {type(batch_dim)}")
    if not isinstance(enable_dp_shard, bool):
        raise TypeError(f"The type of enable_dp_shard must be bool, but got the {type(enable_dp_shard)}")
    if dp > world_size:
        raise ValueError(f"The value of dp must be smaller or equal word_size, but got the dp is {dp},"
                         f"the world_size is {world_size}")
    if sp > world_size:
        raise ValueError(f"The value of sp must be smaller or equal word_size, but got the dsp is {sp},"
                         f"the world_size is {world_size}")
    if enable_dp_shard:
        if dp * sp != world_size:
            raise ValueError(f"The product of dp and sp should be equal to total device number,"
                             f"but got dp = {dp}, sp = {sp} and total device number = {world_size}")

        seq_len = batch.shape[seq_dim]
        batch_sz = batch.shape[batch_dim]
        if seq_len % (2 * sp) != 0:
            raise ValueError(f"The sequence length of input batch is not divisible by 2*sp,"
                             f"but got sequence length {seq_len} and sp is {sp}")
        if batch_sz % dp != 0:
            raise ValueError(f"The batch size of input batch is not divisible by dp,"
                             f"but got batch_size {batch_sz} and dp is {dp}")

        init_sp = get_sequence_parallel_world_size()
        if sp != init_sp:
            raise ValueError(f"The sp group is initialized as {init_sp},"
                             f"but got different sp = {sp} in the input parameters")

        sp_group_index = get_sequence_parallel_group_index()
        world_size = get_group_size()
        dp = world_size // sp
        if dp > 1:
            if batch_dim == 0:
                batch = batch.view(
                    dp,
                    batch.shape[batch_dim] // dp,
                    *batch.shape[(batch_dim + 1) :],
                    )
            else:
                batch = batch.view(
                    *batch.shape[0:batch_dim],
                    dp,
                    batch.shape[batch_dim] // dp,
                    *batch.shape[(batch_dim + 1) :],
                    )
            sp_group_index = Tensor([sp_group_index])
            batch = batch.index_select(batch_dim, sp_group_index).squeeze(batch_dim)

    if sp > 1:
        if seq_dim == 0:
            batch = batch.view(
                2 * sp,
                batch.shape[seq_dim] // (2 * sp),
                *batch.shape[(seq_dim + 1) :],
                )
        else:
            batch = batch.view(
                *batch.shape[0:seq_dim],
                2 * sp,
                batch.shape[seq_dim] // (2 * sp),
                *batch.shape[(seq_dim + 1) :],
                )

        index = Tensor([sp_rank, (2 * sp - sp_rank - 1)])
        batch = batch.index_select(seq_dim, index)

        if seq_dim == 0:
            batch = batch.view(-1, *batch.shape[(seq_dim + 2) :])
        else:
            batch = batch.view(*batch.shape[0:seq_dim], -1, *batch.shape[(seq_dim + 2) :])

    return batch
