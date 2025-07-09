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
# ======================

"""Model and data parallel groups."""
from typing import List, Optional, Union

from mindspore.communication import GlobalComm, create_group, destroy_group, get_group_size, get_rank
from mindspore.communication._comm_helper import _is_initialized

from mindformers.tools import logger

_WORLD_GROUP = None
_DATA_PARALLEL_GROUP = None
_TENSOR_MODEL_PARALLEL_GROUP = None
_PIPELINE_MODEL_PARALLEL_GROUP = None
_MOE_EXPERT_MODEL_PARALLEL_GROUP = None
_MOE_TENSOR_MODEL_PARALLEL_GROUP = None

_PIPELINE_GLOBAL_RANKS = None


class ProcessGroup:
    """ Communication Group Info """

    def __init__(
            self,
            group: Optional[str] = None,
            global_ranks: Optional[List[int]] = None,
            size: Optional[int] = 1,
            rank: Optional[int] = 0,
    ) -> None:
        self._group = group
        self._global_ranks = global_ranks
        self._size = size
        self._rank = rank
        self._is_group_created = False

    def reset(self) -> None:
        if self._group is not None and self._is_group_created:
            destroy_group(self._group)
        self._group = None
        self._global_ranks = None
        self._size = None
        self._rank = None
        self._is_group_created = False

    def create_this_group(self) -> None:
        create_group(self._group, self._global_ranks)
        self._is_group_created = True

    @property
    def group(self) -> str:
        if not self._is_group_created:
            raise RuntimeError("Group is not created")
        return self._group

    @property
    def size(self) -> int:
        return self._size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def is_group_created(self) -> bool:
        return self._is_group_created


default_pgs = ProcessGroup()


def generate_masked_orthogonal_rank_groups(
        world_size: int, parallel_size: List[int], mask: List[bool]
) -> List[List[int]]:
    r"""Generate orthogonal parallel groups based on the parallel size and mask.

    Arguments:
        world_size (int): world size

        parallel_size (List[int]):
            The parallel size of each orthogonal parallel type. For example, if
            tensor_parallel_size = 2, pipeline_model_parallel_group = 3, data_parallel_size = 4,
            and the parallel mapping order is tp-pp-dp, then the parallel_size = [2, 3, 4].

        mask (List[bool]):
            The mask controls which parallel methods the generated groups represent. If mask[i] is
            True, it means the generated group contains the i-th parallelism method. For example,
            if parallel_size = [tp_size, pp_size, dp_size], and mask = [True, False , True], then
            the generated group is the `tp-dp` group, if the mask = [False, True, False], then the
            generated group is the `pp` group.
    """

    def prefix_product(a: List[int], init: int = 1) -> List[int]:
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r

    def inner_product(a: List[int], b: List[int]) -> int:
        return sum(x * y for x, y in zip(a, b))

    def decompose(index: int, shape: List[int], stride: List[int] = None) -> List[int]:
        """
        This function solve the math problem below:
            There is an equation:
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will be used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        """
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        if sum(x * y for x, y in zip(idx, stride[:-1])) != index:
            raise ValueError(f"idx {index} with shape {shape} mismatch the return idx {idx}")
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size

    ranks = []
    for group_index in range(num_of_group):
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        for rank_in_group in range(group_size):
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride)
                + inner_product(decomposed_group_idx, unmasked_stride)
            )
        ranks.append(rank)
    return ranks


class RankGenerator:
    '''Generate ranks for each parallel type.'''

    def __init__(self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str) -> None:
        self.tp = tp
        self.ep = ep
        self.dp = dp
        self.pp = pp
        self.cp = cp
        if self.cp != 1:
            raise ValueError(f"For now context parallel is not supported, but got cp={cp}.")
        self.world_size = tp * dp * pp * ep

        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
        }
        self.order = order
        order = order.lower()

        for name, size in self.name_to_size.items():
            if name not in order and size != 1:
                raise RuntimeError(
                    f"The size of ({name}) is ({size}), but you haven't specified the order ({self.order})."
                )
            if name not in order:
                order = order + '-' + name

        self.order = order
        self.ordered_size = []

        for token in order.split('-'):
            self.ordered_size.append(self.name_to_size[token])

    def get_mask(self, order: str, token: str) -> List[bool]:
        """Create a mask for the specified tokens based on the given order.

        Args:
            order (str): The order of parallelism types (e.g., 'tp-dp-pp').
            token (str): The specific parallelism types to include in the mask,
                         separated by hyphens (e.g., 'tp-dp').
        """
        ordered_token = order.split('-')
        token_list = token.split('-')
        mask = [False] * len(ordered_token)
        for t in token_list:
            mask[ordered_token.index(t)] = True
        return mask

    def get_ranks(self, token: str) -> List[List[int]]:
        """Get rank group by input token.

        Args:
            token (str):
                Specify the ranks type that want to get. If we want
                to obtain multiple parallel types, we can use a hyphen
                '-' to separate them. For example, if we want to obtain
                the TP_DP group, the token should be 'tp-dp'.
        """
        mask = self.get_mask(self.order, token)
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, self.ordered_size, mask)
        return ranks


def initialize_model_parallel(tensor_model_parallel_size: Optional[int] = 1,
                              data_parallel_size: Optional[int] = 1,
                              pipeline_model_parallel_size: Optional[int] = 1,
                              expert_model_parallel_size: Optional[int] = 1,
                              order: Optional[str] = "tp-dp-pp",) -> None:
    """Initialize model data parallel groups.

    Args:
        tensor_model_parallel_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pipeline_model_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if
            tensor_model_parallel_size is 4 and
            pipeline_model_parallel_size is 2, the model will be split
            into 2 groups of 4 GPUs.

        expert_model_parallel_size (int, default = 1):
            The number of Mixture of Experts parallel GPUs in each expert
            parallel group.

        order (str, default=tp-dp-pp):
            The rank initialization order of parallelism.
    """
    if not _is_initialized():
        raise RuntimeError(
            "Distributed communication is not initialized. "
            "Please call `mindspore.communication.init` before `initialize_model_parallel`."
        )
    rank_id = get_rank()
    world_size = get_group_size()
    total_model_size = data_parallel_size * tensor_model_parallel_size * pipeline_model_parallel_size

    if world_size % expert_model_parallel_size != 0:
        raise ValueError(
            f"World size({world_size}) can not be divisible by expert parallel size({expert_model_parallel_size})")

    if world_size != total_model_size:
        raise RuntimeError(f"world_size ({world_size}) is not equal to total_model_size({total_model_size})")

    order = order.lower()
    order_list = order.split('-')
    if not order:
        raise RuntimeError("order can not be empty.")
    if len(set(order_list)) != len(order_list):
        raise RuntimeError(f"Duplicate elements in order ({order}).")

    rank_generator = RankGenerator(tp=tensor_model_parallel_size,
                                   ep=1,
                                   dp=data_parallel_size,
                                   pp=pipeline_model_parallel_size,
                                   cp=1,
                                   order=order)

    def create_process_group(token: str, group_name_prefix: str) -> Union[ProcessGroup, None]:
        """Get ProcessGroup for the specified token."""
        for ranks in rank_generator.get_ranks(token):
            if rank_id in ranks:
                group = group_name_prefix + '-' + '-'.join([str(i) for i in ranks])
                process_group = ProcessGroup(
                    group=group,
                    global_ranks=ranks,
                    size=len(ranks),
                    rank=ranks.index(rank_id),
                )
                return process_group
        return None

    world_group_info = ProcessGroup(
        group=GlobalComm.WORLD_COMM_GROUP,
        global_ranks=[i for i in range(world_size)],
        size=world_size,
        rank=rank_id,
    )
    global _WORLD_GROUP
    _WORLD_GROUP = world_group_info

    data_parallel_process_group = create_process_group('dp', 'dp')
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = data_parallel_process_group

    tensor_parallel_process_group = create_process_group('tp', 'tp')
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = tensor_parallel_process_group

    pipeline_parallel_process_group = create_process_group('pp', 'pp')
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = pipeline_parallel_process_group

    global _PIPELINE_GLOBAL_RANKS
    all_pp_ranks = rank_generator.get_ranks('pp')
    for pp_ranks in all_pp_ranks:
        if rank_id in pp_ranks:
            _PIPELINE_GLOBAL_RANKS = pp_ranks
            break

    moe_tensor_parallel_size = 1
    if expert_model_parallel_size != world_size:
        moe_tensor_parallel_size = world_size // expert_model_parallel_size
        logger.info(f"expert_model_parallel_size({expert_model_parallel_size}) "
                    f"is not equal to world_size({world_size}), "
                    f"so we will use {moe_tensor_parallel_size} as the MOE_tensor_parallel_size.")
    rank_generator = RankGenerator(tp=moe_tensor_parallel_size,
                                   ep=expert_model_parallel_size,
                                   dp=1,
                                   pp=1,
                                   cp=1,
                                   order='ep-tp')

    moe_ep_parallel_process_group = create_process_group('ep', 'moe_ep')
    global _MOE_EXPERT_MODEL_PARALLEL_GROUP
    _MOE_EXPERT_MODEL_PARALLEL_GROUP = moe_ep_parallel_process_group

    moe_tp_parallel_process_group = create_process_group('tp', 'moe_tp')
    global _MOE_TENSOR_MODEL_PARALLEL_GROUP
    _MOE_TENSOR_MODEL_PARALLEL_GROUP = moe_tp_parallel_process_group

    if get_data_parallel_world_size() > 1:
        get_data_parallel_group()
    if get_tensor_model_parallel_world_size() > 1:
        get_tensor_model_parallel_group()


def is_initialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is not None


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if (
            _TENSOR_MODEL_PARALLEL_GROUP is None
            or _PIPELINE_MODEL_PARALLEL_GROUP is None
            or _DATA_PARALLEL_GROUP is None
    ):
        return False
    return True


def _get_group_helper(process_group: ProcessGroup) -> str:
    if not process_group.is_group_created:
        process_group.create_this_group()
    return process_group


def get_world_group() -> ProcessGroup:
    """Get the world group."""
    if not is_initialized():
        raise RuntimeError('Communication is not initialized')
    return _WORLD_GROUP


def get_data_parallel_group() -> ProcessGroup:
    """Get the data parallel group the caller rank belongs to."""
    if _DATA_PARALLEL_GROUP is None:
        raise RuntimeError('Data parallel group is not initialized')
    return _get_group_helper(_DATA_PARALLEL_GROUP)


def get_tensor_model_parallel_group(check_initialized: bool = True) -> ProcessGroup:
    """Get the tensor model parallel group the caller rank belongs to."""
    if check_initialized and _TENSOR_MODEL_PARALLEL_GROUP is None:
        raise RuntimeError('Tensor model parallel group is not initialized')
    return _get_group_helper(_TENSOR_MODEL_PARALLEL_GROUP)


def get_pipeline_model_parallel_group(check_initialized: bool = True) -> ProcessGroup:
    """Get the pipeline model parallel group the caller rank belongs to."""
    if check_initialized and _PIPELINE_MODEL_PARALLEL_GROUP is None:
        raise RuntimeError('Pipeline parallel group is not initialized')
    return _get_group_helper(_PIPELINE_MODEL_PARALLEL_GROUP)


def get_moe_tensor_parallel_group(check_initialized: bool = True) -> ProcessGroup:
    """Get the moe tensor parallel group the caller rank belongs to."""
    if check_initialized and _MOE_TENSOR_MODEL_PARALLEL_GROUP is None:
        raise RuntimeError('MOE tensor parallel group is not initialized')
    return _get_group_helper(_MOE_TENSOR_MODEL_PARALLEL_GROUP)


def get_moe_expert_parallel_group(check_initialized: bool = True) -> ProcessGroup:
    """Get the moe expert parallel group the caller rank belongs to."""
    if check_initialized and _MOE_EXPERT_MODEL_PARALLEL_GROUP is None:
        raise RuntimeError('MOE tensor parallel group is not initialized')
    return _get_group_helper(_MOE_EXPERT_MODEL_PARALLEL_GROUP)


def get_tensor_model_parallel_world_size() -> int:
    """Return world size for the tensor model parallel group."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None:
        return 1
    return _TENSOR_MODEL_PARALLEL_GROUP.size


def get_data_parallel_world_size() -> int:
    """Return world size for the data parallel group."""
    if _DATA_PARALLEL_GROUP is None:
        return 1
    return _DATA_PARALLEL_GROUP.size


def get_pipeline_model_parallel_world_size() -> int:
    """Return world size for the pipeline model parallel group."""
    if _PIPELINE_MODEL_PARALLEL_GROUP is None:
        return 1
    return _PIPELINE_MODEL_PARALLEL_GROUP.size


def get_moe_tensor_parallel_world_size() -> int:
    """Return world size for the MOE tensor parallel group."""
    if _MOE_TENSOR_MODEL_PARALLEL_GROUP is None:
        return 1
    return _MOE_TENSOR_MODEL_PARALLEL_GROUP.size


def get_moe_expert_parallel_world_size() -> int:
    """Return world size for the MOE expert parallel group."""
    if _MOE_EXPERT_MODEL_PARALLEL_GROUP is None:
        return 1
    return _MOE_EXPERT_MODEL_PARALLEL_GROUP.size


def _get_rank_helper(process_group: ProcessGroup) -> int:
    if process_group.rank is not None:
        return process_group.rank
    process_group.rank = 0 if process_group.size == 1 else get_rank(group=_get_group_helper(process_group))
    return process_group.rank


def get_tensor_model_parallel_rank() -> int:
    """Return my rank for the tensor model parallel group."""
    return _get_rank_helper(_TENSOR_MODEL_PARALLEL_GROUP)


def get_data_parallel_rank() -> int:
    """Return my rank for the data parallel group."""
    return _get_rank_helper(_DATA_PARALLEL_GROUP)


def get_pipeline_model_parallel_rank() -> int:
    """Return my rank for the pipeline model parallel group."""
    return _get_rank_helper(_PIPELINE_MODEL_PARALLEL_GROUP)


def get_moe_tensor_parallel_rank() -> int:
    """Return my rank for the MOE tensor parallel group."""
    return _get_rank_helper(_MOE_TENSOR_MODEL_PARALLEL_GROUP)


def get_moe_expert_parallel_rank() -> int:
    """Return my rank for the MOE expert parallel group."""
    return _get_rank_helper(_MOE_EXPERT_MODEL_PARALLEL_GROUP)


def get_pipeline_model_parallel_first_rank() -> int:
    """Return the global rank of the first precess in the pipeline"""
    if _PIPELINE_GLOBAL_RANKS is None:
        raise Exception("Pipeline parallel group is not initialized")
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank() -> int:
    """Return the global rank of the last precess in the pipeline"""
    if _PIPELINE_GLOBAL_RANKS is None:
        raise Exception("Pipeline parallel group is not initialized")
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_prev_rank() -> int:
    """Return the global rank that precedes the caller in the pipeline"""
    if _PIPELINE_GLOBAL_RANKS is None:
        raise Exception("Pipeline parallel group is not initialized")
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_pipeline_model_parallel_next_rank() -> int:
    """Return the global rank that follows the caller in the pipeline"""
    if _PIPELINE_GLOBAL_RANKS is None:
        raise Exception("Pipeline parallel group is not initialized")
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def is_pipeline_first_stage() -> bool:
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage() -> bool:
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)
