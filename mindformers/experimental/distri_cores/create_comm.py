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

import warnings

import mindspore
from mindspore.communication import create_group, get_group_size, get_rank
from mindspore import hal

_TP_GROUP = None
_PP_GROUP = None
_MODEL_PARALLEL_GROUP = None
_EMBEDDING_GROUP = None
_POSITION_EMBEDDING_GROUP = None
_DP_GROUP = None
_TENSOR_AND_DP_GROUP = None
_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_TENSOR_MODEL_PARALLEL_RANK = None
_PIPELINE_MODEL_PARALLEL_RANK = None
_EMBEDDING_GLOBAL_RANKS = None

_PIPELINE_GLOBAL_RANKS = None
_DATA_PARALLEL_GLOBAL_RANKS = None
_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None
_CP_GROUP = None
_CONTEXT_PARALLEL_GLOBAL_RANKS = None
_DP_GROUP_WITH_CP = None
_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = None
_TENSOR_AND_DP_GROUP_WITH_CP = None
_GLOBAL_STREAM = None

_IS_INITIALIZED = False

_SP_SEND_STREAM = None


class CreateCommGroups():
    '''Generate ranks for each parallel type.'''

    def __init__(self, tp, ep, dp, pp, cp, order):
        self.tp = tp
        self.ep = ep
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.world_size = tp * dp * pp * cp

        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "cp": self.cp,
        }
        self.rank = get_rank()
        self.order = order

        for name, size in self.name_to_size.items():
            if name not in order:
                if size == 1:
                    order = order + '-' + name
                else:
                    raise RuntimeError(
                        f"The size of ({name}) is ({size}), \
                        but you haven't specified the order ({self.order})."
                    )

        self.order_w_ep = order
        self.order_wo_ep = '-'.join([token for token in order.split('-') if token != 'ep'])
        self.ordered_size_wo_ep = []
        self.ordered_size_w_ep = []

        for token in order.split('-'):
            if token == 'dp':
                self.ordered_size_w_ep.append(self.dp // self.ep)
                self.ordered_size_wo_ep.append(self.dp)
            elif token == 'ep':
                self.ordered_size_w_ep.append(self.ep)
            else:
                self.ordered_size_w_ep.append(self.name_to_size[token])
                self.ordered_size_wo_ep.append(self.name_to_size[token])

    def get_mask(self, order, token):
        ordered_token = order.split('-')
        token = token.split('-')
        mask = [False] * len(ordered_token)
        for t in token:
            mask[ordered_token.index(t)] = True
        return mask

    def get_ranks(self, token, independent_ep=False):
        '''Get rank group by input token.

        Arguments:
            token (str): Specify the ranks type that want to get. Use a hyphen '-' to separate multiple parallel types.
            independent_ep (bool): Whether to treat EP and DP independently. Default: False.
        '''
        if independent_ep:
            parallel_size = self.ordered_size_w_ep
            order = self.order_w_ep
        else:
            parallel_size = self.ordered_size_wo_ep
            order = self.order_wo_ep
        mask = self.get_mask(order, token)
        ranks = self._dispatch_comm_ranks(self.world_size, parallel_size, mask)
        return ranks

    def create_dp_group(self):
        '''Create data parallel group.'''
        global _DP_GROUP
        global _DATA_PARALLEL_GLOBAL_RANKS

        assert _DP_GROUP is None, 'data parallel group is already initialized'

        for ranks in self.get_ranks('dp'):
            if self.rank in ranks:
                group = 'dp-' + '-'.join([str(i) for i in ranks])
                create_group(group, ranks)
                _DP_GROUP = group
                _DATA_PARALLEL_GLOBAL_RANKS = ranks

    def create_dp_cp_group(self):
        global _DP_GROUP_WITH_CP
        global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
        for ranks_with_cp in self.get_ranks('dp-cp'):
            if self.rank in ranks_with_cp:
                group_with_cp = 'dp-cp-' + '-'.join([str(i) for i in ranks_with_cp])
                create_group(group_with_cp, ranks_with_cp)
                _DP_GROUP_WITH_CP = group_with_cp
                _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

    def create_cp_group(self):
        '''Create context parallel group.'''
        global _CP_GROUP
        global _CONTEXT_PARALLEL_GLOBAL_RANKS
        assert _CP_GROUP is None, 'context parallel group is already initialized'

        for ranks in self.get_ranks('cp'):
            if self.rank in ranks:
                group = 'cp-' + '-'.join([str(i) for i in ranks])
                create_group(group, ranks)
                _CP_GROUP = group
                _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks

    def create_tp_pp_group(self):
        global _MODEL_PARALLEL_GROUP
        assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'

        for ranks in self.get_ranks('tp-pp'):
            if self.rank in ranks:
                group = 'tp-pp-' + '-'.join([str(i) for i in ranks])
                create_group(group, ranks)
                _MODEL_PARALLEL_GROUP = group

    def create_tp_group(self):
        '''Create tensor parallel group.'''
        global _TP_GROUP
        global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
        assert (_TP_GROUP is None
                ), 'tensor model parallel group is already initialized'
        for ranks in self.get_ranks('tp'):
            if self.rank in ranks:
                group = 'tp-' + '-'.join([str(i) for i in ranks])
                create_group(group, ranks)
                _TP_GROUP = group
                _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks

    def create_pp_group(self, pp_split_rank):
        '''Create pipeline parallel group.'''
        global _PP_GROUP
        global _PIPELINE_GLOBAL_RANKS
        assert (
            _PP_GROUP is None
        ), 'pipeline model parallel group is already initialized'
        global _EMBEDDING_GROUP
        global _EMBEDDING_GLOBAL_RANKS
        assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'
        global _POSITION_EMBEDDING_GROUP
        assert _POSITION_EMBEDDING_GROUP is None, 'position embedding group is already initialized'
        for ranks in self.get_ranks('pp'):
            if self.rank in ranks:
                group = 'pp-' + '-'.join([str(i) for i in ranks])
                create_group(group, ranks)
                _PP_GROUP = group
                _PIPELINE_GLOBAL_RANKS = ranks
            # Setup embedding group (to exchange gradients between
            # first and last stages).
            if len(ranks) > 1:
                embedding_ranks = [ranks[0], ranks[-1]]
                position_embedding_ranks = [ranks[0]]
                if pp_split_rank is not None:
                    if ranks[pp_split_rank] not in embedding_ranks:
                        embedding_ranks = [
                            ranks[0],
                            ranks[pp_split_rank],
                            ranks[-1],
                        ]
                    if ranks[pp_split_rank] not in position_embedding_ranks:
                        position_embedding_ranks = [ranks[0], ranks[pp_split_rank]]
            else:
                embedding_ranks = ranks
                position_embedding_ranks = ranks

            if self.rank in embedding_ranks:
                group = 'embedding-' + '-'.join([str(i) for i in ranks])
                create_group(group, embedding_ranks)
                _EMBEDDING_GROUP = group
            if self.rank in ranks:
                _EMBEDDING_GLOBAL_RANKS = embedding_ranks

            if self.rank in position_embedding_ranks:
                group = 'position_embedding-' + '-'.join([str(i) for i in position_embedding_ranks])
                create_group(group, position_embedding_ranks)
                _POSITION_EMBEDDING_GROUP = group

    def create_tp_dp_cp_group(self):
        global _TENSOR_AND_DP_GROUP_WITH_CP
        for ranks in self.get_ranks('tp-dp-cp'):
            if self.rank in ranks:
                group = 'tp-dp-cp-' + '-'.join([str(i) for i in ranks])
                create_group(group, ranks)
                _TENSOR_AND_DP_GROUP_WITH_CP = group

    def create_tp_dp_group(self):
        '''Create tensor parallel and data parallel group.'''
        global _TENSOR_AND_DP_GROUP

        assert (
            _TENSOR_AND_DP_GROUP is None
        ), 'Tensor + data parallel group is already initialized'
        for ranks in self.get_ranks('tp-dp'):
            if self.rank in ranks:
                group = 'tp-dp-' + '-'.join([str(i) for i in ranks])
                create_group(group, ranks)
                _TENSOR_AND_DP_GROUP = group

    def _dispatch_comm_ranks(self, world_size, parallel_size, mask):
        """dispatch comm ranks"""
        def prefix_product(a, init=1):
            r = [init]
            for v in a:
                init = init * v
                r.append(init)
            return r

        def modulo(index, shape, stride=None):
            if stride is None:
                stride = prefix_product(shape)
            idx = [(index // d) % s for s, d in zip(shape, stride)]
            assert (
                sum([x * y for x, y in zip(idx, stride[:-1])]) == index
            ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)
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
            decomposed_group_idx = modulo(group_index, unmasked_shape)
            rank = []
            for rank_in_group in range(group_size):
                # get indices from masked for rank_in_group.
                decomposed_rank_idx = modulo(rank_in_group, masked_shape)
                masked_inner_product = sum([x * y for x, y in zip(decomposed_rank_idx, masked_stride)])
                unmasked_inner_product = sum([x * y for x, y in zip(decomposed_group_idx, unmasked_stride)])
                rank.append(masked_inner_product + unmasked_inner_product)
            ranks.append(rank)
        return ranks


def initialize_model_parallel(tp_size=1,
                              pp_size=1,
                              virtual_pp_size=None,
                              pp_split_rank=None,
                              cp_size=1,
                              ep_size=1,
                              order="tp-cp-ep-dp-pp"):
    """Initialize model data parallel groups.
    """

    # pylint: disable=W0212
    assert mindspore.communication._comm_helper._is_initialized()
    world_size = get_group_size()

    if world_size % (tp_size * pp_size * cp_size) != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tp_size "
            f"({tp_size}) x pp_size ({pp_size}) "
            f"x cp_size ({cp_size})"
        )

    dp_size = world_size // (tp_size * pp_size * cp_size)

    if dp_size % ep_size != 0:
        raise RuntimeError(
            f"dp_size ({dp_size}) is not divisible by ep_size "
        )

    if ep_size > 1 and cp_size > 1:
        raise RuntimeError(
            f"combination of expert model prallellism and context parallelism is not supported"
        )

    if virtual_pp_size is not None:
        if pp_size <= 2:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
            )

    order = order.lower()
    order_list = order.split('-')
    if not order:
        raise RuntimeError(f"order can not be empty.")
    if len(set(order_list)) != len(order_list):
        raise RuntimeError(f"Duplicate elements in order ({order}).")
    if 'ep' in order:
        if 'ep-dp' not in order and 'dp-ep' not in order:
            raise RuntimeError(f"The ep and dp must be adjacent in order ({order}).")

    rank_generator = CreateCommGroups(tp=tp_size, ep=ep_size, dp=dp_size, pp=pp_size,
                                      cp=cp_size, order=order)
    # Build the data-parallel groups.
    if 'dp' in order:
        rank_generator.create_dp_group()

        if 'cp' in order:
            rank_generator.create_dp_cp_group()

    # Build the context-parallel groups.
    if 'cp' in order:
        rank_generator.create_cp_group()

    # Build the model-parallel groups.
    if 'tp' in order and 'pp' in order:
        rank_generator.create_tp_pp_group()

    # Build the tensor model-parallel groups.
    if 'tp' in order:
        rank_generator.create_tp_group()

    # Build the pipeline-parallel groups.
    if 'pp' in order:
        rank_generator.create_pp_group(pp_split_rank)

    # Build the tensor + data parallel groups.
    if 'tp' in order and 'dp' in order:
        rank_generator.create_tp_dp_group()
        if 'cp' in order:
            rank_generator.create_tp_dp_cp_group()

    global _GLOBAL_STREAM
    assert (_GLOBAL_STREAM is None), 'Global stream is already initialized'
    _GLOBAL_STREAM = hal.Stream()

    global _SP_SEND_STREAM
    if cp_size > 1:
        _SP_SEND_STREAM = hal.Stream()

    global _IS_INITIALIZED
    _IS_INITIALIZED = True


def is_initialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _IS_INITIALIZED


def is_unitialized() -> bool:
    """Check if parallel state has been initialized

    Deprecated. Use is_initialized instead.

    """
    warnings.warn(
        "is_unitialized is deprecated, use is_initialized instead", DeprecationWarning,
    )
    return not is_initialized()


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, \
            ("model parallel group is not initialized. Please check whether communication "
             "is initialized and 'tp-pp' in order.")
    return _MODEL_PARALLEL_GROUP


# pylint: disable=C0330
def get_tp_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    if check_initialized:
        assert _TP_GROUP is not None, \
            ("tensor parallel group is not initialized. Please check whether communication "
             "is initialized and 'tp' in order.")
    return _TP_GROUP


def get_pp_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert _PP_GROUP is not None, \
        ("pipeline parallel group is not initialized. Please check whether communication "
         "is initialized and 'pp' in order.")
    return _PP_GROUP


# pylint: disable=C0330
def get_dp_group(with_context_parallel=False):
    """Get the data parallel group the caller rank belongs to."""
    ret = None
    if with_context_parallel:
        assert _DP_GROUP_WITH_CP is not None, \
            ("data parallel group with context parallel combined is not initialized. "
             "Please check whether communication is initialized and 'dp-cp' in order.")
        ret = _DP_GROUP_WITH_CP
    else:
        assert _DP_GROUP is not None, \
            ("data parallel group is not initialized. Please check whether communication "
             "is initialized and 'dp' in order.")
        ret = _DP_GROUP
    return ret


def get_cp_group(check_initialized=True):
    """Get the context parallel group the caller rank belongs to."""
    if check_initialized:
        assert _CP_GROUP is not None, \
            ("context parallel group is not initialized. Please check whether communication "
             "is initialized and 'cp' in order.")
    return _CP_GROUP


# pylint: disable=C0330
def get_cp_global_ranks(check_initialized=True):
    """Get all global ranks of the context parallel group that the caller rank belongs to."""
    if check_initialized:
        assert _CONTEXT_PARALLEL_GLOBAL_RANKS is not None, \
            ("context parallel group is not initialized. Please check whether communication "
             "is initialized and 'cp' in order.")
    return _CONTEXT_PARALLEL_GLOBAL_RANKS


def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP is not None, \
        ("pipeline parallel group is not initialized. Please check whether communication "
         "is initialized and 'pp' in order.")
    return _EMBEDDING_GROUP

def get_tp_world_size():
    """Return world size for the tensor model parallel group."""
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return get_group_size(group=get_tp_group())


def get_pp_world_size():
    """Return world size for the pipeline model parallel group."""
    global _PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return get_group_size(group=get_pp_group())


def get_tp_rank():
    """Return my rank for the tensor model parallel group."""
    global _TENSOR_MODEL_PARALLEL_RANK
    if _TENSOR_MODEL_PARALLEL_RANK is not None:
        return _TENSOR_MODEL_PARALLEL_RANK
    return get_rank(group=get_tp_group())


def get_pp_rank():
    """Return my rank for the pipeline model parallel group."""
    global _PIPELINE_MODEL_PARALLEL_RANK
    if _PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _PIPELINE_MODEL_PARALLEL_RANK
    return get_rank(group=get_pp_group())


def is_pipeline_first_stage():
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    return get_pp_rank() == 0


def is_pipeline_last_stage():
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    return get_pp_rank() == (get_pp_world_size() - 1)


def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    ret = False
    rank = get_rank()
    global _EMBEDDING_GLOBAL_RANKS
    if ignore_virtual:
        return rank in _EMBEDDING_GLOBAL_RANKS
    if rank in _EMBEDDING_GLOBAL_RANKS:
        if rank == _EMBEDDING_GLOBAL_RANKS[0]:
            ret = is_pipeline_first_stage()
        elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
            ret = is_pipeline_last_stage()
        else:
            ret = True
    return ret


def get_dp_world_size(with_context_parallel=False):
    """Return world size for the data parallel group."""

    return get_group_size(group=get_dp_group(with_context_parallel=with_context_parallel))


def get_dp_rank(with_context_parallel=False):
    """Return my rank for the data parallel group."""
    return get_rank(group=get_dp_group(with_context_parallel=with_context_parallel))


def get_cp_world_size():
    """Return world size for the context parallel group."""
    return get_group_size(group=get_cp_group())


def get_cp_rank():
    """Return my rank for the context parallel group."""
    return get_rank(group=get_cp_group())


def get_stream():
    """Return global stream. There is only one stream for each npu."""
    assert _GLOBAL_STREAM is not None, "Global stream is not initialized"
    return _GLOBAL_STREAM


def get_sp_send_stream():
    """Return send stream for sequence parallel."""
    assert _SP_SEND_STREAM is not None, "Sp send stream is not initialized"
    return _SP_SEND_STREAM


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _TP_GROUP
    _TP_GROUP = None
    global _PP_GROUP
    _PP_GROUP = None
    global _DP_GROUP
    _DP_GROUP = None
    global _DP_GROUP_WITH_CP
    _DP_GROUP_WITH_CP = None
    global _DATA_PARALLEL_GLOBAL_RANKS
    _DATA_PARALLEL_GLOBAL_RANKS = None
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = None
    global _CP_GROUP
    _CP_GROUP = None
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    _CONTEXT_PARALLEL_GLOBAL_RANKS = None
    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = None
    global _EMBEDDING_GLOBAL_RANKS
    _EMBEDDING_GLOBAL_RANKS = None
    global _POSITION_EMBEDDING_GROUP
    _POSITION_EMBEDDING_GROUP = None
    global _TENSOR_AND_DP_GROUP
    _TENSOR_AND_DP_GROUP = None
    global _TENSOR_AND_DP_GROUP_WITH_CP
    _TENSOR_AND_DP_GROUP_WITH_CP = None
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    global _PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _PIPELINE_GLOBAL_RANKS
    _PIPELINE_GLOBAL_RANKS = None
    global _TENSOR_MODEL_PARALLEL_RANK
    _TENSOR_MODEL_PARALLEL_RANK = None
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None
    global _PIPELINE_MODEL_PARALLEL_RANK
    _PIPELINE_MODEL_PARALLEL_RANK = None
    global _GLOBAL_STREAM
    _GLOBAL_STREAM = None
    global _SP_SEND_STREAM
    _SP_SEND_STREAM = None
    global _IS_INITIALIZED
    _IS_INITIALIZED = False
