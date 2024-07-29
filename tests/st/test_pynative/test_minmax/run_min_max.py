import argparse
from enum import Enum

import numpy as np
from mindspore import Tensor, nn, ops, Parameter
from mindspore.communication import init

from mindformers.experimental.distri_cores.config import ParallelConfig
from mindformers.experimental.distri_cores.create_comm import (
    get_tp_rank, get_tp_world_size, initialize_model_parallel)
from mindformers.experimental.distri_cores.tensor_parallel import (
    ColumnParallelLinear, RowParallelLinear)
from mindformers.experimental.distri_cores.tensor_parallel.collective_primitives import (
    MaxFromTensorParallelRegion, MinFromTensorParallelRegion)
from mindformers.modules import Linear

last_axis = -1
second_last_axis = -2


class PARALLEL_LINEAR_TYPE(Enum):
    COLUMNPARALLELLINEAR = 0
    ROWPARALLELLINEAR = 1


def get_columnparallellinear(in_channels, out_channels, config, transpose_b):
    return ColumnParallelLinear(in_channels, out_channels, config, transpose_b=transpose_b)


def get_rowparallellinear(in_channels, out_channels, config, transpose_b):
    return RowParallelLinear(in_channels, out_channels, config, input_is_parallel=False, transpose_b=transpose_b)


parallel_linear_helper = {PARALLEL_LINEAR_TYPE.COLUMNPARALLELLINEAR: get_columnparallellinear,
                          PARALLEL_LINEAR_TYPE.ROWPARALLELLINEAR: get_rowparallellinear}


class LinearWrapper(nn.Cell):
    def __init__(self, linear, reduce_out, keepdims):
        super().__init__()
        self.proj = linear
        if reduce_out:
            # Whether to reduce out_channel.
            self.reduce_axis = second_last_axis if self.proj.transpose_b else last_axis
        else:
            self.reduce_axis = last_axis if self.proj.transpose_b else second_last_axis
        self.keepdims = keepdims
        
        self.max = ops.max
        self.min = ops.min

    def get_weight_max_min(self):
        weight_max = self.max(self.proj.weight, self.reduce_axis, self.keepdims)[0]
        weight_min = self.min(self.proj.weight, self.reduce_axis, self.keepdims)[0]
        return weight_max, weight_min


class ParallelLinearWrapper(nn.Cell):
    def __init__(self, linear, reduce_out, keepdims):
        super().__init__()
        self.proj = linear
        if reduce_out:  # Whether to reduce out_channel.
            self.reduce_axis = second_last_axis if self.proj.transpose_b else last_axis
        else:
            self.reduce_axis = last_axis if self.proj.transpose_b else second_last_axis
        self.keepdims = keepdims

        if isinstance(self.proj, ColumnParallelLinear):  # ColumnParallelLinear splits weight in out dimension.
            self.max = MaxFromTensorParallelRegion() if reduce_out else ops.max
            self.min = MinFromTensorParallelRegion() if reduce_out else ops.min
        if isinstance(self.proj, RowParallelLinear):  # RowParallelLinear splits weight in in dimension.
            self.max = ops.max if reduce_out else MaxFromTensorParallelRegion()
            self.min = ops.min if reduce_out else MinFromTensorParallelRegion()

    def get_weight_max_min(self):
        weight_max = self.max(self.proj.weight, self.reduce_axis, self.keepdims)[0]
        weight_min = self.min(self.proj.weight, self.reduce_axis, self.keepdims)[0]
        return weight_max, weight_min


def gen_full_weight(shape):
    """Get full weight. """
    np.random.seed(2024)
    weight = Parameter(np.random.randn(*shape).astype(np.float32))
    return weight


def get_distri_weight(weight, parallel_linear_type, transpose_b):
    """ Get distributed input from tensor-parallel region. """
    tp_size = get_tp_world_size()
    tp_rank = get_tp_rank()
    if parallel_linear_type == PARALLEL_LINEAR_TYPE.COLUMNPARALLELLINEAR:
        axis = second_last_axis if transpose_b else last_axis
    else:
        axis = last_axis if transpose_b else second_last_axis
    weight_chunks = ops.chunk(weight, tp_size, axis)
    param = Parameter(weight_chunks[tp_rank])
    return param


def test_parallel_min_max(parallel_linear_type, transpose_b=True, reduce_out=True, keepdims=False):
    """ Comparative testing of standalone operators and distributed operators. """
    parallel_config = ParallelConfig(tensor_parallel=2, use_zero3=False)
    init()
    initialize_model_parallel(tp_size=parallel_config.tensor_parallel)

    in_channels, out_channels = 48, 32
    weight_shape = (out_channels, in_channels)if transpose_b else (in_channels, out_channels)
    single_linear = Linear(in_channels, out_channels, transpose_b=transpose_b)
    weight = gen_full_weight(weight_shape)
    single_linear.weight.set_data(weight)

    sinlge_wrapper = LinearWrapper(single_linear, reduce_out=reduce_out, keepdims=keepdims)
    golden_max, golden_min = sinlge_wrapper.get_weight_max_min()

    parallel_linear = parallel_linear_helper[parallel_linear_type](
        in_channels, out_channels, config=parallel_config, transpose_b=transpose_b)
    weight_distri = get_distri_weight(weight, parallel_linear_type, parallel_linear.transpose_b)
    parallel_linear.weight.set_data(weight_distri)

    parallel_wrapper = ParallelLinearWrapper(parallel_linear, reduce_out=reduce_out, keepdims=keepdims)
    max, min = parallel_wrapper.get_weight_max_min()

    assert (golden_max == max).all()
    assert (golden_min == min).all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["column", "row"], help="get mode of parallel linear.")
    parser.add_argument("--transpose_b", action="store_true", help="get transpose_b for linear.")
    parser.add_argument("--reduce_out", action="store_true", help="get reduce_out for wrapper.")
    parser.add_argument("--keepdims", action="store_true", help="get min or max with keepdims.")

    args, rest_args = parser.parse_known_args()

    parallel_linear_type = PARALLEL_LINEAR_TYPE.COLUMNPARALLELLINEAR if args.mode == 'column' else PARALLEL_LINEAR_TYPE.ROWPARALLELLINEAR

    test_parallel_min_max(parallel_linear_type, args.transpose_b, args.reduce_out, args.keepdims)
