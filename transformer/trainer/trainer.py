# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
Used for gradient update. We want to use custom dtype for allreduce in data parallel.
"""
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer, _get_datatype, reduce_opt, _cast_datatype
from mindspore.ops import functional as F
from mindspore.nn.wrap.loss_scale import TrainOneStepWithLossScaleCell
import mindspore.common.dtype as mstype

from mindspore.common.api import ms_function


class CustomGradTypeDistributedGradReducer(DistributedGradReducer):
    """We can use set_dtype to control the communication dtype. The other parts are same."""
    def __init__(self, *args, **kwargs):
        super(CustomGradTypeDistributedGradReducer, self).__init__(*args, **kwargs)
        self.dtype = mstype.float32

    def set_dtype(self, dtype):
        self.dtype = dtype

    @ms_function
    def construct(self, grads):
        """
        Under certain circumstances, the data precision of grads could be mixed with float16 and float32. Thus, the
        result of AllReduce is unreliable. To solve the problem, grads must be cast to float32 before AllReduce,
        and cast back after the operation.

        Args:
            grads (Union[Tensor, tuple[Tensor]]): The gradient tensor or tuple before operation.

        Returns:
            new_grads (Union[Tensor, tuple[Tensor]]), the gradient tensor or tuple after operation.
        """
        datatypes = self.map_(F.partial(_get_datatype), grads)
        grads = self.map_(F.partial(_cast_datatype, self.dtype), grads)
        if self.is_pynative_parallel:
            new_grad = self.map_(F.partial(reduce_opt, self.degree, self.mean), self.allreduce_filter, grads)
        elif self.split_fusion:
            if self.enable_parameter_server:
                new_grad = self.map_(F.partial(reduce_opt, self.degree, self.mean, self.allgather),
                                     self.op_list, self.allreduce_filter, grads, self.ps_parameters)
            else:
                new_grad = self.map_(F.partial(reduce_opt, self.degree, self.mean, self.allgather),
                                     self.op_list, self.allreduce_filter, grads)
        else:
            if self.enable_parameter_server:
                new_grad = self.map_(F.partial(reduce_opt, self.degree, self.mean, self.allgather,
                                               self.allreduce), self.allreduce_filter, grads, self.ps_parameters)
            else:
                new_grad = self.map_(F.partial(reduce_opt, self.degree, self.mean, self.allgather,
                                               self.allreduce), self.allreduce_filter, grads)
        new_grad = self.map_(F.partial(_cast_datatype), datatypes, new_grad)
        return new_grad


class TrainOneStepGradWithLossScaleCell(TrainOneStepWithLossScaleCell):
    def set_custom_sync_dtype(self, dtype):
        if self.reducer_flag:
            # Overwrite the Grad Reducer to make it sync gradients in float32 or float16
            self.grad_reducer = CustomGradTypeDistributedGradReducer(self.weights, self.mean, self.degree)
            self.grad_reducer.set_dtype(dtype)
