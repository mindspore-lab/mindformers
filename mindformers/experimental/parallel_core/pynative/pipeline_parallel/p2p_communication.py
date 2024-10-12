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
"""p2p primitives."""

import numpy as np

import mindspore as ms
from mindspore import nn, Tensor
import mindspore.communication.comm_func as comm_func

from mindformers.experimental.parallel_core.pynative.parallel_state import get_pipeline_model_parallel_group, \
    get_pipeline_model_parallel_rank, is_pipeline_last_stage, is_pipeline_first_stage, \
    get_pipeline_model_parallel_prev_rank, get_pipeline_model_parallel_next_rank


class ISend(nn.Cell):
    """ Send a tensor asynchronously """
    def __init__(self, sr_tag, dest_rank, group):
        super(ISend, self).__init__()
        self.group = group
        self.dst_rank = dest_rank
        self.src_tag = sr_tag

    def construct(self, send_data):
        """ ISend forward """
        handle = comm_func.isend(send_data, dst=self.dst_rank, tag=self.src_tag, group=self.group)
        return handle


class IRecv(nn.Cell):
    """ receive a tensor asynchronously """
    def __init__(self, sr_tag, src_rank, shape, dtype, group):
        super(IRecv, self).__init__()
        self.data = Tensor(np.zeros(shape), dtype=dtype)
        self.group = group
        self.src_rank = src_rank
        self.src_tag = sr_tag

    def construct(self):
        """ IRecv forward """
        recv_tensor, handle = comm_func.irecv(self.data, src=self.src_rank, tag=self.src_tag, group=self.group)
        return handle, recv_tensor


# pylint: disable=C0103
class P2PPrimitive():
    """ A class that includes P2P communication methods """
    def __init__(self, config):
        self.config = config


    def send_forward(self,
                     output_tensor):
        """
        Except for the last stage, send forward tensor.
        """
        if not is_pipeline_last_stage():
            self.node_p2p_comm(
                tensor_send_next=output_tensor,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                tensor_shape=None
            )


    def recv_forward(self,
                     tensor_shape):
        """
        Except for the first stage, send forward tensor.
        """
        if is_pipeline_first_stage():
            input_tensor = None
        else:
            input_tensor, _, _ = self.node_p2p_comm(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                tensor_shape=tensor_shape
            )
        return input_tensor


    def send_backward(self,
                      input_tensor_grad):
        """
        Except for the first stage, send backward tensor.
        """
        if not is_pipeline_first_stage():
            self.node_p2p_comm(
                tensor_send_next=None,
                tensor_send_prev=input_tensor_grad,
                recv_prev=False,
                recv_next=False,
                tensor_shape=None
            )


    def recv_backward(self,
                      tensor_shape):
        """
        Except for the last stage, recv backward tensor.
        """
        if is_pipeline_last_stage():
            output_tensor_grad = None
        else:
            _, output_tensor_grad, _ = self.node_p2p_comm(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                tensor_shape=tensor_shape
            )
        return output_tensor_grad


    def send_forward_recv_backward(self,
                                   output_tensor,
                                   tensor_shape):
        """
        Except for the last stage, send forward tensor, while receiving backward tensor.
        """
        if is_pipeline_last_stage():
            output_tensor_grad = None
        else:
            _, output_tensor_grad, _ = self.node_p2p_comm(
                tensor_send_next=output_tensor,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                tensor_shape=tensor_shape
            )
        return output_tensor_grad


    def send_backward_recv_forward(self,
                                   input_tensor_grad,
                                   tensor_shape):
        """
        Except for the first stage, send backward tensor, while receiving forward tensor.
        """
        if is_pipeline_first_stage():
            input_tensor = None
        else:
            input_tensor, _, _ = self.node_p2p_comm(
                tensor_send_next=None,
                tensor_send_prev=input_tensor_grad,
                recv_prev=True,
                recv_next=False,
                tensor_shape=tensor_shape
            )
        return input_tensor


    def send_forward_recv_forward(self,
                                  output_tensor,
                                  recv_prev,
                                  tensor_shape,
                                  overlap_p2p_comm=False):
        """
        Send forward tensor, while receiving forward tensor.
        """
        input_tensor, _, reqs = self.node_p2p_comm(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=recv_prev,
            recv_next=False,
            tensor_shape=tensor_shape,
            wait_on_reqs=(not overlap_p2p_comm)
        )
        if overlap_p2p_comm:
            return input_tensor, reqs
        return input_tensor


    def send_backward_recv_backward(self,
                                    input_tensor_grad,
                                    recv_next,
                                    tensor_shape,
                                    overlap_p2p_comm=False):
        """
        Send backward tensor, while receiving backward tensor.
        """
        _, output_tensor_grad, reqs = self.node_p2p_comm(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            wait_on_reqs=(not overlap_p2p_comm)
        )
        if overlap_p2p_comm:
            return output_tensor_grad, reqs
        return output_tensor_grad


    def send_forward_backward_recv_forward_backward(self,
                                                    output_tensor,
                                                    input_tensor_grad,
                                                    recv_prev,
                                                    recv_next,
                                                    tensor_shape):
        """
        Send forward and backward tensor, while receiving forward and backward tensor.
        """
        input_tensor, output_tensor_grad, _ = self.node_p2p_comm(
            tensor_send_next=output_tensor,
            tensor_send_prev=input_tensor_grad,
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape=tensor_shape
        )
        return input_tensor, output_tensor_grad


    def node_p2p_comm(self,
                      tensor_send_next,
                      tensor_send_prev,
                      recv_prev,
                      recv_next,
                      tensor_shape,
                      wait_on_reqs=True):

        """
        Main function for communication between different nodes

        Args:
            tensor_send_next (Tensor or None): If is not None, send a tensor to next node.
            tensor_send_prev (Tensor or None): If is not None, send a tensor to previous node.
            recv_prev (Bool): If is not None, recv a tensor from previous node.
            recv_next (Bool): If is not None, recv a tensor from next node.
            tensor_shape (List): If 'recv_prev' or 'recv_next' is not None, define the shape of recv tensor.
            wait_on_reqs (Bool): If True, use synchronize method.
        """

        tensor_recv_prev = None
        tensor_recv_next = None

        recv_prev_shape = tensor_shape
        recv_next_shape = tensor_shape

        recv_dtype = self.config.parallel_config.recv_dtype
        if recv_prev:
            if tensor_shape is None:
                raise RuntimeError("Now receiving tensor from the previous stage, but the recv_shape is None.")
            tensor_recv_prev = ms.numpy.empty(recv_prev_shape, dtype=recv_dtype)
        if recv_next:
            if tensor_shape is None:
                raise RuntimeError("Now receiving tensor from the next stage, but the recv_shape is None.")
            tensor_recv_next = ms.numpy.empty(recv_next_shape, dtype=recv_dtype)

        # cast
        if tensor_send_prev is not None:
            tensor_send_prev = tensor_send_prev.astype(recv_dtype)

        if tensor_send_next is not None:
            tensor_send_next = tensor_send_next.astype(recv_dtype)

        # if tensor_send_prev or tensor_send_next is not None, send a tensor to specific stage
        # if tensor_recv_prev or tensor_recv_next is not None, recv a tensor from specific stage
        reqs, tensor_recv_prev, tensor_recv_next = self._isend_and_irecv(
            tensor_send_prev=tensor_send_prev,
            tensor_recv_prev=tensor_recv_prev,
            tensor_send_next=tensor_send_next,
            tensor_recv_next=tensor_recv_next,
            group=get_pipeline_model_parallel_group(),
        )

        # stream synchronize
        if wait_on_reqs and reqs:
            for req in reqs:
                req.wait()
            reqs = None

        # cast
        if tensor_recv_prev is not None:
            tensor_recv_prev = tensor_recv_prev.astype(self.config.compute_dtype)

        if tensor_recv_next is not None:
            tensor_recv_next = tensor_recv_next.astype(self.config.compute_dtype)

        return tensor_recv_prev, tensor_recv_next, reqs


    def _isend_and_irecv(self,
                         tensor_send_prev,
                         tensor_recv_prev,
                         tensor_send_next,
                         tensor_recv_next,
                         group):
        """ Use 'ISend' or 'IRecv' for p2p communication."""
        reqs = []
        rank = get_pipeline_model_parallel_rank()
        if rank % 2 == 0:
            if tensor_send_next is not None:
                send_next_req = ISend(0, get_pipeline_model_parallel_next_rank(), group=group)
                _ = send_next_req(tensor_send_next)

            if tensor_recv_prev is not None:
                recv_prev_req = IRecv(0, get_pipeline_model_parallel_prev_rank(),
                                      tensor_recv_prev.shape, tensor_recv_prev.dtype, group=group)
                recv_prev_stream, tensor_recv_prev = recv_prev_req()
                reqs.append(recv_prev_stream)

            if tensor_send_prev is not None:
                send_prev_req = ISend(0, get_pipeline_model_parallel_prev_rank(), group=group)
                _ = send_prev_req(tensor_send_prev)

            if tensor_recv_next is not None:
                recv_next_req = IRecv(0, get_pipeline_model_parallel_next_rank(),
                                      tensor_recv_next.shape, tensor_recv_next.dtype, group=group)
                recv_next_stream, tensor_recv_next = recv_next_req()
                reqs.append(recv_next_stream)
        else:
            if tensor_recv_prev is not None:
                recv_prev_req = IRecv(1, get_pipeline_model_parallel_prev_rank(),
                                      tensor_recv_prev.shape, tensor_recv_prev.dtype, group=group)
                recv_prev_stream, tensor_recv_prev = recv_prev_req()
                reqs.append(recv_prev_stream)

            if tensor_send_next is not None:
                send_next_req = ISend(1, get_pipeline_model_parallel_next_rank(), group=group)
                _ = send_next_req(tensor_send_next)

            if tensor_recv_next is not None:
                recv_next_req = IRecv(1, get_pipeline_model_parallel_next_rank(),
                                      tensor_recv_next.shape, tensor_recv_next.dtype, group=group)
                recv_next_stream, tensor_recv_next = recv_next_req()
                reqs.append(recv_next_stream)

            if tensor_send_prev is not None:
                send_prev_req = ISend(1, get_pipeline_model_parallel_prev_rank(), group=group)
                _ = send_prev_req(tensor_send_prev)
        return reqs, tensor_recv_prev, tensor_recv_next
