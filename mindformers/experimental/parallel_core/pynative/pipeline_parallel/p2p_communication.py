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

import mindspore as ms
from mindspore import nn
from mindspore.ops.auto_generate.gen_ops_prim import inner_comm_irecv_op, inner_comm_isend_op
from mindspore.ops import operations as P
from mindspore.communication.comm_func import broadcast

from mindformers.experimental.parallel_core.pynative.parallel_state import get_pipeline_model_parallel_group, \
    get_pipeline_model_parallel_rank, is_pipeline_last_stage, is_pipeline_first_stage, get_data_parallel_world_size, \
    get_data_parallel_group, get_hetero_pp_fwd_group, get_hetero_pp_bwd_group, get_hetero_pp_fwd_world_size, \
    get_hetero_pp_bwd_world_size, get_data_parallel_rank, get_tensor_model_parallel_rank, \
    get_tensor_model_parallel_world_size, get_tensor_model_parallel_group, get_tensor_model_parallel_first_rank, \
    get_pipeline_model_parallel_world_size

class ISend(nn.Cell):
    """ Send a tensor asynchronously """
    def __init__(self, src_tag, dst_rank, group):
        super(ISend, self).__init__()
        self.src_tag = src_tag
        self.dst_rank = dst_rank
        self.group = group

    def construct(self, send_data):
        """ ISend forward """
        _, handle = inner_comm_isend_op(send_data, self.dst_rank, self.group, self.src_tag)
        return handle


class IRecv(nn.Cell):
    """ receive a tensor asynchronously """
    def __init__(self, src_tag, src_rank, shape, dtype, group):
        super(IRecv, self).__init__()
        self.src_tag = src_tag
        self.src_rank = src_rank
        self.shape = shape
        self.dtype = dtype
        self.groud = group

    def construct(self):
        """ IRecv forward """
        recv_tensor, handle = inner_comm_irecv_op(self.src_tag, self.src_rank, self.shape, self.groud, self.dtype)
        return handle, recv_tensor


# pylint: disable=C0103
class P2PPrimitive():
    """ A class that includes P2P communication methods """
    def __init__(self, config):
        self.config = config
        self.heterogeneous_pipeline = self.config.parallel_config.heterogeneous_pipeline
        self.pipeline_stage_device = self.config.parallel_config.pipeline_stage_device
        self.recv_dtype = self.config.parallel_config.recv_dtype

    def send_forward(self,
                     output_tensor):
        """
        Except for the last stage, send forward tensor.
        """
        if not is_pipeline_last_stage(heterogeneous_pipeline=self.heterogeneous_pipeline,
                                      pipeline_stage_device=self.pipeline_stage_device):
            if self.heterogeneous_pipeline and get_data_parallel_world_size() > 1:
                allgather = P.AllGather(get_data_parallel_group())
                output_tensor = allgather(output_tensor)
            if not self.heterogeneous_pipeline or \
                (self.heterogeneous_pipeline and get_tensor_model_parallel_rank() == 0):
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
        if is_pipeline_first_stage(heterogeneous_pipeline=self.heterogeneous_pipeline,
                                   pipeline_stage_device=self.pipeline_stage_device):
            input_tensor = None
        else:
            input_tensor = ms.numpy.empty(tensor_shape, dtype=self.recv_dtype)
            if not self.heterogeneous_pipeline or \
                (self.heterogeneous_pipeline and get_tensor_model_parallel_rank() == 0):
                input_tensor, _, _ = self.node_p2p_comm(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=tensor_shape
                )
            if self.heterogeneous_pipeline and get_tensor_model_parallel_world_size() > 1:
                input_tensor = broadcast(input_tensor, get_tensor_model_parallel_first_rank(), \
                                         get_tensor_model_parallel_group())
        return input_tensor

    def send_backward(self,
                      input_tensor_grad):
        """
        Except for the first stage, send backward tensor.
        """
        if not is_pipeline_first_stage(heterogeneous_pipeline=self.heterogeneous_pipeline,
                                       pipeline_stage_device=self.pipeline_stage_device):
            if self.heterogeneous_pipeline and get_data_parallel_world_size() > 1:
                allgather = P.AllGather(get_data_parallel_group())
                input_tensor_grad = allgather(input_tensor_grad)

            if not self.heterogeneous_pipeline or \
                (self.heterogeneous_pipeline and get_tensor_model_parallel_rank() == 0):
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
        if is_pipeline_last_stage(heterogeneous_pipeline=self.pipeline_stage_device,
                                  pipeline_stage_device=self.pipeline_stage_device):
            output_tensor_grad = None
        else:
            output_tensor_grad = ms.numpy.empty(tensor_shape, dtype=self.recv_dtype)
            if not self.heterogeneous_pipeline or \
                (self.heterogeneous_pipeline and get_tensor_model_parallel_rank() == 0):
                _, output_tensor_grad, _ = self.node_p2p_comm(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=True,
                    tensor_shape=tensor_shape
                )
            if self.heterogeneous_pipeline and get_tensor_model_parallel_world_size() > 1:
                output_tensor_grad = broadcast(output_tensor_grad, get_tensor_model_parallel_first_rank(), \
                                               get_tensor_model_parallel_group())
        return output_tensor_grad

    def send_forward_recv_backward(self,
                                   output_tensor,
                                   tensor_shape):
        """
        Except for the last stage, send forward tensor, while receiving backward tensor.
        """
        if is_pipeline_last_stage(heterogeneous_pipeline=self.heterogeneous_pipeline,
                                  pipeline_stage_device=self.pipeline_stage_device):
            output_tensor_grad = None
        else:
            output_tensor_grad = ms.numpy.empty(tensor_shape, dtype=self.recv_dtype)
            if self.heterogeneous_pipeline and get_data_parallel_world_size() > 1:
                allgather = P.AllGather(get_data_parallel_group())
                output_tensor = allgather(output_tensor)

            if not self.heterogeneous_pipeline or \
                (self.heterogeneous_pipeline and get_tensor_model_parallel_rank() == 0):
                _, output_tensor_grad, _ = self.node_p2p_comm(
                    tensor_send_next=output_tensor,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=True,
                    tensor_shape=tensor_shape
                )
            if self.heterogeneous_pipeline and get_tensor_model_parallel_world_size() > 1:
                output_tensor_grad = broadcast(output_tensor_grad, get_tensor_model_parallel_first_rank(), \
                                               get_tensor_model_parallel_group())
        return output_tensor_grad

    def send_backward_recv_forward(self,
                                   input_tensor_grad,
                                   tensor_shape):
        """
        Except for the first stage, send backward tensor, while receiving forward tensor.
        """
        if is_pipeline_first_stage(heterogeneous_pipeline=self.heterogeneous_pipeline,
                                   pipeline_stage_device=self.pipeline_stage_device):
            input_tensor = None
        else:
            input_tensor = ms.numpy.empty(tensor_shape, dtype=self.recv_dtype)
            if self.heterogeneous_pipeline and get_data_parallel_world_size() > 1:
                allgather = P.AllGather(get_data_parallel_group())
                input_tensor_grad = allgather(input_tensor_grad)

            if not self.heterogeneous_pipeline or \
                (self.heterogeneous_pipeline and get_tensor_model_parallel_rank() == 0):
                input_tensor, _, _ = self.node_p2p_comm(
                    tensor_send_next=None,
                    tensor_send_prev=input_tensor_grad,
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=tensor_shape
                )
            if self.heterogeneous_pipeline and get_tensor_model_parallel_world_size() > 1:
                input_tensor = broadcast(input_tensor, get_tensor_model_parallel_first_rank(), \
                                         get_tensor_model_parallel_group())
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

        tensor_info_recv_prev = None
        tensor_info_recv_next = None

        recv_prev_shape = tensor_shape
        recv_next_shape = tensor_shape

        recv_dtype = self.config.parallel_config.recv_dtype
        if recv_prev:
            if tensor_shape is None:
                raise RuntimeError("Now receiving tensor from the previous stage, but the recv_shape is None.")
            tensor_info_recv_prev = (recv_prev_shape, recv_dtype)
        if recv_next:
            if tensor_shape is None:
                raise RuntimeError("Now receiving tensor from the next stage, but the recv_shape is None.")
            tensor_info_recv_next = (recv_next_shape, recv_dtype)

        # if tensor_send_prev or tensor_send_next is not None, send a tensor to specific stage
        # if tensor_info_recv_prev or tensor_info_recv_next is not None, recv a tensor from specific stage
        if self.heterogeneous_pipeline:
            reqs, tensor_recv_prev, tensor_recv_next = self._hetero_isend_and_irecv(
                tensor_send_prev=tensor_send_prev,
                tensor_recv_prev=tensor_info_recv_prev,
                tensor_send_next=tensor_send_next,
                tensor_recv_next=tensor_info_recv_next
            )
        else:
            reqs, tensor_recv_prev, tensor_recv_next = self._isend_and_irecv(
                tensor_send_prev=tensor_send_prev,
                tensor_info_recv_prev=tensor_info_recv_prev,
                tensor_send_next=tensor_send_next,
                tensor_info_recv_next=tensor_info_recv_next,
                group=get_pipeline_model_parallel_group(),
            )

        # stream synchronize
        if wait_on_reqs and reqs:
            for req in reqs:
                req.wait()
            reqs = None

        return tensor_recv_prev, tensor_recv_next, reqs

    @staticmethod
    def _isend_and_irecv(tensor_send_prev,
                         tensor_info_recv_prev,
                         tensor_send_next,
                         tensor_info_recv_next,
                         group):
        """ Use 'ISend' or 'IRecv' for p2p communication."""
        reqs = []
        tensor_recv_prev = None
        tensor_recv_next = None
        rank_in_pipeline = get_pipeline_model_parallel_rank()
        world_size = get_pipeline_model_parallel_world_size()
        if rank_in_pipeline % 2 == 0:
            if tensor_send_next is not None:
                send_next_req = ISend(0, (rank_in_pipeline + 1) % world_size, group=group)
                send_next_handle = send_next_req(tensor_send_next)
                reqs.append(send_next_handle)

            if tensor_info_recv_prev is not None:
                recv_prev_req = IRecv(0, (rank_in_pipeline - 1) % world_size,
                                      tensor_info_recv_prev[0], tensor_info_recv_prev[1], group=group)
                recv_prev_handle, tensor_recv_prev = recv_prev_req()
                reqs.append(recv_prev_handle)

            if tensor_send_prev is not None:
                send_prev_req = ISend(0, (rank_in_pipeline - 1) % world_size, group=group)
                send_prev_handle = send_prev_req(tensor_send_prev)
                reqs.append(send_prev_handle)

            if tensor_info_recv_next is not None:
                recv_next_req = IRecv(0, (rank_in_pipeline + 1) % world_size,
                                      tensor_info_recv_next[0], tensor_info_recv_next[1], group=group)
                recv_next_handle, tensor_recv_next = recv_next_req()
                reqs.append(recv_next_handle)
        else:
            if tensor_info_recv_prev is not None:
                recv_prev_req = IRecv(1, (rank_in_pipeline - 1) % world_size,
                                      tensor_info_recv_prev[0], tensor_info_recv_prev[1], group=group)
                recv_prev_handle, tensor_recv_prev = recv_prev_req()
                reqs.append(recv_prev_handle)

            if tensor_send_next is not None:
                send_next_req = ISend(1, (rank_in_pipeline + 1) % world_size, group=group)
                send_next_handle = send_next_req(tensor_send_next)
                reqs.append(send_next_handle)

            if tensor_info_recv_next is not None:
                recv_next_req = IRecv(1, (rank_in_pipeline + 1) % world_size,
                                      tensor_info_recv_next[0], tensor_info_recv_next[1], group=group)
                recv_next_handle, tensor_recv_next = recv_next_req()
                reqs.append(recv_next_handle)

            if tensor_send_prev is not None:
                send_prev_req = ISend(1, (rank_in_pipeline - 1) % world_size, group=group)
                send_prev_handle = send_prev_req(tensor_send_prev)
                reqs.append(send_prev_handle)
        return reqs, tensor_recv_prev, tensor_recv_next

    def _hetero_isend_and_irecv(self,
                                tensor_send_prev,
                                tensor_recv_prev,
                                tensor_send_next,
                                tensor_recv_next):
        """Use 'ISend' and 'IRecv' for heterogeneous pipeline p2p communication."""
        reqs = []
        rank_in_pipeline = get_pipeline_model_parallel_rank(heterogeneous_pipeline=self.heterogeneous_pipeline,
                                                            pipeline_stage_device=self.pipeline_stage_device)
        if rank_in_pipeline % 2 == 0:
            if tensor_send_next is not None and get_data_parallel_rank() == 0:
                fwd_group_size = get_hetero_pp_fwd_world_size(rank_in_pipeline, cur_stage_send=True)
                group = get_hetero_pp_fwd_group(rank_in_pipeline, cur_stage_send=True)
                for i in range(1, fwd_group_size):
                    send_next_req = ISend(0, i % fwd_group_size, group=group)
                    micro_batch_size = int(tensor_send_next.shape[0] / (fwd_group_size - 1))
                    send_next_stream = send_next_req(tensor_send_next[(i - 1) * micro_batch_size: i * micro_batch_size])
                    reqs.append(send_next_stream)

            if tensor_recv_prev is not None:
                group = get_hetero_pp_fwd_group(rank_in_pipeline, cur_stage_send=False)
                recv_prev_req = IRecv(0, 0,
                                      tensor_recv_prev[0], tensor_recv_prev[1], group=group)
                recv_prev_stream, tensor_recv_prev = recv_prev_req()
                reqs.append(recv_prev_stream)

            if tensor_send_prev is not None and get_data_parallel_rank() == 0:
                bwd_group_size = get_hetero_pp_bwd_world_size(rank_in_pipeline, cur_stage_send=True)
                group = get_hetero_pp_bwd_group(rank_in_pipeline, cur_stage_send=True)
                for i in range(1, bwd_group_size):
                    send_prev_req = ISend(0, i % bwd_group_size, group=group)
                    micro_batch_size = int(tensor_send_prev.shape[0] / (bwd_group_size - 1))
                    send_prev_stream = send_prev_req(tensor_send_prev[(i - 1) * micro_batch_size: i * micro_batch_size])
                    reqs.append(send_prev_stream)

            if tensor_recv_next is not None:
                group = get_hetero_pp_bwd_group(rank_in_pipeline, cur_stage_send=False)
                recv_next_req = IRecv(0, 0,
                                      tensor_recv_next[0], tensor_recv_next[1], group=group)
                recv_next_stream, tensor_recv_next = recv_next_req()
                reqs.append(recv_next_stream)
        else:
            if tensor_recv_prev is not None:
                group = get_hetero_pp_fwd_group(rank_in_pipeline, cur_stage_send=False)
                recv_prev_req = IRecv(1, 0,
                                      tensor_recv_prev[0], tensor_recv_prev[1], group=group)
                recv_prev_stream, tensor_recv_prev = recv_prev_req()
                reqs.append(recv_prev_stream)

            if tensor_send_next is not None and get_data_parallel_rank() == 0:
                fwd_group_size = get_hetero_pp_fwd_world_size(rank_in_pipeline, cur_stage_send=True)
                group = get_hetero_pp_fwd_group(rank_in_pipeline, cur_stage_send=True)
                for i in range(1, fwd_group_size):
                    send_next_req = ISend(1, i % fwd_group_size, group=group)
                    micro_batch_size = int(tensor_send_next.shape[0] / (fwd_group_size - 1))
                    send_next_stream = send_next_req(tensor_send_next[(i - 1) * micro_batch_size: i * micro_batch_size])
                    reqs.append(send_next_stream)

            if tensor_recv_next is not None:
                group = get_hetero_pp_bwd_group(rank_in_pipeline, cur_stage_send=False)
                recv_next_req = IRecv(1, 0,
                                      tensor_recv_next[0], tensor_recv_next[1], group=group)
                recv_next_stream, tensor_recv_next = recv_next_req()
                reqs.append(recv_next_stream)

            if tensor_send_prev is not None and get_data_parallel_rank() == 0:
                bwd_group_size = get_hetero_pp_bwd_world_size(rank_in_pipeline, cur_stage_send=True)
                group = get_hetero_pp_bwd_group(rank_in_pipeline, cur_stage_send=True)
                for i in range(1, bwd_group_size):
                    send_prev_req = ISend(1, i % bwd_group_size, group=group)
                    micro_batch_size = int(tensor_send_prev.shape[0] / (bwd_group_size - 1))
                    send_prev_stream = send_prev_req(tensor_send_prev[(i - 1) * micro_batch_size: i * micro_batch_size])
                    reqs.append(send_prev_stream)

        return reqs, tensor_recv_prev, tensor_recv_next
