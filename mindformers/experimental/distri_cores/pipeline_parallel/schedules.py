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
""" pipeline process function """

from typing import Union
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore import Tensor
from mindspore.parallel._utils import _get_enable_parallel_optimizer
from .p2p_primitives import P2P_Primitive
from ..create_comm import get_pp_rank, get_pp_world_size, get_cp_world_size, \
                          get_tp_world_size, is_pipeline_last_stage, is_pipeline_first_stage, \
                          get_dp_group, get_tp_group, is_rank_in_embedding_group, \
                          get_embedding_group, get_dp_world_size


_grad_scale = C.MultitypeFuncGraph("grad_scale")
_reduce_grads = C.MultitypeFuncGraph("reduce_grads")
reciprocal = P.Reciprocal()
hyper_map = C.HyperMap()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale_pipeline(scale, grad):
    """ Apply grad scale calculation in pipeline parallel """
    new_grad = grad * F.cast(reciprocal(scale), F.dtype(grad))
    return new_grad


@_reduce_grads.register("Bool", "Bool", "Tensor")
def reduce_grads_func(mean, optimizer_parallel, grad):
    """ Apply allreduce or reducescatter operation in dp group """
    if optimizer_parallel:
        grad = P.ReduceScatter(op='sum', group=get_dp_group())(grad)
    else:
        grad = P.AllReduce(op='sum', group=get_dp_group())(grad)

    if mean:
        grad = grad / get_dp_world_size()
    return grad


def get_accumulate_grads_func():
    """ Get accumulate grad fn """
    def accu_fn(accumulate_grad, grad):
        accumulate_grad = accumulate_grad + grad
        return accumulate_grad

    def accumulate_grads_fn(accumulate_grads, grads):
        success = hyper_map(accu_fn, accumulate_grads, grads)
        return success
    return accumulate_grads_fn


def run_forward(*input_data,
                model,
                accumulate_loss,
                recv_data=None):
    """
    Run forward step.
    In first stage, the input_ids will be used from data iterator.
    Otherwise, it will be changes to passed-in recv_data.

    The forward process of the model should include the calculation of loss.
    In last stage, the output_tensor is defaulted to the loss value, which will be accumulated

    Outputs:
        Tuple of 3 Tensor, the output_tensor, accumulate_loss and input_data.
        - **output_tensor** (Tensor) -  forward output.
        - **accumulate_loss** (Tensor) -  model final loss value.
        - **input_data** (Tensor) -  The micro input data after correction by recv_data.
    """

    if isinstance(recv_data, (list, tuple)):
        recv_data = recv_data[0]

    if is_pipeline_last_stage():
        # return loss
        output_tensor = model(*input_data, recv_data=recv_data)
        accumulate_loss += output_tensor
    else:
        # return next stage input
        output_tensor = model(*input_data, recv_data=recv_data)

    input_data += (recv_data,)
    return output_tensor, accumulate_loss, list(input_data)


def run_backward(*input_tensor,
                 recv_grads,
                 model,
                 weight,
                 scale_sense,
                 micro_size,
                 accumulate_grads,
                 reduction='mean'):
    """
    Run backward step.
    In last stage, recv_grads is None, and it will be init as all ones tensor for grad accumulation.
    In first stage, The return value of dout is not used.

    Outputs:
        Tuple of 2 Tensor, the dout and accumulate_grads.
        - **dout** (Tensor) -  input_ids grads result for bprop.
        - **accumulate_grads** (Tuple) -  weight grads for optimize.
    """

    if isinstance(recv_grads, (list, tuple)):
        recv_grads = recv_grads[0]

    # init dout if in last stage
    if is_pipeline_last_stage():
        if reduction == 'mean':
            recv_grads = 1.0 * F.cast(scale_sense / micro_size, mstype.float32)
        elif reduction == 'sum':
            recv_grads = 1.0 * F.cast(scale_sense, mstype.float32)
        else:
            raise TypeError("For gradient and loss reduction methods, "
                            "currently only 'sum' and 'mean' modes are supported")

    # get grad function
    grad_fn = C.GradOperation(get_all=True, get_by_list=True, sens_param=True)(model, weight)

    # get grad function input
    # 'recv_data' needs to be passed as keyword argument into the function
    recv_data = input_tensor[-1]
    input_tensor = tuple(input_tensor[:-1])

    # calculate grads
    grads_output = grad_fn(*input_tensor, recv_data=recv_data, sens=recv_grads)
    dout, weight_grad = grads_output[0], grads_output[1]

    # bprob
    if is_pipeline_first_stage():
        # the first stage does not require backpropagation
        dout = None
    else:
        # bprop for recv_data
        dout = dout[-1]

    # accumulate grads between multi-micro input
    if accumulate_grads is None:
        accumulate_grads = weight_grad
    else:
        accumulate_grads_func = get_accumulate_grads_func()
        accumulate_grads = accumulate_grads_func(accumulate_grads, weight_grad)

    return dout, accumulate_grads

# pylint: disable=C0103
def pipelining_1F1B_without_interleaved(
        model,
        optimizer,
        scale_sense,
        micro_batch_num,
        micro_batch_size,
        seq_length,
        hidden_size,
        config,
        *input_data_tuple,
        **input_data_dict
):
    """ Pipeline 1F1B wrapper for split model run forward and backward """
    # init p2p class
    p2p_primitive = P2P_Primitive(config=config)

    # get config value
    reduction = config.reduction
    use_sequence_parallel = config.use_sequence_parallel

    # get parameters
    weights = optimizer.parameters

    # correct tensor shape if use seq parallel or context parallel
    recv_tensor_shapes = correct_p2p_shape(seq_length, hidden_size, micro_batch_size, use_sequence_parallel)
    send_tensor_shapes = correct_p2p_shape(seq_length, hidden_size, micro_batch_size, use_sequence_parallel)

    # save each forward process input data for running backward
    input_tensors = []
    accumulate_loss = Tensor(0.0, mstype.float32)

    # get warm up stage steps
    warm_up_steps = min(get_pp_world_size() - get_pp_rank() - 1, micro_batch_num)
    input_tensor = None

    # warm up process
    for i in range(warm_up_steps):
        # if is not first stage, get forward input tensor for model
        input_tensor = recv_forward(recv_tensor_shapes, p2p_primitive)

        # get other data from dataset for model input
        slice_index = i
        micro_input_data = get_micro_input(slice_index,
                                           micro_batch_size,
                                           model,
                                           *input_data_tuple,
                                           **input_data_dict)

        # run forward
        output_tensor, accumulate_loss, micro_input_data = run_forward(*micro_input_data,
                                                                       model=model,
                                                                       accumulate_loss=accumulate_loss,
                                                                       recv_data=input_tensor)

        # save micro input data for backward
        input_tensors.append(micro_input_data)

        # send forward result to next stage
        send_forward(output_tensor, send_tensor_shapes, p2p_primitive)

    # prepare input data for 1F1B
    steady_steps = micro_batch_num - warm_up_steps
    if steady_steps > 0:
        slice_index = 0 if is_pipeline_last_stage() else slice_index + 1
        micro_input_data = get_micro_input(slice_index,
                                           micro_batch_size,
                                           model,
                                           *input_data_tuple,
                                           **input_data_dict)
        input_tensor = recv_forward(recv_tensor_shapes, p2p_primitive)

    # 1F1B process
    accumulate_grads = None
    for i in range(steady_steps):
        output_tensor, accumulate_loss, micro_input_data = run_forward(*micro_input_data,
                                                                       model=model,
                                                                       accumulate_loss=accumulate_loss,
                                                                       recv_data=input_tensor)

        recv_grads = send_forward_recv_backward(output_tensor, send_tensor_shapes, p2p_primitive)
        input_tensors.append(micro_input_data)

        # bprop func need forward's input data
        input_tensor = input_tensors.pop(0)

        # run backward
        dout, accumulate_grads = run_backward(*input_tensor,
                                              recv_grads=recv_grads,
                                              model=model,
                                              weight=weights,
                                              scale_sense=scale_sense,
                                              micro_size=micro_batch_num,
                                              accumulate_grads=accumulate_grads,
                                              reduction=reduction)

        if i == steady_steps - 1:
            input_tensor = None
            send_backward(dout, recv_tensor_shapes, p2p_primitive)
        else:
            slice_index += 1
            micro_input_data = get_micro_input(slice_index,
                                               micro_batch_size,
                                               model,
                                               *input_data_tuple,
                                               **input_data_dict)
            input_tensor = send_backward_recv_forward(dout, recv_tensor_shapes, p2p_primitive)

    # cooldown process
    cooldown_steps = warm_up_steps
    for i in range(cooldown_steps):
        input_tensor = input_tensors.pop(0)
        recv_grads = recv_backward(send_tensor_shapes, p2p_primitive)
        dout, accumulate_grads = run_backward(*input_tensor,
                                              recv_grads=recv_grads,
                                              model=model,
                                              weight=weights,
                                              scale_sense=scale_sense,
                                              micro_size=micro_batch_num,
                                              accumulate_grads=accumulate_grads,
                                              reduction=reduction)
        send_backward(dout, recv_tensor_shapes, p2p_primitive)

    # 1.AllReduce embedding (if shared embedding weight in first stage and last stage)
    # 2.AllReduce model grads (if use optimizer parallel, it will be changed to ReduceScatter operation)
    # 3.AllReduce layernorm (if use sequence parallel)
    weight_name = [weight.name for weight in weights]
    accumulate_grads = all_reduce_model_grads(list(accumulate_grads),
                                              weight_name,
                                              model,
                                              use_sequence_parallel)
    # unscale grad
    accumulate_grads = hyper_map(F.partial(_grad_scale, scale_sense), accumulate_grads)

    return accumulate_loss, accumulate_grads


def correct_p2p_shape(seq_length, hidden_size, micro_batch_size, use_sequence_parallel=False):
    """
    Correct right tensor shape under context parallel or sequence parallel.
    """
    seq_length = seq_length // get_cp_world_size()
    if use_sequence_parallel:
        seq_length = seq_length // get_tp_world_size()
    return ((micro_batch_size, seq_length, hidden_size),)


def recv_forward(tensor_shapes: Union[tuple, list],
                 p2p: P2P_Primitive):
    """ Recv forward output tensor from prev rank in pipeline. """
    tensor_shape = tensor_shapes
    if isinstance(tensor_shapes[0], (tuple, list)):
        tensor_shape = tensor_shape[0]

    if tensor_shape is None:
        recv_tensor = None
    else:
        recv_tensor = p2p.recv_forward(tensor_shape)
    recv_tensor = (recv_tensor,) if recv_tensor is not None else None
    return recv_tensor


def send_forward(input_tensors: Union[Tensor, list],
                 tensor_shapes: Union[tuple, list],
                 p2p: P2P_Primitive):
    """ Send forward output tensor to next rank in pipeline. """
    tensor_shape = tensor_shapes
    if isinstance(tensor_shapes[0], (tuple, list)):
        tensor_shape = tensor_shape[0]

    if tensor_shape is not None:
        p2p.send_forward(input_tensors)


def send_backward(input_grads: Union[Tensor, list],
                  tensor_shapes: Union[tuple, list],
                  p2p: P2P_Primitive):
    """ Send backward output tensor to next rank in pipeline. """
    tensor_shape = tensor_shapes
    if isinstance(tensor_shapes[0], (tuple, list)):
        tensor_shape = tensor_shape[0]

    if tensor_shape is not None:
        p2p.send_backward(input_grads)


def recv_backward(tensor_shapes: Union[tuple, list],
                  p2p: P2P_Primitive):
    """ recv backward dout tensor from next rank in pipeline. """
    tensor_shape = tensor_shapes
    if isinstance(tensor_shapes[0], (tuple, list)):
        tensor_shape = tensor_shape[0]

    recv_grads = ()
    if tensor_shape is None:
        recv_grads += (None,)
    else:
        recv_grads += (p2p.recv_backward(tensor_shape),)
    return recv_grads


def send_forward_recv_backward(input_tensors: Union[Tensor, list],
                               tensor_shapes: Union[tuple, list],
                               p2p: P2P_Primitive):
    """ Send forward output and recv backward dout from next rank in pipeline."""
    tensor_shape = tensor_shapes
    input_tensor = input_tensors

    if isinstance(tensor_shape[0], (tuple, list)):
        tensor_shape = tensor_shape[0]

    if isinstance(input_tensor, (tuple, list)):
        input_tensor = input_tensor[0]

    if tensor_shape is None:
        recv_grads = None
    else:
        recv_grads = p2p.send_forward_recv_backward(input_tensor, tensor_shape)

    recv_grads = (recv_grads,) if recv_grads is not None else None
    return recv_grads


def send_backward_recv_forward(input_grads: Union[Tensor, list],
                               tensor_shapes: Union[tuple, list],
                               p2p: P2P_Primitive):
    """ Send backward grad and recv forward output from prev rank in pipeline."""
    tensor_shape = tensor_shapes
    input_grad = input_grads
    if isinstance(tensor_shape[0], (tuple, list)):
        tensor_shape = tensor_shape[0]

    if isinstance(input_grad, (tuple, list)):
        input_grad = input_grad[0]

    input_tensors = ()
    if tensor_shape is None:
        input_tensors += (None,)
    else:
        input_tensor = p2p.send_backward_recv_forward(input_grad, tensor_shape)
        input_tensors += (input_tensor,)
    return input_tensors


def get_micro_input(i, micro_batch_size, model, *input_data_tuple, **input_data_dict):
    """ Get current micro batch from data inputS """
    # get model construct func input attr
    model_args_index = model.model_args_index
    micro_inputs = model.model_args_default_value

    # get current slice start and end idx
    current_micro_begin = micro_batch_size * i
    current_micro_end = micro_batch_size * (i + 1)

    if input_data_tuple:
        for idx, input_tensor in enumerate(input_data_tuple):
            if isinstance(input_tensor, Tensor):
                input_tensor = input_tensor[current_micro_begin: current_micro_end]
            micro_inputs[idx] = input_tensor

    if input_data_dict:
        for key, input_tensor in input_data_dict.items():
            if isinstance(input_tensor, Tensor):
                input_tensor = input_tensor[current_micro_begin: current_micro_end]
            idx = model_args_index.index(key)
            micro_inputs[idx] = input_tensor

    return tuple(micro_inputs)


def all_reduce_model_grads(grads, weight_name, model, reduction="mean", seq_parallel=False):
    """
    step1: Allreduce embedding grads for shared weight
    step2: Allreduce/ReduceScatter all model grads in dp group.
    step3: Allreduce layernorm grads for sequence parallelism,
    """
    # All-reduce embedding grads (for share embedding weight).
    grads = _all_reduce_share_embedding(grads, weight_name, model)

    # All-reduce / reduce-scatter dp group grads.
    grads = _reduce_model_grads(grads, reduction)

    # All-reduce layer-norm grads (for sequence parallelism).
    grads = _all_reduce_norm(grads, weight_name, seq_parallel)
    return tuple(grads)


def _reduce_model_grads(grads, reduction):
    """ Reduce grads in dp group """
    mean = False
    optimizer_parallel = False
    if _get_enable_parallel_optimizer():
        optimizer_parallel = True
    if reduction == "mean":
        mean = True

    grads = hyper_map(F.partial(_reduce_grads, mean, optimizer_parallel), grads)
    return grads


def _all_reduce_norm(grads, weight_name, seq_parallel):
    """ Reduce layernorm grads in tp group """
    if seq_parallel and get_tp_world_size() > 1:
        for i, name in enumerate(weight_name):
            weight = grads[i]
            if 'norm' in name:
                grads[i] = P.AllReduce(op='sum', group=get_tp_group())(weight)
    return grads


def _all_reduce_share_embedding(grads, weight_name, model):
    """ Reduce share embedding grads in embedding comm group """
    if is_rank_in_embedding_group(ignore_virtual=True):
        if model.share_embedding_weight and get_pp_world_size() > 1:
            idx = None
            shared_weight_name = model.shared_weight_name
            for i, name in enumerate(weight_name):
                if name in shared_weight_name:
                    idx = i
                    break

            if idx is None:
                raise AttributeError(f"model is not contain embedding weight or head weight, "
                                     f"but share_embedding_weight is set to True")
            weight = grads[idx]
            grads[idx] = P.AllReduce(op='sum', group=get_embedding_group())(weight)
    return grads
