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
from mindspore import Tensor, mint, nn

from mindformers.experimental.parallel_core.pynative.parallel_state import get_pp_rank, get_pp_world_size, \
    get_cp_world_size, get_tp_world_size, is_pipeline_last_stage, is_pipeline_first_stage, \
    is_rank_in_embedding_group, get_embedding_group, set_vpp_rank

from .p2p_communication import P2P_Primitive
from .pipeline_cell import PipelineCell


def accumulate_grads_func(accumulate_grads, current_micro_grads):
    """ Accumulate grad """
    for i, _ in enumerate(accumulate_grads):
        accumulate_grads[i] += current_micro_grads[i]
    return tuple(accumulate_grads)


# pylint: disable=R1710
def get_set_hidden_states_parameter(model):
    """ Get the parameter which set by set_input_tensor() """
    param = None
    weight_untrainable = model.untrainable_params()
    for cur_param in weight_untrainable:
        if "set_hidden_states" in cur_param.name:
            param = cur_param
            param.requires_grad = True
            return param
    if param is None:
        raise RuntimeError("Parameter 'set_hidden_states' is not found.")


def rename_hidden_states_parameter(model, model_chunk_id=None):
    weight_untrainable = model.untrainable_params()
    for param in weight_untrainable:
        if "set_hidden_states" in param.name:
            param.name = param.name + f"_{model_chunk_id}_chunk"


# pylint: disable=W0613
def run_forward(*input_data,
                model,
                accumulate_loss,
                logits,
                recv_data=None,
                auto_pp_spliting=False,
                calculate_per_token_loss=False,
                tokens_nums_list=None,
                **kwargs):
    """
    Run forward step.
    In first stage, the input_ids will be used from data iterator.
    Otherwise, it will be changes to 'recv_data'.

    The forward process of the model should include the calculation of loss.
    In last stage, the output_tensor is defaulted to the loss value, which will be accumulated

    Outputs:
        Tuple of 3 Tensor, the output_tensor, accumulate_loss and input_data.
        - **output_tensor** (Tensor) -  forward output.
        - **accumulate_loss** (Tensor) -  model final loss value.
        - **input_data** (Tensor) -  The micro input data after correction by recv_data.
    """
    micro_tokens_nums = None
    micro_logits = None

    if isinstance(recv_data, (list, tuple)):
        recv_data = recv_data[0]

    if not auto_pp_spliting and not is_pipeline_first_stage() and recv_data is not None:
        model.set_input_tensor(recv_data)

    if is_pipeline_last_stage():
        # return loss
        if auto_pp_spliting:
            output_tensors = model(*input_data, recv_data=recv_data)
        else:
            output_tensors = model(*input_data, recv_data=None)

        # check output
        if isinstance(output_tensors, tuple):
            # pylint: disable=C1801
            if len(output_tensors) == 2 and output_tensors[-1] is not None:
                if len(output_tensors[-1].shape) == 0:
                    output_tensor, micro_tokens_nums = output_tensors
                elif len(output_tensors[-1].shape) == 2 or len(output_tensors[-1].shape) == 3:
                    output_tensor, micro_logits = output_tensors
            elif len(output_tensors) == 3:
                output_tensor, micro_logits, micro_tokens_nums = output_tensors
        else:
            output_tensor = output_tensors

        if calculate_per_token_loss and micro_tokens_nums is None:
            raise RuntimeError("When 'calculate_per_token_loss=True', the output of model must be: "
                               "(loss, logits, tokens_nums). But now the 'tokens_nums' is missing.")
        # micro acc process
        accumulate_loss.append(output_tensor)
        if micro_logits is not None:
            logits.append(micro_logits)
        if micro_tokens_nums is not None:
            tokens_nums_list.append(micro_tokens_nums)
    else:
        # return next stage input
        if auto_pp_spliting:
            output_tensor = model(*input_data, recv_data=recv_data)
        else:
            output_tensor = model(*input_data, recv_data=None)

    input_data += (recv_data,)
    return output_tensor, accumulate_loss, list(input_data)


# pylint: disable=W0613
def run_backward(*input_tensor,
                 recv_grads,
                 model,
                 weight,
                 scale_sense,
                 accumulate_loss,
                 accumulate_grads,
                 micro_size=None,
                 logits=None,
                 auto_pp_spliting=False,
                 calculate_per_token_loss=False,
                 total_tokens_nums=None,
                 tokens_nums_list=None,
                 **kwargs):
    """
    Run backward step.
    In last stage, recv_grads is None, and it will be init as all ones tensor for grad accumulation.
    In first stage, The return value of dout is None.

    Outputs:
        Tuple of 2 Tensor, the dout and accumulate_grads.
        - **dout** (Tensor) -  input_ids grads result for bprop.
        - **accumulate_grads** (Tuple) -  weight grads for optimize.
    """

    if isinstance(recv_grads, (list, tuple)):
        recv_grads = recv_grads[0]

    # init dout if in last stage
    if is_pipeline_last_stage():
        # scaling grad base on micro_size or tokens nums
        if calculate_per_token_loss:
            micro_tokens_num = tokens_nums_list.pop(0)
            factor = micro_tokens_num / total_tokens_nums
            tokens_nums_list.append(micro_tokens_num)
        else:
            factor = 1.0 / micro_size

        # init grad
        recv_grads = mint.ones_like(accumulate_loss[0]) * F.cast(scale_sense * factor, accumulate_loss[0].dtype)

        # stop 'logits' gradient bprop
        if logits:
            logits_grads = mint.zeros_like(logits[0])
            recv_grads = (recv_grads, logits_grads)

        # stop 'tokens_num' gradient bprop
        if tokens_nums_list:
            tokens_nums_grads = mint.zeros_like(tokens_nums_list[0])
            recv_grads = recv_grads + (tokens_nums_grads,)

    # get grad function input
    # 'recv_data' needs to be passed as keyword argument into the function
    recv_data = input_tensor[-1]
    input_tensor = tuple(input_tensor[:-1])

    if auto_pp_spliting:
        # get grad function
        grad_fn = C.GradOperation(get_all=True, get_by_list=True, sens_param=True)(model, weight)

        # calculate grads
        grads_output = grad_fn(*input_tensor, recv_data=recv_data, sens=recv_grads)
        dout, weight_grad = grads_output[0], grads_output[1]

        # bprop for recv_data
        dout = dout[-1]
    else:
        # set input tensor for backpropagation
        if not is_pipeline_first_stage():
            model.set_input_tensor(recv_data)

        # get grad function
        grad_fn = C.GradOperation(get_by_list=True, sens_param=True)(model, weight)

        # calculate grads
        weight_grad_res = grad_fn(*input_tensor, recv_data=None, sens=recv_grads)
        weight_grad_list = list(weight_grad_res)

        # get dout and weight_grad
        dout = weight_grad_list.pop(0)
        weight_grad = tuple(weight_grad_list)

    # the first stage do not require backpropagation
    if is_pipeline_first_stage():
        dout = None

    # accumulate grads between multi-micro input
    if accumulate_grads is None:
        accumulate_grads = weight_grad
    else:
        accumulate_grads = accumulate_grads_func(list(accumulate_grads), list(weight_grad))

    return dout, accumulate_grads


# pylint: disable=C0103
def forward_backward_pipelining_with_interleaving(
        model,
        num_microbatches,
        seq_length,
        micro_batch_size,
        *input_data_tuple,
        decoder_seq_length=None,
        forward_only=False,
        collect_non_loss_data=False,
        first_val_step=None,
        config=None,
        scale_sense=1.0,
        total_tokens_nums=None,
        **input_data_dict
):
    """ Pipeline with interleaving wrapper for split model run forward and backward """
    if not isinstance(model, nn.CellList):
        raise TypeError("The 'model' input of 'forward_backward_pipelining_with_interleaving' must be nn.CellList. "
                        "Here it is recommended to use the 'get_model' function to build the model")
    if len(model) < 2:
        raise TypeError("The length of 'model' must be greater than 1 ")
    model_type_correction = all(isinstance(model_chunk, PipelineCell) for model_chunk in model)
    if not model_type_correction:
        raise TypeError("Each chunk of 'model' must be PipelineCell for "
                        "'forward_backward_pipelining_with_interleaving' function.")
    if decoder_seq_length is not None:
        raise NotImplementedError("decoder_seq_length is not supported for now.")
    if collect_non_loss_data:
        raise NotImplementedError("collect_non_loss_data is not supported for now.")
    if first_val_step is not None:
        raise NotImplementedError("first_val_step is not supported for now.")
    if config is None:
        raise ValueError("Please input config for 'forward_backward_pipelining_without_interleaving' function")
    if forward_only:
        raise NotImplementedError("'forward_only' input of pipeline interleaved is not supported for now.")

    # set grad
    requires_grad = True
    if forward_only:
        requires_grad = False
    for sub_model in model:
        sub_model.set_grad(requires_grad=requires_grad)

    # different processes depending on the model
    auto_pp_spliting = False

    # init p2p class
    p2p_primitive = P2P_Primitive(config=config.model_config)

    # get model weights and merge `set_hidden_states` parameter
    weights = [sub_model.trainable_params() for sub_model in model]
    set_hidden_states_parameters = []
    for i, _ in enumerate(weights):
        set_hidden_states_parameter = get_set_hidden_states_parameter(model[i])
        weights[i].insert(0, set_hidden_states_parameter)
        set_hidden_states_parameters.append(set_hidden_states_parameter)

    # get config value
    hidden_size = config.model_config.hidden_size
    use_sequence_parallel = config.model_config.parallel_config.use_sequence_parallel
    overlap_p2p_comm = config.model_config.parallel_config.overlap_p2p_comm
    calculate_per_token_loss = config.training_config.calculate_per_token_loss
    data_layout = config.model_config.dataset_config.data_layout

    # correct tensor shape if use seq parallel or context parallel
    tensor_shape = correct_p2p_shape(seq_length, hidden_size, micro_batch_size, data_layout, use_sequence_parallel)[0]

    # save each forward process input data for running backward
    if not forward_only:
        input_tensors = [[] for _ in range(len(model))]
        output_tensor_grads = [[] for _ in range(len(model))]
        accumulate_grads_list = [None] * len(model)
    accumulate_loss = []
    tokens_nums_list = []
    logits = []

    # warm up process
    pp_world_size = get_pp_world_size()
    pp_rank = get_pp_rank()
    num_model_chunks = len(model)
    total_num_microbatches = num_microbatches * num_model_chunks
    if num_microbatches % pp_world_size != 0:
        raise RuntimeError("When using pipeline with interleaved, "
                           "the 'num_microbatches' must be divisible by pipeline world size.")
    if total_num_microbatches == pp_world_size:
        warm_up_steps = total_num_microbatches
        all_warmup_steps = True
    else:
        warm_up_steps = 2 * (pp_world_size - pp_rank - 1)
        warm_up_steps += (num_model_chunks - 1) * pp_world_size
        warm_up_steps = min(warm_up_steps, total_num_microbatches)
        all_warmup_steps = False

    def run_forward_helper(microbatch_id, accumulate_loss):
        """ run forward helper function """
        # get model vpp rank
        model_chunk_id = get_model_chunk_id(microbatch_id, pp_world_size, num_model_chunks, forward=True)
        set_vpp_rank(model_chunk_id)

        # get forward micro data and recv_data
        micro_input_data = input_tensors[model_chunk_id][-1]
        input_tensor = micro_input_data[-1]
        micro_input_data = micro_input_data[:-1]

        # run forward
        output_tensor, accumulate_loss, _ = run_forward(*micro_input_data,
                                                        model=model[model_chunk_id],
                                                        accumulate_loss=accumulate_loss,
                                                        logits=logits,
                                                        recv_data=input_tensor,
                                                        auto_pp_spliting=auto_pp_spliting,
                                                        calculate_per_token_loss=calculate_per_token_loss,
                                                        tokens_nums_list=tokens_nums_list)

        if forward_only:
            input_tensors[model_chunk_id].pop()

        return output_tensor

    def run_backward_helper(microbatch_id):
        """ run forward helper function """
         # get model vpp rank
        model_chunk_id = get_model_chunk_id(microbatch_id, pp_world_size, num_model_chunks, forward=False)
        set_vpp_rank(model_chunk_id)

        # get micro data and recv_grads
        input_tensor = input_tensors[model_chunk_id].pop(0)
        recv_grads = output_tensor_grads[model_chunk_id].pop(0)

        # run backward
        dout, accumulate_grads = run_backward(*input_tensor,
                                              recv_grads=recv_grads,
                                              model=model[model_chunk_id],
                                              weight=weights[model_chunk_id],
                                              scale_sense=scale_sense,
                                              accumulate_loss=accumulate_loss,
                                              accumulate_grads=accumulate_grads_list[model_chunk_id],
                                              micro_size=num_microbatches,
                                              logits=logits,
                                              auto_pp_spliting=auto_pp_spliting,
                                              calculate_per_token_loss=calculate_per_token_loss,
                                              total_tokens_nums=total_tokens_nums,
                                              tokens_nums_list=tokens_nums_list)
        accumulate_grads_list[model_chunk_id] = accumulate_grads
        return dout

    forward_reqs = None
    backward_reqs = None
    input_tensor = None

    # set virtual rank
    set_vpp_rank(0)

    # get first iteration micro data input
    slice_index_list = [0] * num_model_chunks
    slice_index = slice_index_list[0]
    micro_input_data = get_micro_input(slice_index, micro_batch_size, model[0], *input_data_tuple, **input_data_dict)
    micro_input_data = micro_input_data + (p2p_primitive.recv_forward(tensor_shape),)
    input_tensors[0].append(micro_input_data)

    # warm up process
    for i in range(warm_up_steps):
        # if overlap_p2p_comm is True, wait for comm stream
        if forward_reqs is not None:
            # pylint: disable=E1133
            for req in forward_reqs:
                req.wait()

        # run warm up forward
        ouput_tensor = run_forward_helper(i, accumulate_loss)

        # decide communication operation
        recv_prev = True
        next_model_chunk_id = get_model_chunk_id(i + 1, pp_world_size, num_model_chunks, forward=True)
        if is_pipeline_first_stage(ignore_virtual=True) and next_model_chunk_id == 0:
            recv_prev = False
        if i == total_num_microbatches - 1:
            recv_prev = False
        if is_pipeline_last_stage():
            ouput_tensor = None

        # warm up send and recv
        if not overlap_p2p_comm:
            if i == warm_up_steps - 1 and not forward_only and not all_warmup_steps:
                if is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                else:
                    recv_next = True
                input_tensor, recv_grads = \
                    p2p_primitive.send_forward_backward_recv_forward_backward(ouput_tensor,
                                                                              None,
                                                                              recv_prev=recv_prev,
                                                                              recv_next=recv_next,
                                                                              tensor_shape=tensor_shape)
                output_tensor_grads[-1].append(recv_grads)
            else:
                input_tensor = p2p_primitive.send_forward_recv_forward(ouput_tensor,
                                                                       recv_prev=recv_prev,
                                                                       tensor_shape=tensor_shape)
            # slice micro input data for next iteration
            cur_model_chunk_id = get_model_chunk_id(i, pp_world_size, num_model_chunks, forward=True)
            slice_index_list[cur_model_chunk_id] += 1
            slice_index = slice_index_list[next_model_chunk_id]
            micro_input_data = get_micro_input(slice_index,
                                               micro_batch_size,
                                               model[next_model_chunk_id],
                                               *input_data_tuple,
                                               **input_data_dict)
            input_tensors[next_model_chunk_id].append(micro_input_data + (input_tensor,))
        else:
            raise NotImplementedError("Interleaved pipeline do not support 'overlap_p2p_comm=True' now.")

    # 1F1B process
    steady_steps = total_num_microbatches - warm_up_steps
    for i in range(steady_steps):
        forward_i = i + warm_up_steps
        if not overlap_p2p_comm:
            # run forward
            ouput_tensor = run_forward_helper(forward_i, accumulate_loss)

            # run backward
            dout = run_backward_helper(i)

            # set forward virtual id
            forward_model_chunk_id = get_model_chunk_id(forward_i, pp_world_size, num_model_chunks, forward=True)
            set_vpp_rank(forward_model_chunk_id)
            if is_pipeline_last_stage():
                ouput_tensor = None

            # set backward virtual id
            backward_model_chunk_id = get_model_chunk_id(i, pp_world_size, num_model_chunks, forward=False)
            set_vpp_rank(backward_model_chunk_id)
            if is_pipeline_first_stage():
                dout = None

            # decide communication operation
            # recv forward
            recv_prev = True
            next_forward_model_chunk_id = get_model_chunk_id(forward_i + 1,
                                                             pp_world_size,
                                                             num_model_chunks,
                                                             forward=True)
            if is_pipeline_first_stage(ignore_virtual=True):
                if next_forward_model_chunk_id == 0:
                    recv_prev = False

            # recv backward
            recv_next = True
            next_backward_model_chunk_id = get_model_chunk_id(i + 1,
                                                              pp_world_size,
                                                              num_model_chunks,
                                                              forward=False)
            if is_pipeline_last_stage(ignore_virtual=True):
                if next_backward_model_chunk_id == num_model_chunks - 1:
                    recv_next = False

            # if running last micro batch data, do not recv anything
            if i == steady_steps - 1:
                recv_prev = False

            input_tensor, recv_grads = \
                p2p_primitive.send_forward_backward_recv_forward_backward(ouput_tensor,
                                                                          dout,
                                                                          recv_prev=recv_prev,
                                                                          recv_next=recv_next,
                                                                          tensor_shape=tensor_shape)
            if not i == steady_steps - 1:
                # slice micro input data for next iteration
                slice_index_list[forward_model_chunk_id] += 1
                slice_index = slice_index_list[next_forward_model_chunk_id]
                micro_input_data = get_micro_input(slice_index,
                                                   micro_batch_size,
                                                   model[next_model_chunk_id],
                                                   *input_data_tuple,
                                                   **input_data_dict)
                # save comm tensor for next 1F1B iteration
                input_tensors[next_forward_model_chunk_id].append(micro_input_data + (input_tensor,))
            output_tensor_grads[next_backward_model_chunk_id].append(recv_grads)
        else:
            raise NotImplementedError("Interleaved pipeline do not support 'overlap_p2p_comm=True' now.")

    # wait backward comm stream
    if overlap_p2p_comm and backward_reqs is not None:
        # pylint: disable=E1133
        for req in backward_reqs:
            req.wait()

    # recv grad for running cooldown
    if all_warmup_steps:
        recv_grads = p2p_primitive.recv_backward(tensor_shape)
        output_tensor_grads[-1].append(recv_grads)

    # cooldown process
    for i in range(steady_steps, total_num_microbatches):
        dout = run_backward_helper(i)
        next_backward_model_chunk_id = get_model_chunk_id(i + 1,
                                                          pp_world_size,
                                                          num_model_chunks,
                                                          forward=False)
        # decide communication operation
        recv_next = True
        if is_pipeline_last_stage(ignore_virtual=True) and \
            next_backward_model_chunk_id == num_model_chunks - 1:
            recv_next = False
        if i == total_num_microbatches - 1:
            recv_next = False
        recv_grads = p2p_primitive.send_backward_recv_backward(dout,
                                                               recv_next=recv_next,
                                                               tensor_shape=tensor_shape)
        output_tensor_grads[next_backward_model_chunk_id].append(recv_grads)

    # get all model chunk param name
    weights_name = []
    for model_chunk_params in weights:
        model_chunk_params_name = [param.name for param in model_chunk_params]
        # remove 'set_hidden_states' parameter
        weights_name.extend(model_chunk_params_name[1:])
    del weights

    # merge grads
    all_model_chunk_grads = None
    for model_chunk_grad in accumulate_grads_list:
        if all_model_chunk_grads is None:
            all_model_chunk_grads = model_chunk_grad
        else:
            all_model_chunk_grads += model_chunk_grad
    del accumulate_grads_list

    # AllReduce embedding (if shared embedding weight in first stage and last stage)
    all_model_chunk_grads = all_reduce_share_embedding(list(all_model_chunk_grads),
                                                       weights_name,
                                                       model)

    # get return value of loss and logits
    accumulate_loss, logits = calculate_loss_and_logits(accumulate_loss,
                                                        logits,
                                                        num_microbatches,
                                                        tokens_nums_list,
                                                        total_tokens_nums,
                                                        calculate_per_token_loss)

    # reset set_hidden_states attr
    for cur_set_hidden_states_param in set_hidden_states_parameters:
        cur_set_hidden_states_param.requires_grad = False

    return accumulate_loss, logits, all_model_chunk_grads


# pylint: disable=C0103
def forward_backward_pipelining_without_interleaving(
        model,
        num_microbatches,
        seq_length,
        micro_batch_size,
        *input_data_tuple,
        decoder_seq_length=None,
        forward_only=False,
        collect_non_loss_data=False,
        first_val_step=None,
        config=None,
        scale_sense=1.0,
        total_tokens_nums=None,
        **input_data_dict
):
    """ Pipeline 1F1B wrapper for split model run forward and backward """
    if not isinstance(model, PipelineCell):
        raise TypeError("The 'model' input of 'forward_backward_pipelining_without_interleaving' must be PipelineCell")
    if decoder_seq_length is not None:
        raise NotImplementedError("decoder_seq_length is not supported for now.")
    if collect_non_loss_data:
        raise NotImplementedError("collect_non_loss_data is not supported for now.")
    if first_val_step is not None:
        raise NotImplementedError("first_val_step is not supported for now.")
    if config is None:
        raise ValueError("Please input config for 'forward_backward_pipelining_without_interleaving' function")

    # set grad
    requires_grad = True
    if forward_only:
        requires_grad = False
    model.set_grad(requires_grad=requires_grad)

    # different processes depending on the model
    auto_pp_spliting = not model.new_model

    # init p2p class
    p2p_primitive = P2P_Primitive(config=config.model_config)

    # get model weights and merge `set_hidden_states` parameter
    weights = model.trainable_params()
    set_hidden_states_param = get_set_hidden_states_parameter(model)
    weights.insert(0, set_hidden_states_param)

    # get config value
    hidden_size = config.model_config.hidden_size
    use_sequence_parallel = config.model_config.parallel_config.use_sequence_parallel
    calculate_per_token_loss = config.training_config.calculate_per_token_loss
    data_layout = config.model_config.dataset_config.data_layout

    # correct tensor shape if use seq parallel or context parallel
    recv_tensor_shapes = correct_p2p_shape(seq_length, hidden_size,\
                                           micro_batch_size, data_layout, use_sequence_parallel)
    send_tensor_shapes = correct_p2p_shape(seq_length, hidden_size,\
                                           micro_batch_size, data_layout, use_sequence_parallel)

    # save each forward process input data for running backward
    if not forward_only:
        input_tensors = []
    accumulate_loss = []
    tokens_nums_list = []
    logits = []

    # get warm up stage steps
    warm_up_steps = min(get_pp_world_size() - get_pp_rank() - 1, num_microbatches)
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
        output_tensor, accumulate_loss,\
            micro_input_data = run_forward(*micro_input_data,
                                           model=model,
                                           accumulate_loss=accumulate_loss,
                                           logits=logits,
                                           recv_data=input_tensor,
                                           auto_pp_spliting=auto_pp_spliting,
                                           calculate_per_token_loss=calculate_per_token_loss,
                                           tokens_nums_list=tokens_nums_list)

        # save micro input data for backward
        if not forward_only:
            input_tensors.append(micro_input_data)

        # send forward result to next stage
        send_forward(output_tensor, send_tensor_shapes, p2p_primitive)

    # prepare input data for 1F1B
    steady_steps = num_microbatches - warm_up_steps
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
        output_tensor, accumulate_loss, \
            micro_input_data = run_forward(*micro_input_data,
                                           model=model,
                                           accumulate_loss=accumulate_loss,
                                           logits=logits,
                                           recv_data=input_tensor,
                                           auto_pp_spliting=auto_pp_spliting,
                                           calculate_per_token_loss=calculate_per_token_loss,
                                           tokens_nums_list=tokens_nums_list)

        if forward_only:
            # only send forward result
            send_forward(output_tensor, send_tensor_shapes, p2p_primitive)

            # recv data from prev stage
            if not i == steady_steps - 1:
                slice_index += 1
                micro_input_data = get_micro_input(slice_index,
                                                   micro_batch_size,
                                                   model,
                                                   *input_data_tuple,
                                                   **input_data_dict)
                input_tensor = recv_forward(recv_tensor_shapes, p2p_primitive)
        else:
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
                                                  accumulate_loss=accumulate_loss,
                                                  accumulate_grads=accumulate_grads,
                                                  micro_size=num_microbatches,
                                                  logits=logits,
                                                  auto_pp_spliting=auto_pp_spliting,
                                                  calculate_per_token_loss=calculate_per_token_loss,
                                                  total_tokens_nums=total_tokens_nums,
                                                  tokens_nums_list=tokens_nums_list)

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

    if not forward_only:
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
                                                  accumulate_loss=accumulate_loss,
                                                  accumulate_grads=accumulate_grads,
                                                  micro_size=num_microbatches,
                                                  logits=logits,
                                                  auto_pp_spliting=auto_pp_spliting,
                                                  calculate_per_token_loss=calculate_per_token_loss,
                                                  total_tokens_nums=total_tokens_nums,
                                                  tokens_nums_list=tokens_nums_list)
            send_backward(dout, recv_tensor_shapes, p2p_primitive)

        # AllReduce embedding (if shared embedding weight in first stage and last stage)
        weights.pop(0)
        weight_name = [weight.name for weight in weights]
        accumulate_grads = all_reduce_share_embedding(list(accumulate_grads),
                                                      weight_name,
                                                      model)

    # get return value of loss and logits
    accumulate_loss, logits = calculate_loss_and_logits(accumulate_loss,
                                                        logits,
                                                        num_microbatches,
                                                        tokens_nums_list,
                                                        total_tokens_nums,
                                                        calculate_per_token_loss)

    # reset set_hidden_states attr
    set_hidden_states_param.requires_grad = False

    return accumulate_loss, logits, accumulate_grads


def get_model_chunk_id(microbatch_id, pp_world_size, num_model_chunks, forward):
    """ get model chunk id by micro batch id """
    micro_group = microbatch_id % (pp_world_size * num_model_chunks)
    model_chunk_id = micro_group // pp_world_size
    if not forward:
        model_chunk_id = num_model_chunks - model_chunk_id - 1
    return model_chunk_id


def correct_p2p_shape(seq_length, hidden_size, micro_batch_size, data_layout, use_sequence_parallel=False):
    """
    Correct right tensor shape under context parallel or sequence parallel.
    """
    seq_length = seq_length // get_cp_world_size()
    if use_sequence_parallel:
        seq_length = seq_length // get_tp_world_size()
    if data_layout == "BSH":
        return ((micro_batch_size, seq_length, hidden_size),)
    return ((seq_length, micro_batch_size, hidden_size),)


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


def all_reduce_share_embedding(grads, weight_name, model):
    """ Reduce share embedding grads in embedding comm group """
    if is_rank_in_embedding_group(ignore_virtual=True):
        if isinstance(model, nn.CellList):
            shared_weight_name_list = []
            untie_embeddings_and_output_weights = model[0].untie_embeddings_and_output_weights
            for sub_model in model:
                shared_weight_name_list.extend(sub_model.shared_weight_name_list)
        else:
            untie_embeddings_and_output_weights = model.untie_embeddings_and_output_weights
            shared_weight_name_list = model.shared_weight_name_list
        if get_pp_world_size() > 1 and not untie_embeddings_and_output_weights:
            idx = None
            shared_weight_index = []
            weight_grad = None
            for i, name in enumerate(weight_name):
                if name in shared_weight_name_list:
                    weight_grad = weight_grad + grads[i] if weight_grad is not None else grads[i]
                    shared_weight_index.append(i)
            weight_grad = P.AllReduce(op='sum', group=get_embedding_group())(weight_grad)
            for idx in shared_weight_index:
                grads[idx] = weight_grad
    return grads


def calculate_loss_and_logits(accumulate_loss,
                              logits,
                              num_microbatches,
                              tokens_nums_list,
                              total_tokens_nums,
                              calculate_per_token_loss):
    """ calculate loss and logits"""
    total_loss = Tensor(0.0, mstype.float32)
    if is_pipeline_last_stage(ignore_virtual=True):
        # calculate final loss
        if not accumulate_loss:
            accumulate_loss = total_loss
        else:
            if calculate_per_token_loss:
                for micro_loss, micro_tokens_nums in zip(accumulate_loss, tokens_nums_list):
                    weighted_loss = (micro_loss * micro_tokens_nums) / total_tokens_nums
                    total_loss += weighted_loss
                accumulate_loss = total_loss
            else:
                accumulate_loss = sum(accumulate_loss) / num_microbatches
        # concat micro logits
        logits = mint.cat(logits, dim=0) if logits else None
    else:
        accumulate_loss = total_loss
        logits = None
    return accumulate_loss, logits
