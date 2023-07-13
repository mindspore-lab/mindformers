# Copyright 2023 Huawei Technologies Co., Ltd
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
"""GLM model."""
import os
import numpy as np

import mindspore as ms
from mindspore import dtype as mstype
from mindspore import nn, ops, Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter

from mindformers.mindformer_book import MindFormerBook
from mindformers.modules.transformer import VocabEmbedding, EmbeddingOpParallelConfig, OpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.core.loss import CrossEntropyLoss
from mindformers.modules.layers import LayerNorm
from mindformers.tools.utils import is_version_ge
from mindformers.pet.tuners.pet_adapter import PetAdapter
from mindformers.pet.tuners.lora_adapter import LoraAdapter

from .glm_config import GLMConfig
from .layers import DeepNormWithGLULayer
from ..base_model import BaseModel

#  Get MS backend: 0 vm 1 GE
is_ge = os.getenv('MS_ENABLE_GE')
if is_ge == '1':
    jit_level = "O3"
else:
    jit_level = "O1"

default_dpmp_config = OpParallelConfig()
default_embedding_parallel_config = EmbeddingOpParallelConfig()

__all__ = ['GLMForPreTraining', 'GLMChatModel', 'GLMForPreTrainingWithLora', 'GLMChatModelWithLora']


def topk_fun(logits, topk=5):
    """Get topk"""
    batch_value = []
    batch_index = []
    for i in range(logits.shape[0]):
        target_column = logits[i].tolist()
        sorted_array = [(k, v) for k, v in enumerate(target_column)]
        sorted_array.sort(key=lambda x: x[1], reverse=True)
        topk_array = sorted_array[:topk]
        index, value = zip(*topk_array)
        batch_value.append(value)
        batch_index.append(index)
    return np.array(batch_value), np.array(batch_index)


def batch_select(data, index):
    """bathc operation to sorted_logits[:, :top_p_num]"""
    output = []
    for i in range(data.shape[0]):
        res = data[i, :index[i]]
        output.append(res.reshape(1, -1))
    return np.concatenate(output, 0)


def sampler(log_probs_revised, top_p, top_k, use_pynative=False):
    """Convert the log_probs to probability"""
    if use_pynative:
        logits = P.Pow()(np.e, Tensor(log_probs_revised, mstype.float32))
    else:
        logits = np.power(np.e, np.array(log_probs_revised, np.float32))

    # If top_p is less than 1.0, use top_p sampling
    if top_p < 1.0:
        # Only consider the 5000 largest logits to reduce computation
        if use_pynative:
            sorted_logits, index = P.TopK(sorted=True)(logits, 5000)
            cumsum_logits = P.CumSum()(sorted_logits, 1)
            cumsum_logits = cumsum_logits.asnumpy()
            index = index.asnumpy()
            sorted_logits = sorted_logits.asnumpy()
        else:
            sorted_logits, index = topk_fun(logits, 5000)
            cumsum_logits = np.cumsum(sorted_logits, 1)
        cumsum_logits = cumsum_logits
        index = index
        sorted_logits = sorted_logits
        top_p_num = np.sum(cumsum_logits < top_p, axis=-1) + 1
        # Get the corresponding probs and indices
        probs = batch_select(sorted_logits, top_p_num)
        p_args = batch_select(index, top_p_num)
        p = probs / np.sum(probs, -1, keepdims=True)
        # if top_p is set to 1.0, use top_k sampling
    else:
        # Get the corresponding probs and indices
        if use_pynative:
            probs, p_args = P.TopK(sorted=True)(logits, top_k)
            probs = probs.asnumpy()
            p_args = p_args.asnumpy()
        else:
            probs, p_args = topk_fun(logits, top_k)
        probs = probs
        p_args = p_args
        # Avoid rounding error
        for i in range(probs.shape[0]):
            if np.sum(probs[i]) == 0:
                probs[i] = np.array([1 / top_k for _ in range(top_k)])
        p = probs / np.sum(probs, -1, keepdims=True)
    return p, p_args


def precision_correct(p, top_p, top_k, batch_size):
    # Avoid rounding error
    if top_p == 1:
        for i in range(batch_size):
            if np.sum(p[i]) == 0:
                p[i] = np.array([1 / top_k for _ in range(top_k)])
        p = p / np.sum(p, -1, keepdims=True)
    return p


class ProcessLogits(nn.Cell):
    r"""Process logits into probability distribution."""

    def __init__(self, use_past=False):
        super(ProcessLogits, self).__init__()
        self.e = ms.Tensor(np.e)
        self.gather = P.Gather()
        self.logsoftmax = P.LogSoftmax()
        self.reshape = P.Reshape()
        self.use_past = use_past

    def construct(self, logits, current_index=None, is_first_iteration=False):
        logits = logits.reshape(-1, logits.shape[-1])
        if self.use_past and not is_first_iteration:
            logits = logits
        elif current_index is not None:
            index = current_index.view(-1,)
            logits = self.gather(logits, index, 0)
        outputs = self.logsoftmax(logits)
        outputs = F.tensor_pow(self.e, outputs)
        return outputs


class GLMModel(nn.Cell):
    """
    The backbone of GLM network

    Args:
        config (GLMConfig): The config of network.
        op_parallel_config (optional): Operator parallel strategy. Default: `OpParallelConfig()`.
        embed_parallel_config (optional): Operator parallel strategy. Default: `EmbeddingOpParallelConfig()`.
    """
    def __init__(self,
                 config,
                 op_parallel_config=default_dpmp_config,
                 embed_parallel_config=default_embedding_parallel_config):
        super(GLMModel, self).__init__()
        # recording parameters
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.seq_length = config.seq_length
        self.use_past = config.use_past
        layernorm = LayerNorm
        if config.parallel_config:
            op_parallel_config = config.parallel_config

        # create embedding parameters
        if is_version_ge(ms.__version__, '2.0.0'):
            self.embedding_dropout = nn.Dropout(p=config.embedding_dropout_prob)
        else:
            self.embedding_dropout = nn.Dropout(keep_prob=1 - config.embedding_dropout_prob)

        embed_parallel_config.data_parallel = op_parallel_config.data_parallel
        embed_parallel_config.model_parallel = op_parallel_config.model_parallel
        embed_parallel_config.vocab_emb_dp = False
        self.word_embeddings = VocabEmbedding(vocab_size=config.vocab_size, embedding_size=config.hidden_size,
                                              parallel_config=embed_parallel_config)

        self.matmul = ops.MatMul().shard(((1, 1), (1, embed_parallel_config.model_parallel)))
        self.transpose = ops.Transpose().shard(((embed_parallel_config.model_parallel, 1),))

        def get_layer(layer_id):
            return DeepNormWithGLULayer(
                self.num_layers,
                self.hidden_size,
                self.num_heads,
                config.batch_size,
                config.attention_dropout_rate,
                config.hidden_dropout_rate,
                config.layernorm_epsilon,
                layer_id,
                max_seq_len=self.seq_length,
                inner_hidden_size=config.inner_hidden_size,
                hidden_size_per_attention_head=config.hidden_size_per_attention_head,
                layernorm_order=config.layernorm_order,
                layernorm=layernorm,
                use_bias=True,
                activation_func=config.activation_func,
                position_encoding_2d=config.position_encoding_2d,
                params_dtype=config.param_init_type,
                layernorm_dtype=config.layernorm_compute_type,
                softmax_dtype=config.softmax_compute_type,
                compute_dtype=config.compute_dtype,
                use_past=self.use_past,
                parallel_config=op_parallel_config,
            )

        self.layers = nn.CellList(
            [get_layer(layer_id) for layer_id in range(config.num_layers)])

        # Final layer norm before output.
        self.use_final_layernorm = config.use_final_layernorm
        if config.use_final_layernorm:
            self.final_layernorm = layernorm(config.hidden_size, eps=config.layernorm_epsilon)
            self.final_layernorm.shard(((op_parallel_config.data_parallel, 1, 1),))

    def construct(self, input_ids, position_ids, attention_mask, init_reset=True, batch_valid_length=None):
        """
        Get output logits

        Inputs:
            input_ids (Tensor): The tokenized inputs with dtype int32.
            input_mask (Tensor): The mask indicating whether each position is a valid input.
            position_ids (Tensor): Used to identify each token's position in the list of tokens.
            attention_mask (Tensor): Used when batching sequences together.
            init_reset (bool, optional): Default: True.
            batch_valid_length (Tensor, optional): Default: None.

        Returns:
            logits (Tensor): The output logit of backbone.
            table (Tensor): The embedding table for the vocabulary.
        """
        if attention_mask is None:
            attention_mask = ops.ones((1, 1), mstype.int32)

        hidden_states, table = self.word_embeddings(input_ids)

        hidden_states = self.embedding_dropout(hidden_states)

        for i in range(self.num_layers):
            layer_ret = self.layers[i](hidden_states, attention_mask, position_ids, init_reset, batch_valid_length)

            if isinstance(layer_ret, tuple):
                layer_ret = layer_ret[0]
            hidden_states = layer_ret

        # Final layer norm.
        if self.use_final_layernorm:
            logits = self.final_layernorm(hidden_states)
        else:
            logits = hidden_states

        return logits, table


class GLMHead(nn.Cell):
    r"""Head for GLM to get the logits of each token in the vocab."""

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 param_init_type=mstype.float32,
                 compute_dtype=mstype.float16,
                 embed_parallel_config=None):
        super(GLMHead, self).__init__()
        self.param_init_type = param_init_type
        self.compute_dtype = compute_dtype
        self.weight = Parameter(initializer("normal", [vocab_size, hidden_size], compute_dtype), name="weight")
        self.transpose = ops.Transpose().shard(((embed_parallel_config.model_parallel, 1),))
        self.matmul = ops.MatMul(transpose_b=True).shard(
            ((embed_parallel_config.data_parallel, 1), (embed_parallel_config.model_parallel, 1)))

    def construct(self, state, embedding_table=None):
        """Get vocab probs"""
        state = F.reshape(state, (-1, F.shape(state)[-1]))
        state = ops.cast(state, self.compute_dtype)
        if embedding_table is None:
            embedding_table = self.weight
        embedding_table = self.cast(embedding_table, self.compute_dtype)
        logits_parallel = self.matmul(state, embedding_table)
        return logits_parallel


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class GLMForPreTraining(BaseModel):
    r"""
    Provide glm training loss or logits through network.

    Args:
        config (GLMConfig): The config of GLMModel.
    """
    _support_list = MindFormerBook.get_model_support_list()['glm']

    def __init__(self, config: GLMConfig):
        super(GLMForPreTraining, self).__init__(config)
        self.config = config
        self.position_encoding_2d = config.position_encoding_2d
        self.transformer = GLMModel(config)
        self.lm_head = GLMHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            param_init_type=config.param_init_type,
            compute_dtype=config.compute_dtype,
            embed_parallel_config=config.parallel_config)
        self.stridedslice = ops.StridedSlice().shard(((1, 1),))
        self.loss = CrossEntropyLoss(parallel_config=config.parallel_config)
        self.gmask = config.gmask_token_id
        self.bos_token_id = config.bos_token_id
        self.ones = P.Ones()
        self.load_checkpoint(config)

    def get_masks_np(self, input_ids):
        batch_size, seq_length = input_ids.shape
        context_lengths = [list(seq).index(self.config.bos_token_id) for seq in input_ids]
        attention_mask = np.tril(np.ones((batch_size, seq_length, seq_length)))
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        attention_mask = np.expand_dims(attention_mask, axis=1)
        attention_mask = np.array(attention_mask < 0.5, np.bool_)
        return attention_mask

    def get_position_ids_np(self, input_ids, mask_positions, use_gmasks=None):
        """Get position ids from input_ids and mask_positions with numpy"""
        batch_size, seq_length = input_ids.shape
        if use_gmasks is None:
            use_gmasks = [False] * batch_size
        context_lengths = [list(seq).index(self.config.bos_token_id) for seq in input_ids]
        if self.config.position_encoding_2d:
            position_ids = np.repeat(np.expand_dims(np.arange(seq_length), 0), batch_size, axis=0)
            for i, context_length in enumerate(context_lengths):
                position_ids[i, context_length:] = mask_positions[i]
            block_position_ids = [np.concatenate((
                np.zeros(context_length, np.int32),
                np.arange(seq_length - context_length, dtype=np.int32) + 1
            )) for context_length in context_lengths]
            block_position_ids = np.stack(block_position_ids, axis=0)
            position_ids = np.stack((position_ids, block_position_ids), axis=1)
        else:
            position_ids = np.repeat(np.expand_dims(np.arange(seq_length), 0), batch_size, axis=0)
            for i, context_length in enumerate(context_lengths):
                if not use_gmasks[i]:
                    position_ids[context_length:] = mask_positions[i]
        return position_ids

    def create_position_ids_np(self, input_ids):
        """Get position ids from input_ids with numpy"""
        mask, gmask = self.config.mask_token_id, self.config.gmask_token_id
        seqs = list(input_ids)

        mask_positions, use_gmasks = [], []
        for seq in seqs:
            mask_token = gmask if gmask in seq else mask
            use_gmask = mask_token == gmask
            mask_positions.append(list(seq).index(mask_token))
            use_gmasks.append(use_gmask)
        position_ids = self.get_position_ids_np(input_ids, mask_positions, use_gmasks=None)
        return position_ids

    def _incremental_infer(self,
                           input_ids,
                           current_index,
                           valid_length_each_example,
                           position_ids=None,
                           attention_mask=None):
        # Claim the first graph
        if self.is_first_iteration:
            self.add_flags_recursive(is_first_iteration=True)
            res = self(
                input_ids=Tensor(input_ids, mstype.int32),
                # input_ids (1,512) int32
                position_ids=Tensor(position_ids, mstype.int32),
                # position_ids (1, 2, 512) int32
                attention_mask=Tensor(attention_mask, mstype.float32),
                # attention_mask (1, 1, 512, 512) float32
                input_position=Tensor(current_index, mstype.int32),
                init_reset=Tensor([False], mstype.bool_),  # init_reset (1,) bool False
                batch_valid_length=Tensor([valid_length_each_example], mstype.int32)
            )  # batch_valid_length (1,) int32 4
            # first iter done, go to other iters
            self.is_first_iteration = False
        else:
            self.add_flags_recursive(is_first_iteration=False)

            inputs_tmp = []
            position_ids_tmp = []
            attention_mask_tmp = []
            for i in range(len(current_index)):
                current_index_tmp = int(current_index[i]) - i * input_ids.shape[1]  # multibatch
                # use numpy to slice array to avoid complie ascend slice op
                inputs_tmp.append(input_ids[i][current_index_tmp:current_index_tmp + 1])
                position_ids_tmp.append(position_ids[i][..., current_index_tmp:current_index_tmp + 1])
                attention_mask_tmp.append(attention_mask[i][:, current_index_tmp:current_index_tmp + 1, :])
            inputs_tmp = np.array(inputs_tmp, dtype=np.int32)
            position_ids_tmp = np.array(position_ids_tmp, dtype=np.int32)
            attention_mask_tmp = np.array(attention_mask_tmp, dtype=np.float32)
            res = self(
                input_ids=Tensor(inputs_tmp, mstype.int32),
                # input_ids (1,512) int32
                position_ids=Tensor(position_ids_tmp, mstype.int32),
                # position_ids (1, 2, 1) int32
                attention_mask=Tensor(attention_mask_tmp, mstype.float32),
                # attention_mask (1, 1, 1, 512) float32
                input_position=Tensor(current_index, mstype.int32),
                init_reset=Tensor([True], mstype.bool_),  # init_reset (1,) bool True
                batch_valid_length=Tensor([valid_length_each_example], mstype.int32)
            )  # batch_valid_length (1,) int32 5

        return res

    def _forward(self,
                 origin_inputs,
                 top_k,
                 top_p,
                 repetition_penalty,
                 max_length,
                 eos_token_id,
                 streamer=None,
                 pad_token_id=None):
        """
        Text generation given the model and origin inputs

        Inputs:
            model: The model to run the prediction
            end_token(int): The model will stop generating the words when it reaches the end_token.
            origin_inputs(list): The prompt for generation, should be a list of ids.
            model_origin_max_length(int): The sequence length of the model trained.
            max_length(int):  The maximum of generated length.
            vocab_size(int): The vocabulary length of the model.
            config: Inference configurations.
            streamer: Streamer object that will be used to stream the generated sequences.

        Returns:
            outputs: the ids for the generated text
        """
        if pad_token_id is None:
            pad_token_id = 0
        # Get configurations for inference
        use_pynative = True

        if streamer is not None:
            streamer.put(origin_inputs[0])

        batch_size = origin_inputs.shape[0]
        is_npu_acceleration = self.config.is_npu_acceleration
        valid_length_each_example = []
        for i in range(batch_size):
            # As the nonzero returns the index and we need length
            valid_length_each_example.append(np.max(np.argwhere(origin_inputs[i] != pad_token_id)) + 1)
        valid_length_each_example = np.array(valid_length_each_example)
        if np.max(valid_length_each_example) > max_length:
            raise ValueError("The max_length set is smaller than the length in the input_ids. You shout set "
                             f"max_length to {np.max(valid_length_each_example)}")
        target_length = self.config.seq_length if max_length > self.config.seq_length else max_length
        # A list of the frequency of each token
        frequency_list = None
        input_ids = self._pad_inputs_using_max_length(origin_inputs=origin_inputs, pad_token_id=pad_token_id)

        # for GLM `attention_mask` and `position_ids` generation
        attention_mask = self.get_masks_np(input_ids)
        position_ids = self.create_position_ids_np(input_ids)

        input_mask = np.zeros_like(input_ids)
        for i in range(valid_length_each_example.shape[0]):
            input_mask[i, :valid_length_each_example[i]] = 1

        # A single loop generates one token, loop until reaching target model_origin_max_length or generating eod token
        is_finished = [False] * batch_size

        # setup is_first_iteration flag for incremental infer
        if self.config.use_past:
            self.is_first_iteration = True

        is_first_iteration = False
        while np.sum(is_finished) != batch_size:
            # for GLM generation
            # model basic setting
            self.top_p = top_p
            self.top_k = top_k
            self.repetition_penalty = repetition_penalty

            seq_length = input_ids.shape[1]
            current_index = [valid_length_each_example[i] - 1 + i * seq_length for i in range(batch_size)]

            if self.config.use_past:
                is_first_iteration = self.is_first_iteration
                res = self._incremental_infer(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    current_index=current_index,
                    valid_length_each_example=valid_length_each_example
                )
            else:
                res = self(
                    input_ids=Tensor(input_ids, mstype.int32),
                    position_ids=Tensor(position_ids, mstype.int32),
                    attention_mask=Tensor(attention_mask, mstype.float32)
                )
            if is_npu_acceleration:
                p, p_args = res
                p = p.asnumpy()
                p_args = p_args.asnumpy()
                # Avoid rounding error
                p = precision_correct(p, top_p, top_k, batch_size)
            else:
                log_probs = self.process_logits(res, Tensor(current_index, mstype.int32),
                                                is_first_iteration, self.config.use_past)
                # Sample
                log_probs = log_probs.asnumpy()
                vocab_size = log_probs.shape[-1]
                if repetition_penalty != 1 and frequency_list is None:
                    frequency_list = np.array([[0 for _ in range(vocab_size)]])
                log_probs_revised = log_probs.reshape(batch_size, vocab_size)
                if repetition_penalty != 1:
                    log_probs_revised = log_probs - frequency_list * repetition_penalty - \
                                        (frequency_list > 0) * repetition_penalty
                p, p_args = sampler(log_probs_revised, top_p, top_k, use_pynative)

            # Random select a token as final output for this round
            for i in range(batch_size):
                if is_finished[i]:
                    continue
                target_index = np.random.choice(len(p[i]), p=p[i])

                # update frequency list
                target = p_args[i][target_index]

                if repetition_penalty != 1:
                    frequency_list[0][target] = frequency_list[0][target] + 1
                input_ids[i, valid_length_each_example[i]] = p_args[i, target_index]

                if streamer is not None:
                    streamer.put(np.asarray([target]))

                valid_length_each_example[i] += int(1)
                input_mask[i][valid_length_each_example[i] - 1] = 1

                # Stop judgment
                if p_args[i][target_index] == eos_token_id or valid_length_each_example[i] == target_length:
                    is_finished[i] = True
                    continue

        # Return valid outputs out of padded outputs
        output_ids = []
        for i in range(batch_size):
            output_ids.append(input_ids[i, : int(valid_length_each_example[i])].astype(np.int32))
        if streamer is not None:
            streamer.end()
        return output_ids

    # pylint: disable=W0613
    def construct(self, input_ids, label=None, position_ids=None, attention_mask=None,
                  input_position=None, init_reset=True, batch_valid_length=None):
        """
        Extract logits and calculate loss

        Inputs:
            input_ids (Tensor): The tokenized inputs with dtype int32.
            label (Tensor): The indices of input sequence tokens in the vocabulary.
            position_ids (Tensor): Used to identify each token's position in the list of tokens.
            attention_mask (Tensor): Used when batching sequences together.
            init_reset (bool, optional): Default: True.
            batch_valid_length(Tensor, optional): Default: None.

        Returns:
            Training phase:
                loss: Training loss.
            Other phase:
                logits (Tensor): The output logit of backbone.
        """
        batch_size, seq_length = input_ids.shape

        if self.phase == "train":
            tokens = self.stridedslice(input_ids, (0, 0), (batch_size, seq_length), (1, 1))
        else:
            tokens = input_ids

        output_states, _ = self.transformer(tokens, position_ids,
                                            attention_mask, init_reset, batch_valid_length)
        logits = self.lm_head(output_states)

        if self.phase != 'train':
            return logits

        logits_shape = logits.shape
        label = label.reshape((-1,))
        logits = logits.reshape((-1, logits_shape[-1]))
        input_mask = self.ones(tokens.shape, logits.dtype)
        input_mask = input_mask.reshape((-1,))
        loss = self.loss(logits, label, input_mask)
        return loss


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class GLMChatModel(GLMForPreTraining):
    r"""
    Provide glm chat capability through network.
    Args:
        config (GLMConfig): The config of GLMModel.

    Returns:
        Tensor, the probability distribution of network loss.
    """
    _support_list = MindFormerBook.get_model_support_list()['glm']

    def __init__(self, config: GLMConfig):
        super(GLMChatModel, self).__init__(config)
        self.e = ms.Tensor(np.e, dtype=mstype.float32)
        self.pow = P.Pow()
        self.topk = P.TopK(sorted=True)
        self.cumsum = P.CumSum()
        if is_version_ge(ms.__version__, '2.0.0'):
            self.sum = ops.sum
        else:
            self.sum = P.ReduceSum(keep_dims=False)
        self.vocab_size = config.vocab_size
        self.batch_size = config.batch_size
        self.frequency_list = ms.Tensor([[0 for _ in range(self.vocab_size)]])
        self.post_logits = ProcessLogits(use_past=config.use_past)
        # seems not supported yet.
        # self.top_p = config.top_p
        self.top_p = 1
        self.top_k = config.top_k
        self.repetition_penalty = config.repetition_penalty
        self.is_first_iteration = False
        self.is_npu_acceleration = config.is_npu_acceleration

    def sample(self, log_probs):
        """Convert the log_probs to probability"""

        if self.repetition_penalty != 1:
            log_probs = log_probs - self.frequency_list * self.repetition_penalty - \
                        (self.frequency_list > 0) * self.repetition_penalty

        # Process sample in graph to accelerate generate
        logits = self.pow(self.e, log_probs)

        # If top_p is less than 1.0, use top_p sampling
        # seems not supported yet.
        if self.top_p < 1.0:
            sorted_logits, index = self.topk(logits, 5000)
            cumsum_logits = self.cumsum(sorted_logits, 1)
            top_p_num = self.sum((cumsum_logits < self.top_p).astype(mstype.int32), -1) + 1
            top_p_num = int(top_p_num)
            # Get the corresponding probs and indices
            probs = sorted_logits[:, :top_p_num]
            p_args = index[:, :top_p_num]
            p = probs / self.sum(probs, -1, keepdim=True)

        # if top_p is set to 1.0, use top_k sampling
        else:
            probs, p_args = self.topk(logits, self.top_k)
            p = probs

        return p, p_args

    # pylint:disable=arguments-differ
    def construct(self, input_ids, position_ids=None, attention_mask=None,
                  input_position=None, init_reset=True, batch_valid_length=None):
        """Get probs and p_args"""
        # model forward
        output_states, _ = self.transformer(input_ids, position_ids, attention_mask, init_reset, batch_valid_length)
        logits = self.lm_head(output_states)

        if not self.is_npu_acceleration:
            return logits

        # logit post process
        log_probs = self.post_logits(logits, input_position, self.is_first_iteration)

        # logit sort and sample
        probs, p_args = self.sample(log_probs)

        return probs, p_args


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class GLMForPreTrainingWithLora(GLMForPreTraining):
    """GLM Model for pretraining with LoRA

    Args:
        config (GLMConfig): The config of network.
    """

    def __init__(self, config: GLMConfig = None, pet=None, **kwargs):
        _ = kwargs
        super().__init__(config)
        # get Pet tuning model.
        self.pet = pet
        self.pet.pet_config.reg_rules = r'.*query_key_value*'
        self.transformer = LoraAdapter.get_pet_model(self.transformer, self.pet.pet_config)
        # freeze pretrained model
        PetAdapter.freeze_pretrained_model(self, self.pet.pet_type)


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class GLMChatModelWithLora(GLMChatModel):
    """GLM Model for pretraining with LoRA

    Args:
        config (GLMConfig): The config of network.
    """

    def __init__(self, config: GLMConfig = None, pet=None, **kwargs):
        _ = kwargs
        ckpt_cfg = config.checkpoint_name_or_path
        config.checkpoint_name_or_path = None
        super().__init__(config)
        # get Pet tuning model.
        self.pet = pet
        self.pet.pet_config.reg_rules = r'.*query_key_value*'
        self.transformer = LoraAdapter.get_pet_model(self.transformer, self.pet.pet_config)
        config.checkpoint_name_or_path = ckpt_cfg
        self.load_checkpoint(config)
