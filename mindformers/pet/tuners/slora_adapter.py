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
"""Linear Layers with LoRA."""

import abc
import re
import json
import os
from typing import Tuple, Union

from mindspore import Parameter
from mindspore.common.initializer import initializer
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.nn import Cell
from mindspore._checkparam import args_type_check

from mindformers.modules.layers import Linear, Dropout
from mindformers.models.llama.llama_layer import LlamaEmbedding
from mindformers.pet.pet_config import SLoraConfig
from mindformers.tools.logger import logger


class SLoraLinear(Cell):
    """
    Decorator of Linear layer for S-LoRA
    """
    def __init__(self, linear: Linear, adapter_ids: Parameter, config: SLoraConfig):
        super().__init__()
        self.adapter_ids = adapter_ids
        self.in_channels = linear.in_channels
        self.out_channels = linear.out_channels
        self.expert_num = linear.expert_num
        self.dtype = linear.dtype
        self.expert_flag = linear.expert_flag
        self.use_gmm = linear.use_gmm
        self.use_expert_group_size = linear.use_expert_group_size
        self.expert_group_size = linear.expert_group_size
        self.outer_batch = linear.outer_batch
        self.has_bias = linear.has_bias
        self.activation_flag = linear.activation_flag
        self.weight = linear.weight
        self.matmul = linear.matmul
        if self.has_bias:
            self.bias = linear.bias
            self.bias_add = linear.bias_add
        if self.activation_flag:
            self.activation = linear.activation

        self.lora_num = config.lora_num
        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.lora_scaling = self.lora_alpha / self.lora_rank

        self.cast = P.Cast()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.lora_mul = P.Mul()
        self.lora_add = P.Add()
        self.lora_a_gather = P.Gather()
        self.lora_b_gather = P.Gather()
        self.lora_dropout = Dropout(keep_prob=1 - config.lora_dropout)
        self.lora_a_matmul = P.BatchMatMul(transpose_b=True)
        self.lora_b_matmul = P.BatchMatMul(transpose_b=True)
        self.lora_a_shape = [self.lora_num, self.lora_rank, self.in_channels]
        self.lora_b_shape = [self.lora_num, self.out_channels, self.lora_rank]
        self.lora_a = Parameter(initializer('zero', self.lora_a_shape, config.lora_dtype), requires_grad=False)
        self.lora_b = Parameter(initializer('zero', self.lora_b_shape, config.lora_dtype), requires_grad=False)


    def construct(self, x, expert_ids=None):
        """Forward process, x should be a tensor"""
        batch_size = self.adapter_ids.shape[0]
        out_shape = self.shape(x)[:-1] + (self.out_channels,)
        x = self.reshape(x, (-1, self.in_channels))
        if self.expert_flag and not self.use_gmm:
            if self.use_expert_group_size is True:
                x = self.reshape(x, (-1, self.expert_num, self.expert_group_size, self.in_channels))
            else:
                x = self.reshape(x, (self.outer_batch, self.expert_num, -1, self.in_channels))
        ori_dtype = F.dtype(x)
        weight = self.cast(self.weight, self.dtype)
        lora_a = self.cast(self.lora_a, self.dtype)
        lora_b = self.cast(self.lora_b, self.dtype)
        lora_a = self.lora_a_gather(lora_a, self.adapter_ids.reshape(-1), 0)
        lora_b = self.lora_b_gather(lora_b, self.adapter_ids.reshape(-1), 0)

        x = self.cast(x, self.dtype)
        if self.use_gmm:
            dense_result = self.matmul([x], [weight], None, None, None, None, None, expert_ids)[0]
        else:
            dense_result = self.matmul(x, weight)

        #-------- LoRA part ----------
        x = self.reshape(x, (batch_size, -1, self.in_channels))

        x = self.lora_dropout(x)
        lora_result = self.lora_a_matmul(x, lora_a)
        lora_result = self.lora_b_matmul(lora_result, lora_b)
        lora_scaling = self.cast(self.lora_scaling, self.dtype)
        lora_result = self.reshape(lora_result, dense_result.shape)
        lora_result = self.lora_mul(lora_result, lora_scaling)

        if self.has_bias:
            dense_result = self.bias_add(dense_result, self.cast(self.bias, self.dtype))
        # Result addition
        out = self.lora_add(dense_result, lora_result)
        if self.activation_flag:
            out = self.activation(out)
        out = self.cast(out, ori_dtype)
        output = self.reshape(out, out_shape)
        return output

    def shard(self):
        """
        Set the shard for the linear. the strategy size should be equal to the inputs.

        Note:
            It is valid only in semi auto parallel or auto parallel mode.
            In other parallel modes, strategies set here will be ignored.
        """
        strategy = self.matmul.in_strategy
        self.lora_a_gather.shard(((1, 1, strategy[1][1]), (1,)))
        self.lora_b_gather.shard(((1, strategy[1][0], 1), (1,)))
        self.lora_a_matmul.shard(((strategy[0][0], 1, strategy[0][1]), (strategy[0][0], 1, strategy[1][1])))
        self.lora_b_matmul.shard(((strategy[0][0], 1, 1), (strategy[0][0], strategy[1][0], 1)))
        self.lora_mul.shard(((strategy[0][0], strategy[1][0]), ()))
        self.lora_add.shard(((strategy[0][0], strategy[1][0]), (strategy[0][0], strategy[1][0])))


class SLoraEmbedding(Cell):
    r"""
        Decorator of Embedding layer for SLoRA.

        Args:
            embedding (Cell): The base embedding cell.
            adapter_ids (Patameter): The ids of the adapters.
            config (SLoraConfig): The config of SLora.

        Inputs:
            - **input_ids** (Tensor) - Tensor of shape :math:`(batch, seq_length)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """
    def __init__(self, embedding: Cell, adapter_ids: Parameter, config: SLoraConfig):
        super().__init__()
        self.adapter_ids = adapter_ids
        self.lora_extra_vocab_size = config.lora_extra_vocab_size
        self.lora_vocab_size = embedding.vocab_table_size + config.lora_extra_vocab_size
        self.dtype = embedding.embedding_weight.dtype
        self.embedding_weight = embedding.embedding_weight
        self.lora_scaling = config.lora_alpha / config.lora_rank
        self.embedding_size = embedding.embedding_size

        self.broadcast_to = P.BroadcastTo((config.lora_num, embedding.vocab_table_size, self.embedding_size))
        self.concat = P.Concat(1)
        self.transpose = P.Transpose()
        self.cast = P.Cast()
        self.lora_mul = P.Mul()
        self.lora_add = P.Add()
        self.gather = embedding.gather
        self.lora_a_gather = P.Gather()
        self.lora_b_gather = P.Gather()
        self.embedding_gather = P.Gather()
        self.lora_gather = P.Gather(1)
        self.reshape = P.Reshape()
        self.lora_matmul = P.BatchMatMul(transpose_b=True)

        self.lora_a_shape = [config.lora_num, config.lora_rank, self.lora_vocab_size]
        self.lora_b_shape = [config.lora_num, embedding.embedding_size, config.lora_rank]
        self.lora_a = Parameter(initializer('zero', self.lora_a_shape, config.lora_dtype), requires_grad=False)
        self.lora_b = Parameter(initializer('zero', self.lora_b_shape, config.lora_dtype), requires_grad=False)

        if self.lora_extra_vocab_size:
            self.lora_embedding_shape = [config.lora_num, self.lora_extra_vocab_size, self.embedding_size]
            self.lora_embedding = Parameter(initializer('zero', self.lora_embedding_shape, config.lora_dtype),
                                            requires_grad=False)

    def construct(self, input_ids):
        """Forward process"""
        embedding_weight = self.cast(self.embedding_weight, self.dtype)
        lora_a = self.cast(self.lora_a, self.dtype)
        lora_b = self.cast(self.lora_b, self.dtype)
        lora_a = self.lora_a_gather(lora_a, self.adapter_ids.reshape(-1), 0)
        lora_b = self.lora_b_gather(lora_b, self.adapter_ids.reshape(-1), 0)

        #-------- Embedding part ----------
        if self.lora_extra_vocab_size != 0:
            lora_embedding = self.cast(self.lora_embedding, self.dtype)
            embedding_weight = self.broadcast_to(embedding_weight)
            embedding_weight = self.concat((embedding_weight, lora_embedding))
            embedding_weight = self.gather(embedding_weight, self.adapter_ids, 0)
            embedding_result = self.lora_gather(embedding_weight, input_ids, 1)
        else:
            embedding_result = self.gather(embedding_weight, input_ids, 0)

        #-------- LoRA part ----------
        lora_result = self.lora_gather(self.transpose(lora_a, (0, 2, 1)), input_ids, 1)
        lora_result = self.lora_matmul(lora_result, lora_b)
        lora_scaling = self.cast(self.lora_scaling, self.dtype)
        lora_result = self.reshape(lora_result, embedding_result.shape)
        lora_result = self.lora_mul(lora_result, lora_scaling)

        out = self.lora_add(embedding_result, lora_result)
        out = self.cast(out, self.dtype)
        output = self.reshape(out, (-1, out.shape[-1]))

        return output

    def shard(self):
        """
        Set the shard for the SLoRAEbedding. the strategy size should be equal to the inputs.

        Note:
            It is valid only in semi auto parallel or auto parallel mode.
            In other parallel modes, strategies set here will be ignored.
        """
        strategy = self.gather.in_strategy
        self.lora_mul.shard(((strategy[0][0], strategy[1][0]), ()))
        self.lora_add.shard(((strategy[0][0], strategy[1][0]), (strategy[0][0], strategy[1][0])))


class SLoraHead(Cell):
    r"""
        Decorator of Head layer for SLoRA.

        Args:
            linear (Linear): The base head cell.
            adapter_ids (Patameter): The ids of the adapters.
            config (SLoraConfig): The config of SLora.

        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq_length, hidden_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, lora_vocab_size)`.
    """
    def __init__(self, linear: Linear, adapter_ids: Parameter, config: SLoraConfig):
        super().__init__()
        self.adapter_ids = adapter_ids
        self.lora_num = config.lora_num
        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.lora_scaling = self.lora_alpha / self.lora_rank
        self.in_channels = linear.in_channels
        self.out_channels = linear.out_channels
        self.lora_extra_vocab_size = config.lora_extra_vocab_size
        self.lora_vocab_size = self.out_channels + self.lora_extra_vocab_size

        self.expert_flag = linear.expert_flag
        self.use_gmm = linear.use_gmm
        self.dtype = linear.dtype
        self.use_expert_group_size = linear.use_expert_group_size
        self.expert_group_size = linear.expert_group_size
        self.expert_num = linear.expert_num
        self.shape = linear.shape
        self.weight = linear.weight
        self.outer_batch = linear.outer_batch
        self.has_bias = linear.has_bias
        self.activation_flag = linear.activation_flag
        if self.has_bias:
            self.bias = linear.bias
            self.bias_add = linear.bias_add
        if self.activation_flag:
            self.activation = linear.activation
        self.matmul = linear.matmul

        self.lora_mul = P.Mul()
        self.lora_add = P.Add()
        self.gather = P.Gather()
        self.cast = P.Cast()
        self.reshape = P.Reshape()

        self.lora_dropout = Dropout(keep_prob=1 - config.lora_dropout)
        self.lora_concat = P.Concat(axis=-1)
        self.lora_embedding_matmul = P.BatchMatMul(transpose_b=True)
        self.lora_a_matmul = P.BatchMatMul(transpose_b=True)
        self.lora_b_matmul = P.BatchMatMul(transpose_b=True)

        if self.lora_extra_vocab_size != 0:
            self.lora_embedding_shape = [self.lora_num, self.lora_extra_vocab_size, self.in_channels]
            self.lora_embedding = Parameter(initializer('zero', self.lora_embedding_shape, config.lora_dtype),
                                            name="lora_embedding", requires_grad=False)
        self.lora_a_shape = [self.lora_num, self.lora_rank, self.in_channels]
        self.lora_b_shape = [self.lora_num, self.lora_vocab_size, self.lora_rank]

        self.lora_a = Parameter(initializer('zero', self.lora_a_shape, config.lora_dtype), requires_grad=False)
        self.lora_b = Parameter(initializer('zero', self.lora_b_shape, config.lora_dtype), requires_grad=False)

    def construct(self, x, expert_ids=None):
        """Forward process, x should be a tensor"""
        batch_size = self.adapter_ids.shape[0]
        out_shape = self.shape(x)[:-1] + (self.lora_vocab_size,)
        x = self.reshape(x, (-1, self.in_channels))
        if self.expert_flag and not self.use_gmm:
            if self.use_expert_group_size is True:
                x = self.reshape(x, (-1, self.expert_num, self.expert_group_size, self.in_channels))
            else:
                x = self.reshape(x, (self.outer_batch, self.expert_num, -1, self.in_channels))

        ori_dtype = F.dtype(x)
        weight = self.cast(self.weight, self.dtype)
        if self.lora_extra_vocab_size != 0:
            lora_embedding = self.cast(self.lora_embedding, self.dtype)
            lora_embedding = self.gather(lora_embedding, self.adapter_ids.reshape(-1), 0)
        lora_a = self.cast(self.lora_a, self.dtype)
        lora_b = self.cast(self.lora_b, self.dtype)
        lora_a = self.gather(lora_a, self.adapter_ids.reshape(-1), 0)
        lora_b = self.gather(lora_b, self.adapter_ids.reshape(-1), 0)

        x = self.cast(x, self.dtype)
        if self.use_gmm:
            dense_result = self.matmul([x], [weight], None, None, None, None, None, expert_ids)[0]
        else:
            dense_result = self.matmul(x, weight)

        # -------- LoRA Embedding part ----------
        x = self.reshape(x, (batch_size, -1, self.in_channels))
        if self.lora_extra_vocab_size != 0:
            lora_logit = self.lora_embedding_matmul(x, lora_embedding)
            lora_logit = self.reshape(lora_logit, dense_result.shape[:-1] + (self.lora_extra_vocab_size,))

        # -------- LoRA part ----------
        x = self.lora_dropout(x)
        lora_result = self.lora_a_matmul(x, lora_a)
        lora_result = self.lora_b_matmul(lora_result, lora_b)
        lora_scaling = self.cast(self.lora_scaling, self.dtype)
        lora_result = self.reshape(lora_result, dense_result.shape[:-1] + (self.lora_vocab_size,))
        lora_result = self.lora_mul(lora_result, lora_scaling)

        if self.has_bias:
            dense_result = self.bias_add(dense_result, self.cast(self.bias, self.dtype))
        # Linear supplementation
        if self.lora_extra_vocab_size != 0:
            out = self.lora_concat([dense_result, lora_logit])
            out = self.lora_add(out, lora_result)
        else:
            out = self.lora_add(dense_result, lora_result)
        if self.activation_flag:
            out = self.activation(out)
        out = self.cast(out, ori_dtype)
        output = self.reshape(out, out_shape)
        return output

    def shard(self):
        """
        Set the shard for the SLoRAHead. the strategy size should be equal to the inputs.

        Note:
            It is valid only in semi auto parallel or auto parallel mode.
            In other parallel modes, strategies set here will be ignored.
        """
        strategy = self.matmul.in_strategy
        self.lora_a_matmul.shard(((strategy[0][0], 1, strategy[0][1]), (strategy[0][0], 1, strategy[1][1])))
        self.lora_b_matmul.shard(((strategy[0][0], 1, 1), (strategy[0][0], strategy[1][0], 1)))
        self.lora_mul.shard(((strategy[0][0], strategy[1][0]), ()))
        self.lora_add.shard(((strategy[0][0], strategy[1][0]), (strategy[0][0], strategy[1][0])))


class SLoraAdapter(abc.ABC):
    """
    LoRA Model Manager to wrap layers
    """
    adapter_names = []
    registered_loras = {}

    def __init__(self, config: SLoraConfig, adapter_ids: Parameter):
        self.adapter_ids = adapter_ids
        self.common_lora_cell: dict = {}
        self.slora_config = config

    @classmethod
    @args_type_check(pet_config=(dict, SLoraConfig))
    def init_slora_config(cls, pet_config: Union[dict, SLoraConfig]):
        """read S-LoRA config"""
        if isinstance(pet_config, SLoraConfig):
            return pet_config
        if not isinstance(pet_config.adapter_path, str):
            raise TypeError(f"adapter_path should be string type, but get {type(pet_config.adapter_path)}.")
        adapter_path = pet_config.adapter_path
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"The adapter_path must be correct, but get {adapter_path}")
        with open(adapter_path, 'r') as file:
            path_dict = json.load(file)
        cls.adapter_names = list(path_dict.keys())

        num = len(path_dict) + 1
        max_rank = 0
        target_modules_set = set()

        lora_extra_vocab_size = 0
        for _, slora_path in path_dict.items():
            added_tokens_path = os.path.join(slora_path, "added_tokens.json")
            if os.path.exists(slora_path):
                with open(added_tokens_path, 'r') as file:
                    added_tokens = json.load(file)
                lora_extra_vocab_size = len(added_tokens)
                break

        for _, slora_path in path_dict.items():
            config_path = os.path.join(slora_path, "adapter_config.json")
            if not os.path.exists(config_path):
                raise ValueError(f"{config_path} does not exist, "
                                 f"please pass a valid the slora path in config.adapter_path file.")
            with open(config_path, 'r') as file:
                config = json.load(file)
            if not isinstance(config["r"], int):
                raise TypeError(f"rank should be int type, but get {type(config['r'])} type.")
            max_rank = max(max_rank, int(config["r"]))
            if not all(isinstance(module, str) for module in config["target_modules"]):
                raise TypeError(f"target_modules should be string type, but get wrong type.")
            target_modules_set.update(config["target_modules"])
        target_modules_list = [".*" + module for module in list(target_modules_set)]
        target_modules = '|'.join(target_modules_list)

        # read alpha, dropout from the first lora.
        config_path = os.path.join(path_dict[list(path_dict.keys())[0]], "adapter_config.json")
        with open(config_path, 'r') as file:
            config = json.load(file)
        if not isinstance(config["lora_alpha"], int):
            raise TypeError(f"lora_alpha should be int type, but get {type(config['lora_alpha'])}.")
        alpha = int(config["lora_alpha"])
        if not isinstance(config["lora_dropout"], float):
            raise TypeError(f"lora_dropout should be float type, but get {type(config['lora_dropout'])}.")
        dropout = float(config["lora_dropout"])

        return SLoraConfig(target_modules, num, max_rank, alpha, dropout, lora_extra_vocab_size)

    def register(self, name: str, cell: Cell):
        if isinstance(cell, SLoraEmbedding) and cell.lora_extra_vocab_size != 0:
            self.registered_loras['model.tok_embeddings.lora_embedding'] = cell.lora_embedding_shape
        lora_a_name = name + ".lora_a"
        lora_b_name = name + ".lora_b"
        self.registered_loras[lora_a_name] = cell.lora_a_shape
        self.registered_loras[lora_b_name] = cell.lora_b_shape

    def build(self):
        self.common_lora_cell[Linear] = SLoraLinear
        self.common_lora_cell[LlamaEmbedding] = SLoraEmbedding

    def get_pet_model(self, network: Cell):
        """Replace layers"""
        if "lm_head" in self.slora_config.target_modules:
            if hasattr(network, "lm_head"):
                new_head = SLoraHead(network.lm_head, self.adapter_ids, self.slora_config)
                new_head.shard()
                network.lm_head = new_head
                if isinstance(network.lm_head, SLoraEmbedding) and cell.lora_extra_vocab_size != 0:
                    self.registered_loras['lm_head.lora_embedding'] = network.lm_head.lora_embedding_shape
                self.registered_loras["lm_head.lora_a"] = network.lm_head.lora_a_shape
                self.registered_loras["lm_head.lora_b"] = network.lm_head.lora_b_shape
                if network.lm_head.lora_extra_vocab_size != 0:
                    self.registered_loras["lm_head.lora_embedding"] = network.lm_head.lora_embedding_shape
                logger.info(f"Apply LoRA to lm_head.")
            else:
                logger.warning("The base model must has an attribute named in \'lm_head\'.")
        self.build()
        self.process(network.model)
        network.model.update_parameters_name()
        return network.model

    def process(self, root: Cell, name_prefix: str = "model"):
        """Recursively search for target layers in the network and wrap them"""
        if root is None:
            return root
        for name, cell in root.name_cells().items():
            full_cell_name = f"{name_prefix}.{name}"
            new_cell, is_end_point = self.process_cell(full_cell_name, cell)
            if new_cell is not cell:
                root.insert_child_to_cell(name, new_cell)
                self.register(full_cell_name, new_cell)
            if not is_end_point:
                _ = self.process(new_cell, full_cell_name)
        return root

    def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
        """Match the layer with target list to get the layer with lora"""
        if not isinstance(cell, Cell):
            return cell, True
        if re.match(self.slora_config.target_modules, cell_name) or \
            (re.match('.*embeddings', cell_name) and ("embed_tokens" in self.slora_config.target_modules)):
            wrap_lora = self.common_lora_cell.get(type(cell))
            if wrap_lora:
                logger.info(f"Apply LoRA to {cell_name}.")
                new_cell = wrap_lora(cell, self.adapter_ids, self.slora_config)
                new_cell.shard()
                return new_cell, True
        return cell, False
