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
"""Configuration."""
import os
from typing import Optional
from collections import OrderedDict
import yaml

try:
    from mindspore._checkparam import Validator
    from mindspore._checkparam import Rel
except ImportError:
    import mindspore._checkparam as Validator
    import mindspore._checkparam as Rel

from mindformers.experimental.distri_cores.utils import convert_mstype


_SUPPORT_DTYPE = ("float32", "float16", "bfloat16")


def load_yaml(stream, yaml_loader=yaml.SafeLoader,
              object_pairs_hook=OrderedDict):
    """ Load yaml file in orderedly. """
    class OrderedLoader(yaml_loader):
        pass

    def _construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        _construct_mapping)
    return yaml.load(stream, OrderedLoader)


class BaseConfig(dict):
    """ Base config class. """
    def __init__(self, **kwargs):
        super(BaseConfig, self).__init__()
        BaseConfig._dict2config(self, kwargs)

    def __getattr__(self, key):
        """ __getattr__ override. """
        if key not in self:
            raise ValueError("Config has no attribute {}".format(key))
        return self[key]

    def __setattr__(self, key, value):
        """ __setattr__ override. """
        self[key] = value

    def __delattr__(self, key):
        """ __delattr__ override. """
        del self[key]

    @staticmethod
    def _dict2config(config, dic):
        r"""
        Convert dictionary to config.

        Args:
            config : Config object.
            dic (dict) : Keyword arguments passed by dictionary.
        """
        if isinstance(dic, dict):
            for key, value in dic.items():
                if isinstance(value, dict):
                    sub_config = BaseConfig()
                    dict.__setitem__(config, key, sub_config)
                    BaseConfig._dict2config(sub_config, value)
                else:
                    config[key] = dic[key]


def _config_to_str(cls):
    """ Return class attribute str for print. """
    attributes = vars(cls)
    print_str = cls.__class__.__name__ + "\n"
    for name, val in attributes.items():
        new_str = str(val)
        if "\t" in new_str:
            new_str = new_str.replace("\t", "\t\t")
        print_str += "\t{}: {}\n".format(name, new_str)

    return print_str


class TrainingConfig:
    r"""
    Training config.

    Args:
        epochs (int): Epochs number for training. Default: 1.
        batch_size (int): Batch size for training. Default: 1.
    """
    def __init__(self,
                 epochs: int = 1,
                 batch_size: int = 1,
                 **kwargs):
        super(TrainingConfig, self).__init__()
        Validator.check_positive_int(epochs, "epochs")
        Validator.check_positive_int(batch_size, "batch_size")
        self.epochs = epochs
        self.batch_size = batch_size
        self.__dict__.update(kwargs)

    def __str__(self):
        return _config_to_str(self)


class DatasetConfig:
    r"""
    Dataset config class.

    Args:
        dataset_dir (str): Dataset file directory. Default: './dataset'.
        shuffle (Optional[bool], None): Shuffle dataset. Default: None.
    """
    def __init__(self,
                 dataset_dir: str = "./dataset",
                 shuffle: Optional[bool] = None,
                 **kwargs):
        super(DatasetConfig, self).__init__()
        Validator.check_value_type("dataset_dir", dataset_dir, [str])
        Validator.check_value_type("shuffle", shuffle, [bool, type(None)])
        self.dataset_dir = dataset_dir
        self.shuffle = shuffle
        self.__dict__.update(kwargs)

    def __str__(self):
        return _config_to_str(self)


class ParallelConfig:
    r"""
    Parallel config class.

    Args:
        tensor_parallel (int): Dimensionality of tensor parallel. Default: 1.
        pipeline_stage (int): Number of stages when using pipeline parallel. Default: 1.
        context_parallel (int): Dimensionality of context parallel. Default: 1.
        expert_parallel (int): Dimensionality of expert parallel. Default: 1.
        micro_batch_num (int): Number of micro batch when using pipeline parallel. Default: 1.
        use_sequence_parallel (bool): Enable sequence parallel. Default: False.
        recv_dtype (bool): Communication data type of p2p communication when using pipeline
            parallel. Default: 'float32'.
    """
    def __init__(self,
                 tensor_parallel: int = 1,
                 pipeline_stage: int = 1,
                 context_parallel: int = 1,
                 expert_parallel: int = 1,
                 micro_batch_num: int = 1,
                 use_sequence_parallel: bool = False,
                 recv_dtype: str = "float32",
                 **kwargs):
        super(ParallelConfig, self).__init__()
        Validator.check_positive_int(tensor_parallel, "tensor_parallel")
        Validator.check_positive_int(pipeline_stage, "pipeline_stage")
        Validator.check_positive_int(context_parallel, "context_parallel")
        Validator.check_positive_int(expert_parallel, "expert_parallel")
        Validator.check_positive_int(micro_batch_num, "micro_batch_num")
        Validator.check_bool(use_sequence_parallel, "use_sequence_parallel")
        Validator.check_string(recv_dtype, _SUPPORT_DTYPE, "recv_dtype")
        self.tensor_parallel = tensor_parallel
        self.pipeline_stage = pipeline_stage
        self.context_parallel = context_parallel
        self.expert_parallel = expert_parallel
        self.micro_batch_num = micro_batch_num
        self.use_sequence_parallel = use_sequence_parallel
        self.recv_dtype = convert_mstype(recv_dtype)
        self.__dict__.update(kwargs)

    def __str__(self):
        return _config_to_str(self)


class ModelConfig:
    r"""
    Model config class.

    Args:
        vocab_size (int): Vocabulary size.
        num_layers (int): Number of model layers.
        num_heads (int): Number of heads for MultiHeadAttention.
        hidden_size (int): Dimensionality of the encoder layers.
        ffn_hidden_size (int): Dimensionality the FeedForward block project to.
        parallel_config (ParallelConfig): Parallel config.
        attention_type (str): Attention type. Default: 'self_attn'.
        use_gqa (bool): Enable group query attention. Default: False.
        kv_num_heads (int): Number of heads for key and value when using group query attention.
            Default: 32.
        qkv_has_bias (bool): Linears apply on query, key and value in Attention block has bias
            parameter. Default: True.
        out_proj_has_bias (bool): Linear applies on output of core attention block has bias
            parameter. Default: True.
        apply_query_key_layer_scaling (bool): Apply query key scaling in core attention block.
            Default: False.
        use_flash_attention (bool): Enable flash attention. Default: False.
        mask_func_type (str): Attention mask compute method. Default: 'attn_mask_add'.
        mlp_has_bis (bool): Linears in MLP block have bias parameters. Default: True.
        hidden_act (str): Activation used in MLP block. Default: 'gelu'.
        normalization (str): Normalization used in transformerlayer block. Default: 'LayerNorm'.
        layernorm_epsilon (float): Epsilon of normalization. Default: 1.e-5.
        apply_residual_connection_post_norm (bool): Apply residual connection after normalization.
            Default: False.
        residual_connection_dtype (str): Compute data type of residual connection. Default: 'float32'.
        param_init_dtype (str): Parameter initialize data type. Default: 'float32'.
        compute_dtype (str): Compute data type of linear module. Default: 'float16'.
        softmax_compute_dtype (str): Compute data type of softmax layer. Default: 'float32'.
        hidden_dropout_rate (float): Dropout rate for output of attention block and mlp block in transformerlayer.
            Default: 0.0.
        attention_dropout_rate (float): Dropout rate for attention socre. Default: 0.0.
        num_experts (Optional[int], None): Number of experts. Default: None.
        share_embedding_weight (bool): Share embedding table between embedding layer and llm head. Default: False.
    """
    def __init__(self,
                 vocab_size: int,
                 num_layers: int,
                 num_heads: int,
                 hidden_size: int,
                 ffn_hidden_size: int,
                 parallel_config: ParallelConfig,
                 attention_type: str = "self_attn",
                 use_gqa: bool = False,
                 kv_num_heads: int = 32,
                 qkv_has_bias: bool = True,
                 out_proj_has_bias: bool = True,
                 apply_query_key_layer_scaling: bool = False,
                 use_flash_attention: bool = False,
                 mask_func_type: str = "attn_mask_add",
                 mlp_has_bias: bool = True,
                 hidden_act: str = "gelu",
                 normalization: str = "LayerNorm",
                 layernorm_epsilon: float = 1.e-5,
                 apply_residual_connection_post_norm: bool = False,
                 residual_connection_dtype: str = "float32",
                 param_init_dtype: str = "float32",
                 compute_dtype: str = "float16",
                 softmax_compute_dtype: str = "float32",
                 hidden_dropout_rate: float = 0.0,
                 attention_dropout_rate: float = 0.0,
                 num_experts: Optional[int] = None,
                 share_embedding_weight: bool = False,
                 **kwargs):
        super(ModelConfig, self).__init__()
        Validator.check_positive_int(vocab_size, "vocab_size")
        Validator.check_positive_int(num_layers, "num_layers")
        Validator.check_positive_int(num_heads, "num_heads")
        Validator.check_positive_int(hidden_size, "hidden_size")
        Validator.check_positive_int(ffn_hidden_size, "ffn_hidden_size")
        Validator.check_value_type("attention_type", attention_type, [str])
        Validator.check_bool(use_gqa, "use_gqa")
        Validator.check_positive_int(kv_num_heads, "kv_num_heads")
        Validator.check_bool(qkv_has_bias, "qkv_has_bias")
        Validator.check_bool(out_proj_has_bias, "out_proj_has_bias")
        Validator.check_bool(apply_query_key_layer_scaling, "apply_query_key_layer_scaling")
        Validator.check_bool(use_flash_attention, "use_flash_attention")
        Validator.check_value_type("mask_func_type", mask_func_type, [str])
        Validator.check_bool(mlp_has_bias, "mlp_has_bias")
        Validator.check_value_type("hidden_act", hidden_act, [str])
        Validator.check_value_type("normalization", normalization, [str])
        Validator.check_positive_float(layernorm_epsilon, "layernorm_epsilon")
        Validator.check_bool(apply_residual_connection_post_norm,
                             "apply_residual_connection_post_norm")
        Validator.check_string(residual_connection_dtype, _SUPPORT_DTYPE, "recv_dtype")
        Validator.check_string(param_init_dtype, _SUPPORT_DTYPE, "param_init_dtype")
        Validator.check_string(compute_dtype, _SUPPORT_DTYPE, "compute_dtype")
        Validator.check_string(softmax_compute_dtype, _SUPPORT_DTYPE, "softmax_compute_dtype")
        Validator.check_float_range(hidden_dropout_rate, 0, 1, Rel.INC_BOTH, "hidden_dropout_rate")
        Validator.check_float_range(attention_dropout_rate, 0, 1, Rel.INC_BOTH, "attention_dropout_rate")
        Validator.check_value_type("num_experts", num_experts, [int, type(None)])
        Validator.check_bool(share_embedding_weight, "share_embedding_weight")
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.parallel_config = parallel_config
        self.attention_type = attention_type
        self.use_gqa = use_gqa
        self.kv_num_heads = kv_num_heads
        self.qkv_has_bias = qkv_has_bias
        self.out_proj_has_bias = out_proj_has_bias
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.use_flash_attention = use_flash_attention
        self.mask_func_type = mask_func_type
        self.mlp_has_bias = mlp_has_bias
        self.hidden_act = hidden_act
        self.normalization = normalization
        self.layernorm_epsilon = layernorm_epsilon
        self.apply_residual_connection_post_norm = apply_residual_connection_post_norm
        self.residual_connection_dtype = convert_mstype(residual_connection_dtype)
        self.param_init_dtype = convert_mstype(param_init_dtype)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.softmax_compute_dtype = convert_mstype(softmax_compute_dtype)
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.num_experts = num_experts
        self.share_embedding_weight = share_embedding_weight
        self.__dict__.update(kwargs)

    def __str__(self):
        return _config_to_str(self)


def set_base_config(config: dict):
    """ Set base running config. """
    config.setdefault("seed", 0)
    config.setdefault("output_dir", "./output")


def set_lr_scheduler_config(lr_scheduler_config: dict, learning_rate: float):
    """ Set learning rate scheduler config. """
    lr_scheduler_config.setdefault("type", None)
    lr_scheduler_config.learning_rate = learning_rate


def set_optimizer_config(optimizer_config: dict):
    """ Set optimizer config. """
    optimizer_config.setdefault("type", "AdamWeightDecay")


# pylint: disable=W0613
def init_arguments(file: str, **kwargs):
    """ Initialize config class from configuration yaml file. """
    if isinstance(file, str):
        if file.endswith('yaml') or file.endswith('yml'):
            filepath = os.path.realpath(file)
            with open(filepath, encoding='utf-8') as fp:
                raw_dict = load_yaml(fp, yaml_loader=yaml.FullLoader)
    config = BaseConfig(**raw_dict)
    config.training = TrainingConfig(**raw_dict["training"])
    print(config.training)
    config.train_dataset = DatasetConfig(**raw_dict["train_dataset"])
    print(config.train_dataset)
    set_optimizer_config(config.optimizer)
    if "lr_scheduler" in config:
        assert "learning_rate" in config.optimizer, "when use lr_scheduler, \
            learning rate of optimizer should be specified."
        set_lr_scheduler_config(config.lr_scheduler, config.optimizer.learning_rate)
    else:
        config.lr_scheduler = None
    config.parallel_config = ParallelConfig(**raw_dict["parallel_config"])
    print(config.parallel_config)
    config.model_config = ModelConfig(parallel_config=config.parallel_config, **raw_dict["model_config"])
    print(config.model_config)

    return config
