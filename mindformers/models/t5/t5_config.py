# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""T5 Configuration"""
import mindspore.common.dtype as mstype
from mindformers.modules.transformer.transformer import default_transformer_config, TransformerOpParallelConfig
from ..base_config import BaseConfig
from ...tools.register import MindFormerRegister, MindFormerModuleType
from ...mindformer_book import MindFormerBook

__all__ = ['T5Config']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class T5Config(BaseConfig):
    r"""
    Config For T5 model

    Args:
        vocab_size(int): The vocabuary size, it determines the size of embedding table and the final projection size.
        d_model(int): The hidden size of the model. Default 512.
        d_kv(int): The internal hidden size of the attention head. Default 64.
        d_ff(int): The intermediate size of the T5 FFN layer. Default 2048.
        num_layers(int): The layers of the encoder and decoder parts. Default 6.
        num_decoder_layers(int): The layers of the decoder. If not set, the value will
            copy from num_layers. Default None.
        num_heads(int): The number of attention heads. Default 8.
        relative_attention_num_buckets(int): The size of relative attention buckets. Default 32.
        dropout_rate(float): The dropout rate of the hidden state. Default 0.1.
        layer_norm_epsilon(float): The layernorm epsilon. Default 1e-6.
        initializer_factor(float): The factor of the weight initialization. Default 1.0.
        feed_froward_proj(str): The type of activation. Default `relu`.
        is_encoder_decoder(bool): Whether the current model is encdeor-decoder. Default True.
        use_cache(bool): If use cache or not for inference. Default True.
        pad_token_id(int): The pad token id. Default 0.
        eos_token_id(int): The end of sentence token id. Default 1.
        max_position_embeddings(int): The length of the position embedding. Default 1024.
        max_decode_length(int): The sequence length of the decoder part. Default 128.
        dtype(str): The initialization type of the parameters in T5 model. Default "float32".
        compute_dtype(str): The computation type of the dense layer in the T5 model. Default "float32".
        has_relative_bias(bool): Whether add relative attention bias. Default True.
        scale_output(bool): Whether to scale the output. Default True.
        seq_length(int): The sequence length of the encoder part. Default 1024.
        batch_size(int): The batch size of the model. Default 1.
        checkpoint_name_or_path(str): The path to the checkpoint. If set, it will load the checkpoint from the given
            path.
        top_p(float): Used in the `generate` method of the BaseModel. Default 0.95. The accumulation probability of
            the candidate token ids below the top_p will be select as the condaite ids. The validate the value of
            top_p is between (0, 1]. If the value is larger than 1,
        top_k(int): Used in the `generate` method of the BaseModel. Determine the topK numbers token id as candidate.
            This should be a positive number. Default 1.
        repetition_penalty(float): The penalty of the repeated words when call `generate` of the BaseModel.
        max_length(int): The maximum length of the generated words. If set None, it follow the setting in the
            configureation in the model. Default 20.
        do_sample(bool): Do sampling on the candidate ids. If set True it will be enabled, and set it to be
            False to disable the sampling, equivalent to topk 1. If set None, it follow the setting in the
            configureation in the model. Default False.
        **kwargs:

    Examples:
        >>> from mindformers import T5Config
        >>> T5Config(d_model=256, vocab_size=40000)
            {'d_model': 256, 'vocab_size': 40000,
             'max_position_embeddings': 77, 'num_layers': 12}
    """
    _support_list = MindFormerBook.get_config_support_list()['t5']

    def __init__(self,
                 vocab_size: int = 32128,
                 d_model: int = 512,
                 d_kv: int = 64,
                 d_ff: int = 2048,
                 num_layers: int = 6,
                 num_decoder_layers=None,
                 num_heads: int = 8,
                 relative_attention_num_buckets: int = 32,
                 dropout_rate: float = 0.1,
                 layer_norm_epsilon: float = 1e-6,
                 initializer_factor: float = 1.0,
                 feed_froward_proj: str = "relu",
                 is_encoder_decoder=True,
                 use_cache: bool = True,
                 pad_token_id: int = 0,
                 eos_token_id: int = 1,
                 # The following are mindformers parameters
                 batch_size: int = 1,
                 seq_length: int = 1024,
                 max_position_embeddings: int = 1024,
                 initializer_range: float = 0.02,
                 max_decode_length: int = 128,
                 length_penalty_weight: float = 1.0,
                 dtype: str = "float32",
                 compute_dtype: str = "float32",
                 has_relative_bias: bool = True,
                 scale_output: bool = True,
                 parallel_config: TransformerOpParallelConfig = None,
                 checkpoint_name_or_path: str = None,
                 top_p=0.95,
                 top_k=1,
                 repetition_penalty=1,
                 max_length=20,
                 do_sample=False,
                 **kwargs):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.hidden_act = feed_froward_proj
        self.kv_size = d_kv
        self.dropout_rate = dropout_rate
        self.initializer_factor = initializer_factor
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.max_decode_length = max_decode_length
        self.length_penalty_weight = length_penalty_weight
        self._dtype = dtype
        self._compute_dtype = compute_dtype
        self.has_relative_bias = has_relative_bias
        self.scale_output = scale_output
        self._parallel_config = parallel_config
        self.num_decoder_layers = num_decoder_layers
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_cache = use_cache
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.pad_token_id = pad_token_id
        # Basic the configuration for the generation
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.do_sample = do_sample

        super(T5Config, self).__init__(**kwargs)

    @property
    def dtype(self):
        return mstype.float32 if self._dtype == "float32" else mstype.float16

    @property
    def compute_dtype(self):
        return mstype.float32 if self._compute_dtype == "float32" else mstype.float16

    @property
    def parallel_config(self):
        if not self._parallel_config:
            return default_transformer_config
        return self._parallel_config
