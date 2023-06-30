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
"""Llama Config API."""


from mindformers.modules.transformer.transformer import default_transformer_config, TransformerOpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from ..utils import convert_mstype
from ..base_config import BaseConfig
from ...mindformer_book import MindFormerBook

__all__ = ['LlamaConfig']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class LlamaConfig(BaseConfig):
    """
    LLaMA config class which defines the model size.
    """

    _support_list = MindFormerBook.get_config_support_list()['llama']

    def __init__(self,
                 batch_size: int = None,
                 seq_length: int = 2048,
                 hidden_size: int = 4096,
                 num_layers: int = 32,
                 num_heads: int = 32,
                 vocab_size: int = 32000,   # defined later by tokenizer
                 multiple_of: int = 256,    # make SwiGLU hidden layer size multiple of large power of 2
                 rms_norm_eps: float = 1e-5,
                 bos_token_id: int = 1,
                 eos_token_id: int = 2,
                 pad_token_id: int = 32000,
                 ignore_token_id: int = -100,
                 compute_dtype: str = "float16",
                 layernorm_compute_type: str = "float32",
                 softmax_compute_type: str = "float32",
                 param_init_type: str = "float16",
                 parallel_config: TransformerOpParallelConfig = default_transformer_config,
                 use_past: bool = False,
                 offset: int = 0,
                 checkpoint_name_or_path: str = "",
                 repetition_penalty: float = 1.0,
                 max_decode_length: int = 1024,
                 top_k: int = 5,
                 top_p: float = 1.0,
                 do_sample: bool = True,
                 **kwargs):
        super(LlamaConfig, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.multiple_of = multiple_of
        self.rms_norm_eps = rms_norm_eps
        self.param_init_type = convert_mstype(param_init_type)
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.parallel_config = parallel_config
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.ignore_token_id = ignore_token_id
        self.use_past = use_past
        self.offset = offset
        self.repetition_penalty = repetition_penalty
        self.max_decode_length = max_decode_length
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
