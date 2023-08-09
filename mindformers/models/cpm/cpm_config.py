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
"""GLM config"""
from typing_extensions import TypedDict
from typing import Optional, Tuple, List
from mindspore import Tensor
from mindspore.common import dtype as mstype
from ...mindformer_book import MindFormerBook
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from ..base_config import BaseConfig


__all__ = ['CPMBeeConfig']


class CPMBeeInferenceState(TypedDict):
    buffer_position: Tensor
    buffer_context: Tensor
    buffer_sample_ids: Tensor
    buffer_num_segments: Tensor
    buffer_segments: Tensor
    buffer: List[Tuple[Tensor, Tensor]]


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class CPMBeeConfig(BaseConfig):

    _support_list = MindFormerBook.get_config_support_list()['cpm']

    def __init__(
            self,
            vocab_size=30720,
            dim_model=4096,
            num_heads=64,
            dim_head=64,
            dim_ff=10240,
            num_layers=32,
            dropout_p=0.0,
            position_bias_num_buckets=256,
            position_bias_num_segment_buckets=256,
            position_bias_max_distance=2048,
            eps=1e-6,
            half: bool = True,
            mask_modules: Optional[List[Tuple[bool, bool]]] = None,
            **kwargs
    ):

        super().__init__(**kwargs)
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.position_bias_num_buckets = position_bias_num_buckets
        self.position_bias_num_segment_buckets = position_bias_num_segment_buckets
        self.position_bias_max_distance = position_bias_max_distance
        self.dropout_p = dropout_p
        self.eps = eps
        if half:
            self.dtype = mstype.half
        else:
            self.dtype = mstype.single
        self.vocab_size = vocab_size
        self.mask_modules = mask_modules
