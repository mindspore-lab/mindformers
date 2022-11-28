# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Mae Config API."""
from dataclasses import dataclass

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
@dataclass
class MaeConfig:
    """Mae Config."""
    mask_ratio: float = 0.75
    num_classes: int = 0
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    drop_rate: float = 0.
    drop_path_rate: float = 0.
    use_abs_pos_emb: bool = True
    decoder_layers: int = 8
    decoder_num_heads: int = 16
    decoder_dim: int = 512
    norm_pixel_loss: bool = True
    parallel_config: dict = None
    moe_config: dict = None
