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
"""MindFormers DataLoader."""
from .build_dataloader import build_dataset_loader
from .common_dataloader import CommonDataLoader
from .multi_source_dataloader import MultiSourceDataLoader
from .adgen_dataloader import ADGenDataLoader
from .sft_dataloader import SFTDataLoader
from .training_dataloader import TrainingDataLoader
from .toolaplaca_dataloader import ToolAlpacaDataLoader
from .multi_modal_dataloader import BaseMultiModalDataLoader
from .indexed_dataset import IndexedDataLoader
from .blended_megatron_dataloader import BlendedMegatronDatasetDataLoader

__all__ = ['BlendedMegatronDatasetDataLoader']
