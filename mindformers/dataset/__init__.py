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
"""MindFormers Dataset."""
from .dataloader import (
    ADGenDataLoader,
    BaseMultiModalDataLoader,
    BlendedMegatronDatasetDataLoader,
    CommonDataLoader,
    IndexedDataLoader,
    MultiSourceDataLoader,
    SFTDataLoader,
    ToolAlpacaDataLoader,
    TrainingDataLoader,
)
from .transforms import (
    BCHW2BHWC,
    BatchCenterCrop,
    BatchNormalize,
    BatchPILize,
    BatchResize,
    BatchToTensor,
    RandomCropDecodeResize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
)
from .handler import (
    AdgenInstructDataHandler,
    AlpacaInstructDataHandler,
    CodeAlpacaInstructDataHandler,
    LlavaInstructDataHandler,
    build_data_handler
)
from .build_dataset import build_dataset
from .dataloader.build_dataloader import build_dataset_loader
from .mask.build_mask import build_mask
from .sampler.build_sampler import build_sampler
from .transforms.build_transforms import build_transforms
from .base_dataset import BaseDataset
from .causal_language_model_dataset import CausalLanguageModelDataset
from .keyword_gen_dataset import KeyWordGenDataset
from .multi_turn_dataset import MultiTurnDataset
from .general_dataset import GeneralDataset
from .utils import (
    check_dataset_config,
    check_dataset_iterable
)
from .modal_to_text_sft_dataset import ModalToTextSFTDataset

__all__ = [
    'CausalLanguageModelDataset', 'KeyWordGenDataset', 'MultiTurnDataset',
]
__all__.extend(dataloader.__all__)
