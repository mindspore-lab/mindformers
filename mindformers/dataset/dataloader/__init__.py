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
from .flickr8k_dataloader import Flickr8kDataLoader
from .cifar100_dataloader import Cifar100DataLoader
from .wmt16_dataloader import WMT16DataLoader
from .cluener_dataloader import CLUENERDataLoader
from .squad_dataloader import SQuADDataLoader

__all__ = ['Flickr8kDataLoader', 'Cifar100DataLoader', 'WMT16DataLoader',
           'CLUENERDataLoader', 'SQuADDataLoader']
