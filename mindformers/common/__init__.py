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
"""MindFormers Common."""

from .lr import build_lr
from .loss import build_loss
from .optim import build_optim
from .metric import build_metric
from .parallel_config import build_parallel_config
from .callback import build_callback
from .context import build_context

__all__ = ['build_lr', 'build_loss', 'build_optim', 'build_metric', 'build_parallel_config', 'build_callback',
           'build_context']
