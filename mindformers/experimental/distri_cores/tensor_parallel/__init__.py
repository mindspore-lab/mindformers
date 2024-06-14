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

"""mindformers init"""

from .collective_primitives import all_to_all_hp2sp, all_to_all_sp2hp
from .collective_primitives import CopyToModelParallelRegion
from .collective_primitives import ScatterToModelParallelRegion
from .collective_primitives import GatherFromModelParallelRegion
from .collective_primitives import ReduceFromModelParallelRegion
from .collective_primitives import ReduceScatterToSequenceParallelRegion
from .collective_primitives import ReduceScatterToTensorParallelRegion
from .collective_primitives import ScatterToSequenceParallelRegion
from .collective_primitives import GatherFromSequenceParallelRegion
from .collective_primitives import AllGatherFromTensorParallelRegion
from .collective_primitives import AllToAll
from .layers import *

__all__ = [
    'all_to_all_hp2sp',
    'all_to_all_sp2hp',
    'CopyToModelParallelRegion',
    'ScatterToModelParallelRegion',
    'GatherFromModelParallelRegion',
    'ReduceFromModelParallelRegion',
    'ReduceScatterToSequenceParallelRegion',
    'ReduceScatterToTensorParallelRegion',
    'ScatterToSequenceParallelRegion',
    'GatherFromSequenceParallelRegion',
    'AllGatherFromTensorParallelRegion',
    'AllToAll'
]

__all__.extend(layers.__all__)
