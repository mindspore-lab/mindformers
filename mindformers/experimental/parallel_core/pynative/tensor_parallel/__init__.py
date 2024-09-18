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

from .mappings import all_to_all_hp2sp, all_to_all_sp2hp
from .mappings import CopyToModelParallelRegion
from .mappings import ScatterToModelParallelRegion
from .mappings import GatherFromModelParallelRegion
from .mappings import ReduceFromModelParallelRegion
from .mappings import ReduceScatterToSequenceParallelRegion
from .mappings import ReduceScatterToTensorParallelRegion
from .mappings import ScatterToSequenceParallelRegion
from .mappings import GatherFromSequenceParallelRegion
from .mappings import GatherFromTensorAndExpertParallelRegion
from .mappings import AllGatherFromTensorParallelRegion
from .mappings import AllToAll, AllToAllSP2HP
from .layers import *
from .lora_layers import *
from .random import *

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
    'GatherFromTensorAndExpertParallelRegion',
    'AllGatherFromTensorParallelRegion',
    'AllToAll',
    'AllToAllSP2HP'
]

__all__.extend(layers.__all__)
__all__.extend(random.__all__)
