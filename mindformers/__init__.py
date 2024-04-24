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

__version__ = "1.1"

from mindformers import core, dataset, experimental, \
    models, modules, wrapper, tools
from mindformers.pipeline import *
from mindformers.trainer import *
from mindformers.core import *
from mindformers.dataset import *
from mindformers.experimental import *
from mindformers.models import *
from mindformers.modules import *
from mindformers.wrapper import *
from mindformers.tools import *
from mindformers import generation
from mindformers.generation import *
from mindformers.pet import *
from mindformers import model_runner
from mindformers.model_runner import *
from .mindformer_book import MindFormerBook

__all__ = []
__all__.extend(dataset.__all__)
__all__.extend(experimental.__all__)
__all__.extend(models.__all__)
__all__.extend(core.__all__)
__all__.extend(tools.__all__)
__all__.extend(generation.__all__)
__all__.extend(model_runner.__all__)
