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
from fm.src.engine.options import ScenarioOption, AppConfigOption, DataPathOption, \
    OutputPathOption, NodeNumOption, DeviceNumOption, DeviceTypeOption, BackendOption, ResumeOption, \
    ModelConfigPathOption, JobNameOption, CkptFileOption


def evaluate_options():
    options = [ScenarioOption(),
               AppConfigOption(),
               DataPathOption(),
               OutputPathOption(),
               NodeNumOption(),
               DeviceNumOption(),
               DeviceTypeOption(),
               BackendOption(),
               ResumeOption(),
               ModelConfigPathOption(),
               JobNameOption(),
               CkptFileOption()]

    def decorator(f):
        if not hasattr(f, '__click_params__'):
            f.__click_params__ = []
        f.__click_params__ += options
        return f

    return decorator
