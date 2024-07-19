# Copyright 2024 Huawei Technologies Co., Ltd
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
"""networks."""

from .network import NetworkRegister, BaseNetwork
from .llama2 import Llama2Network

NetworkRegister.instance().reg("llama2_7b", Llama2Network)
NetworkRegister.instance().reg("llama2_13b", Llama2Network)
NetworkRegister.instance().reg("llama2_57b", Llama2Network)
NetworkRegister.instance().reg("llama2_70b", Llama2Network)
