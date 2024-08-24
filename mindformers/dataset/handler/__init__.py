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
"""MindFormers DataHandler."""
from mindformers.dataset.handler.build_data_handler import build_data_handler
from mindformers.dataset.handler.alpaca_handler import AlpacaInstructDataHandler
from mindformers.dataset.handler.codealpaca_handler import CodeAlpacaInstructDataHandler
from mindformers.dataset.handler.adgen_handler import AdgenInstructDataHandler
from mindformers.dataset.handler.llava_handler import LlavaInstructDataHandler

__all__ = ["build_data_handler", "AlpacaInstructDataHandler", "CodeAlpacaInstructDataHandler",
           "AdgenInstructDataHandler", "LlavaInstructDataHandler"]
