# Copyright 2023 Huawei Technologies Co., Ltd
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
"""context of mslite apis."""

from mindspore_lite.context import Context

from mindformers.inference.infer_config import InferConfig


def build_ascend_context(config: InferConfig):
    """build context for ascend."""
    context = Context()
    context.ascend.rank_id = config.rank_id
    context.ascend.device_id = config.device_id
    context.ascend.provider = "ge"
    context.target = [config.target]
    return context


def build_context(config: InferConfig):
    """build context for mslite."""
    if config.target == "Ascend":
        return build_ascend_context(config)

    raise KeyError(f"Supported data type keywords include: "
                   f"[Ascend], but get {config.target}")
