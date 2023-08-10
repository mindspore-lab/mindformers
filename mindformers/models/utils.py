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
"""Check Model Input Config."""
import os

import mindspore as ms
import mindspore.common.dtype as mstype
from ..tools.utils import is_version_ge
from ..tools.logger import logger


def convert_mstype(ms_type: str = "float16"):
    """Convert the string type to MindSpore type."""
    if isinstance(ms_type, mstype.Float):
        return ms_type
    if ms_type == "float16":
        return mstype.float16
    if ms_type == "float32":
        return mstype.float32
    raise KeyError(f"Supported data type keywords include: "
                   f"[float16, float32], but get {ms_type}")


def cell_reuse():
    """Cell reuse decorator."""
    def decorator(func):
        if os.getenv("MS_DEV_CELL_REUSE", "0") == "0" or not is_version_ge(ms.__version__, "2.1.0"):
            return func
        logger.info("Enable cell use mode at %s.", func.__class__.__name__)
        from mindspore._extends import cell_attr_register
        return cell_attr_register()(func)
    return decorator
