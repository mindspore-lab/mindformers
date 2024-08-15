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

"""MultiModal model API."""

from .base_multi_modal_processor import BaseXModalToTextTransform, BaseXModalToTextProcessor, \
    BaseImageToTextImageProcessor
from .modal_content import BaseTextContentBuilder, ModalContentTransformTemplate
from .base_model import BaseXModalToTextModel

__all__ = ["BaseXModalToTextModel", "BaseXModalToTextTransform", "BaseXModalToTextProcessor",
           "BaseImageToTextImageProcessor", "BaseTextContentBuilder", "ModalContentTransformTemplate"]