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

"""models init"""
from .auto import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskGeneration,
    AutoModelForPreTraining,
    AutoModelWithLMHead,
    AutoProcessor,
    AutoTokenizer,
    CONFIG_MAPPING,
    CONFIG_MAPPING_NAMES,
    IMAGE_PROCESSOR_MAPPING,
    ImageProcessingMixin,
    TOKENIZER_MAPPING
)
from .deepseek3 import (
    DeepseekV3Config,
    DeepseekV3ForCausalLM
)
from .glm2 import (
    ChatGLM2Config,
    ChatGLM2ForConditionalGeneration,
    ChatGLM2Model,
    ChatGLM2Tokenizer,
    ChatGLM2WithPtuning2,
    ChatGLM3Tokenizer,
    ChatGLM4Tokenizer,
    GLMProcessor
)
from .llama import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    LlamaProcessor,
    LlamaTokenizer,
    LlamaTokenizerFast
)
from .qwen2 import (
    Qwen2Config,
    Qwen2PreTrainedModel,
    Qwen2ForCausalLM,
)
from .qwen3 import (
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3ForCausalLM,
)
from .qwen3_moe import (
    Qwen3MoeConfig,
    Qwen3MoePreTrainedModel,
    Qwen3MoeForCausalLM,
)
from .tokenization_utils import (
    PreTrainedTokenizer,
    PreTrainedTokenizerBase
)
from .tokenization_utils_fast import PreTrainedTokenizerFast
from .modeling_utils import PreTrainedModel
from .multi_modal import (
    BaseImageToTextImageProcessor,
    BaseTextContentBuilder,
    BaseXModalToTextModel,
    BaseXModalToTextProcessor,
    BaseXModalToTextTransform,
    ModalContentTransformTemplate
)
from .configuration_utils import PretrainedConfig
from .base_config import BaseConfig
from .base_model import BaseModel
from .image_processing_utils import BaseImageProcessor
from .processing_utils import ProcessorMixin
from .base_processor import (
    BaseAudioProcessor,
    BaseProcessor
)
from .build_tokenizer import build_tokenizer
from .build_processor import build_processor
from .build_model import (
    build_encoder,
    build_head,
    build_model,
    build_model_config,
    build_network
)
from .utils import (
    CONFIG_NAME,
    FEATURE_EXTRACTOR_NAME,
    IMAGE_PROCESSOR_NAME
)

__all__ = ['PreTrainedTokenizer', 'PreTrainedTokenizerFast']
__all__.extend(auto.__all__)
__all__.extend(glm2.__all__)
__all__.extend(llama.__all__)
__all__.extend(multi_modal.__all__)
__all__.extend(configuration_utils.__all__)
__all__.extend(modeling_utils.__all__)
