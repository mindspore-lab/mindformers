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

__version__ = "1.6.0"

from mindformers import (
    core,
    dataset,
    models,
    modules,
    pet,
    tools,
    trainer,
    wrapper
)
from mindformers.pipeline import (
    FillMaskPipeline,
    MultiModalToTextPipeline,
    Pipeline,
    QuestionAnsweringPipeline,
    TextClassificationPipeline,
    TextGenerationPipeline,
    TokenClassificationPipeline,
    TranslationPipeline,
    pipeline
)
from mindformers.trainer import (
    BaseArgsConfig,
    BaseTrainer,
    CausalLanguageModelingTrainer,
    CheckpointConfig,
    ConfigArguments,
    ContextConfig,
    DataLoaderConfig,
    DatasetConfig,
    GeneralTaskTrainer,
    LRConfig,
    MaskedLanguageModelingTrainer,
    MultiModalToTextGenerationTrainer,
    OptimizerConfig,
    ParallelContextConfig,
    QuestionAnsweringTrainer,
    RunnerConfig,
    TextClassificationTrainer,
    Trainer,
    TrainingArguments,
    TranslationTrainer,
    WrapperConfig,
)
from mindformers.core import (
    ADGENMetric,
    AdamW,
    Came,
    CheckpointMonitor,
    ClipGradNorm,
    ConstantWarmUpLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CosineWithRestartsAndWarmUpLR,
    CosineWithWarmUpLR,
    CrossEntropyLoss,
    EmF1Metric,
    EntityScore,
    EvalCallBack,
    LearningRateWiseLayer,
    LinearWithWarmUpLR,
    MFLossMonitor,
    PerplexityMetric,
    PolynomialWithWarmUpLR,
    ProfileMonitor,
    PromptAccMetric,
    SummaryMonitor,
    TrainingStateMonitor,
    build_context,
    build_parallel_config,
    get_context,
    init_context,
    reset_parallel_config,
    set_context
)
from mindformers.dataset import (
    ADGenDataLoader,
    AdgenInstructDataHandler,
    AlpacaInstructDataHandler,
    BCHW2BHWC,
    BaseDataset,
    BaseMultiModalDataLoader,
    BatchCenterCrop,
    BatchNormalize,
    BatchPILize,
    BatchResize,
    BatchToTensor,
    CLUENERDataLoader,
    CausalLanguageModelDataset,
    CodeAlpacaInstructDataHandler,
    CommonDataLoader,
    GeneralDataset,
    IndexedDataLoader,
    KeyWordGenDataset,
    MaskLanguageModelDataset,
    ModalToTextSFTDataset,
    MultiSourceDataLoader,
    MultiTurnDataset,
    QuestionAnsweringDataset,
    RandomCropDecodeResize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    SFTDataLoader,
    SQuADDataLoader,
    TextClassificationDataset,
    ToolAlpacaDataLoader,
    TrainingDataLoader,
    TranslationDataset,
    WMT16DataLoader,
    build_data_handler,
    check_dataset_config,
    check_dataset_iterable
)
from mindformers.models import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskGeneration,
    AutoModelForMultipleChoice,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTextEncoding,
    AutoModelForTokenClassification,
    AutoModelForZeroShotImageClassification,
    AutoModelWithLMHead,
    AutoProcessor,
    AutoTokenizer,
    BaseAudioProcessor,
    BaseConfig,
    BaseImageProcessor,
    BaseImageToTextImageProcessor,
    BaseModel,
    BaseProcessor,
    BaseTextContentBuilder,
    BaseXModalToTextModel,
    BaseXModalToTextProcessor,
    BaseXModalToTextTransform,
    BasicTokenizer,
    BertConfig,
    BertForMultipleChoice,
    BertForPreTraining,
    BertForQuestionAnswering,
    BertForTokenClassification,
    BertModel,
    BertProcessor,
    BertTokenizer,
    BertTokenizerFast,
    CONFIG_MAPPING,
    CONFIG_MAPPING_NAMES,
    ChatGLM2Config,
    ChatGLM2ForConditionalGeneration,
    ChatGLM2Model,
    ChatGLM2Tokenizer,
    ChatGLM2WithPtuning2,
    ChatGLM3Tokenizer,
    ChatGLM4Tokenizer,
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
    GLMProcessor,
    IMAGE_PROCESSOR_MAPPING,
    ImageProcessingMixin,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    LlamaProcessor,
    LlamaTokenizer,
    LlamaTokenizerFast,
    MT5ForConditionalGeneration,
    ModalContentTransformTemplate,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3PreTrainedModel,
    Qwen3MoeConfig,
    Qwen3MoePreTrainedModel,
    Qwen3MoeForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    PretrainedConfig,
    T5Config,
    T5ForConditionalGeneration,
    T5PegasusTokenizer,
    T5Processor,
    T5Tokenizer,
    T5TokenizerFast,
    TOKENIZER_MAPPING,
)
from mindformers.modules import (
    AlibiTensor,
    AlibiTensorV2,
    AttentionMask,
    AttentionMaskHF,
    Dropout,
    EmbeddingOpParallelConfig,
    FeedForward,
    FixedSparseAttention,
    LayerNorm,
    Linear,
    LocalBlockSparseAttention,
    LowerTriangularMaskWithDynamic,
    MoEConfig,
    MultiHeadAttention,
    OpParallelConfig,
    RotaryEmbedding,
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerOpParallelConfig,
    TransformerRecomputeConfig,
    VocabEmbedding
)
from mindformers.wrapper import (
    AdaptiveLossScaleUpdateCell,
    MFPipelineWithLossScaleCell,
    MFTrainOneStepCell
)
from mindformers.tools import (
    ActionDict,
    DictConfig,
    MindFormerConfig,
    MindFormerModuleType,
    MindFormerRegister,
    logger,
)
from mindformers import generation
from mindformers.generation import (
    BaseStreamer,
    FrequencyPenaltyLogitsProcessor,
    GenerationConfig,
    GenerationMixin,
    GreedySearchLogitsProcessor,
    LogitNormalization,
    LogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    PresencePenaltyLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SamplingLogitsProcessor,
    TemperatureLogitsWarper,
    TextIteratorStreamer,
    TextStreamer,
    TopKLogitsWarper,
    TopPLogitsWarper
)
from mindformers.pet import (
    AdaAdapter,
    AdaLoraAdapter,
    LoraAdapter,
    LoraConfig,
    LoraModel,
    PetAdapter,
    PetConfig,
    PrefixTuningAdapter,
    PrefixTuningConfig,
    Ptuning2Adapter,
    Ptuning2Config
)
from mindformers import model_runner
from mindformers.model_runner import (
    ModelRunner,
    get_model
)
from mindformers.run_check import run_check
from mindformers.mindformer_book import MindFormerBook

__all__ = ['ModelRunner', 'run_check', 'pipeline', 'MultiModalToTextPipeline']
__all__.extend(core.__all__)
__all__.extend(dataset.__all__)
__all__.extend(generation.__all__)
__all__.extend(models.__all__)
__all__.extend(pet.__all__)
__all__.extend(tools.__all__)
__all__.extend(trainer.__all__)
__all__.extend(wrapper.__all__)
