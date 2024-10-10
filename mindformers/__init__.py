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

__version__ = "r1.3.0"

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
    ImageClassificationPipeline,
    ImageToTextPipeline,
    MaskedImageModelingPipeline,
    MultiModalToTextPipeline,
    Pipeline,
    QuestionAnsweringPipeline,
    SegmentAnythingPipeline,
    TextClassificationPipeline,
    TextGenerationPipeline,
    TokenClassificationPipeline,
    TranslationPipeline,
    ZeroShotImageClassificationPipeline,
    pipeline
)
from mindformers.trainer import (
    BaseArgsConfig,
    BaseTrainer,
    CausalLanguageModelingTrainer,
    CheckpointConfig,
    CloudConfig,
    ConfigArguments,
    ContextConfig,
    ContrastiveLanguageImagePretrainTrainer,
    DataLoaderConfig,
    DatasetConfig,
    GeneralTaskTrainer,
    ImageClassificationTrainer,
    ImageToTextGenerationTrainer,
    ImageToTextRetrievalTrainer,
    LRConfig,
    MaskedImageModelingTrainer,
    MaskedLanguageModelingTrainer,
    MultiModalToTextGenerationTrainer,
    OptimizerConfig,
    ParallelContextConfig,
    QuestionAnsweringTrainer,
    RunnerConfig,
    TextClassificationTrainer,
    TokenClassificationTrainer,
    Trainer,
    TrainingArguments,
    TranslationTrainer,
    WrapperConfig,
    ZeroShotImageClassificationTrainer
)
from mindformers.core import (
    ADGENMetric,
    AdamW,
    Came,
    CheckpointMonitor,
    ClipGradNorm,
    CompareLoss,
    ConstantWarmUpLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CosineWithRestartsAndWarmUpLR,
    CosineWithWarmUpLR,
    CrossEntropyLoss,
    EmF1Metric,
    EntityScore,
    EvalCallBack,
    FP32StateAdamWeightDecay,
    FusedAdamWeightDecay,
    FusedCastAdamWeightDecay,
    L1Loss,
    LearningRateWiseLayer,
    LinearWithWarmUpLR,
    MFLossMonitor,
    MSELoss,
    ObsMonitor,
    PerplexityMetric,
    PolynomialWithWarmUpLR,
    ProfileMonitor,
    PromptAccMetric,
    SQuADMetric,
    SoftTargetCrossEntropy,
    SummaryMonitor,
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
    CaptionTransform,
    CausalLanguageModelDataset,
    Cifar100DataLoader,
    CodeAlpacaInstructDataHandler,
    CommonDataLoader,
    ContrastiveLanguageImagePretrainDataset,
    Flickr8kDataLoader,
    GeneralDataset,
    ImageCLSDataset,
    IndexedDataLoader,
    KeyWordGenDataset,
    LabelPadding,
    LlavaInstructDataHandler,
    MIMDataset,
    MaeMask,
    MaskLanguageModelDataset,
    Mixup,
    ModalToTextSFTDataset,
    MultiImgCapDataLoader,
    MultiSourceDataLoader,
    MultiTurnDataset,
    QuestionAnsweringDataset,
    RandomChoiceTokenizerForward,
    RandomCropDecodeResize,
    RandomErasing,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    RewardModelDataset,
    SFTDataLoader,
    SQuADDataLoader,
    SimMask,
    TextClassificationDataset,
    TokenClassificationDataset,
    TokenizeWithLabel,
    TokenizerForward,
    ToolAlpacaDataLoader,
    TrainingDataLoader,
    TranslationDataset,
    WMT16DataLoader,
    ZeroShotImageClassificationDataset,
    augment_and_mix_transform,
    auto_augment_transform,
    build_data_handler,
    check_dataset_config,
    check_dataset_iterable,
    rand_augment_transform
)
from mindformers.models import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForMaskGeneration,
    AutoModelForMaskedImageModeling,
    AutoModelForMultipleChoice,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTextEncoding,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    AutoModelForVisualQuestionAnswering,
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
    BloomConfig,
    BloomLMHeadModel,
    BloomModel,
    BloomProcessor,
    BloomRewardModel,
    BloomTokenizer,
    BloomTokenizerFast,
    CLIPConfig,
    CLIPImageProcessor,
    CLIPModel,
    CLIPProcessor,
    CLIPTextConfig,
    CLIPTokenizer,
    CLIPVisionConfig,
    CONFIG_MAPPING,
    CONFIG_MAPPING_NAMES,
    ChatGLM2Config,
    ChatGLM2ForConditionalGeneration,
    ChatGLM2Model,
    ChatGLM2Tokenizer,
    ChatGLM2WithPtuning2,
    ChatGLM3Tokenizer,
    ChatGLM4Tokenizer,
    ChatGLMTokenizer,
    CogVLM2Config,
    CogVLM2ContentTransformTemplate,
    CogVLM2ForCausalLM,
    CogVLM2ImageContentTransformTemplate,
    CogVLM2ImageForCausalLM,
    CogVLM2Tokenizer,
    CogVLM2VideoLM,
    CogVLM2VideoLMModel,
    EVA02Config,
    EVAModel,
    GLMChatModel,
    GLMConfig,
    GLMForPreTraining,
    GLMProcessor,
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2LMHeadModel,
    GPT2Model,
    GPT2Processor,
    GPT2Tokenizer,
    GPT2TokenizerFast,
    IMAGE_PROCESSOR_MAPPING,
    ImageEncoderConfig,
    ImageProcessingMixin,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaForCausalLMForCogVLM2Image,
    LlamaModel,
    LlamaProcessor,
    LlamaTokenizer,
    LlamaTokenizerFast,
    MT5ForConditionalGeneration,
    MaskData,
    ModalContentTransformTemplate,
    PanguAlphaConfig,
    PanguAlphaHeadModel,
    PanguAlphaModel,
    PanguAlphaProcessor,
    PanguAlphaPromptTextClassificationModel,
    PanguAlphaTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    PretrainedConfig,
    SamConfig,
    SamImageEncoder,
    SamImageProcessor,
    SamMaskDecoder,
    SamModel,
    SamProcessor,
    SamPromptEncoder,
    SwinConfig,
    SwinForImageClassification,
    SwinImageProcessor,
    SwinModel,
    SwinProcessor,
    T5Config,
    T5ForConditionalGeneration,
    T5PegasusTokenizer,
    T5Processor,
    T5Tokenizer,
    T5TokenizerFast,
    TOKENIZER_MAPPING,
    VHead,
    ViTConfig,
    ViTForImageClassification,
    ViTImageProcessor,
    ViTMAEConfig,
    ViTMAEForPreTraining,
    ViTMAEImageProcessor,
    ViTMAEModel,
    ViTMAEProcessor,
    ViTModel,
    ViTProcessor,
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_area,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle,
    nms,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points
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
    Local2ObsMonitor,
    MindFormerConfig,
    MindFormerModuleType,
    MindFormerRegister,
    Obs2Local,
    cloud_monitor,
    logger,
    mox_adapter
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
