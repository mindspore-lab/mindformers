mindformers.pipeline
=====================

.. py:function:: mindformers.pipeline(task = None, model = None, tokenizer = None, image_processor = None, audio_processor = None, backend = "ms", **kwargs)

    通过流水线执行套件中已集成任务和模型的推理流程。

    参数：
        - **task** (str) - 支持的任务列表['zero_shot_image_classification', 'image_to_text_generation',
          'multi_modal_to_text_generation', 'masked_image_modeling', 'image_classification', 'translation',
          'fill_mask', 'text_classification', 'token_classification', 'question_answering', 'text_generation',
          'image_to_text_retrieval', 'segment_anything']。默认值： ``None`` 。
        - **model** (Union[str, PreTrainedModel, Model, Tuple[str, str]], 可选) - 执行任务的模型。默认值： ``None`` 。
        - **tokenizer** (PreTrainedTokenizerBase, 可选) - 模型分词器。默认值： ``None`` 。
        - **image_processor** (BaseImageProcessor, 可选) - 图片处理器。默认值： ``None`` 。
        - **audio_processor** (BaseAudioProcessor, 可选) - 音频处理器。默认值： ``None`` 。
        - **backend** (str, 可选) - 推理后端，当前仅支持 `ms`。默认值： ``"ms"`` 。
        - **kwargs** (Any) - 参考对应流水线任务的 kwargs 描述。

    返回：
        一个流水线任务。

    异常：
        - **KeyError** - 如果输入模型和任务不在支持列表中。
