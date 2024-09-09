mindformers.AutoModelForVisualQuestionAnswering
================================================

.. py:class:: mindformers.AutoModelForVisualQuestionAnswering(*args, **kwargs)

    这是一个通用的模型类，当使用 from_pretrained() 类方法时，它会自动实例化模型，并返回。
    这个类不能直接使用 \_\_init\_\_() 实例化（会抛出异常）。

    .. py:method:: from_config(config, **kwargs)
        :classmethod:

        通过Config实例或者yaml文件，实例化模型，并返回。

        .. warning::
            这个API正处于实验阶段，在下一个版本中可能会有一些突破性的变化。

        参数：
            - **config** (Union[MindFormerConfig, PretrainedConfig, str]) - MindFormerConfig实例，yaml文件路径，或者PretrainedConfig实例（实验特性）。
            - **kwargs** (额外参数) - 传入的配置信息将会覆盖config中的配置信息。

        返回：
            一个模型实例。

    .. py:method:: from_pretrained(pretrained_model_name_or_dir: str, *model_args, **kwargs)
        :classmethod:

        从文件夹、或魔乐社区读取配置信息，实例化为模型，并返回。

        在实验特性中，根据配置对象的model_type属性，选择要实例化的模型类，对应关系如下：

        - "blip2" - Blip2ForConditionalGeneration

        .. warning::
            这个API正处于实验阶段，在下一个版本中可能会有一些突破性的变化。

        参数：
            - **pretrained_model_name_or_dir** (str) - 包含yaml文件和ckpt文件的文件夹路径、包含config.json文件和对应的权重文件的文件夹路径、或魔乐社区上的model_id。后两者为实验特性。
            - **model_args** (额外参数) - 会在实例化模型时，传给模型的 \_\_init\_\_() 方法。仅在实验特性时生效。
            - **kwargs** (额外参数) - 传入的配置信息将会覆盖从pretrained_model_name_or_dir中读取到的配置信息。

        返回：
            一个继承自PretrainedModel类的模型实例。

    .. py:method:: register(config_class, model_class, exist_ok=False)

        注册一个新的模型类到此类中。

        .. warning::
            这个API正处于实验阶段，在下一个版本中可能会有一些突破性的变化。

        参数：
            - **config_class** (PretrainedConfig) - 模型的Config类。
            - **model_class** (PretrainedModel) - 用于注册的模型类。
            - **exist_ok** (bool, 可选) - 为True时，若config_class已存在也不报错。默认值： ``False`` 。
