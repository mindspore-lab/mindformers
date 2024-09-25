mindformers.dataset.MultiTurnDataset
====================================

.. py:class:: mindformers.dataset.MultiTurnDataset(dataset_config: dict = None)

    多轮对话数据集。
    生成的数据集有两列 `[input_ids, labels]` 。列 `input_ids` 为int32类型。列 `labels` 为int32类型。

    参数：
        - **dataset_config** (dict) - 必选。数据集配置信息，必须是至少包含以下键值对的字典。

          - **data_loader** - 对应值必须是包含data loader配置信息的字典。 `data_loader` 的键可以是"type"、"dataset_dir"、"phase"、"shuffle"和"origin_columns"。

            - ``"type"`` - 必选。数据集的类型。必须是 `str` 或 `type` 类型。
            - ``"dataset_dir"`` - 必选。数据集文件所在目录。
            - ``"phase"`` - 必选。需要读取的数据集的子集，可选值为"train"和"eval"。
            - ``"shuffle"`` - 必选。指示是否混洗数据集。必须是 `bool` 类型。
            - ``"origin_columns"`` - 必选。必须是两个字符串组成的列表，表示"prompt"和"answer"在数据文件中的对应列名。

          - **tokenizer** - 对应值必须是包含分词器配置信息的字典，或一个分词器实例。
          - **batch_size** - 每个批次的大小。
          - **drop_remainder** - 是否在最后一个批次的数据项数小于批次大小时，丢弃最后一个批次。
          - **num_parallel_workers** - 并行执行数据映射处理的进程/线程数。
          - **python_multiprocessing** - 是否启用Python的Multi-Process模块以加速映射操作。
          - **repeat** - 数据集重复的次数。
          - **max_seq_length** - 序列的最大长度。

    返回：
        `MultiTurnDataset` 实例。

    异常：
        - **ValueError** -  Python版本低于3.9。
        - **ValueError** -  `dataset_config.data_loader` 中缺少 `dataset_dir` 或 `dataset_config.data_loader.dataset_dir` 指示的路径不存在。
        - **ValueError** -  词元数和预测词元的损失掩膜数不一致。
        - **ValueError** -  输入词元的索引数和标签数不一致。