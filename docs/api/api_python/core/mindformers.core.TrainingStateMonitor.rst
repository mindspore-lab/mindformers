mindformers.core.TrainingStateMonitor
=====================================

.. py:class:: mindformers.core.TrainingStateMonitor(origin_epochs, config=None, step_interval=1, dataset_size=None, initial_epoch=0, initial_step=0, micro_batch_num=0, global_batch_size=0, tensor_model_parallel_size=0, check_for_nan_in_loss_and_grad=False, use_skip_data_by_global_norm=False, embedding_size=4096, use_local_norm=False)

    监控训练过程中指标变化的回调函数。

    参数：
        - **origin_epochs** (int) - 必选。训练的epoch数量。
        - **config** (dict, 可选) - 指标落盘的配置信息字典。字典的键取值如下。默认值： ``None`` ，此时将按照默认的键取值设置。

          - ``"target"`` - 指定要监控的参数的命名或正则表达式。必须是字符串列表，例如["layers.[01]", "attention"]。默认值： ``[".*"]`` ，即选择所有参数。
          - ``"invert"`` - 反选target指定的参数，即target指定的参数不会被监控。默认值： ``False`` 。
          - ``"local_norm_format"`` - 决定local norm的展示方式。必须是字符串'tensorboard'、'log'之一（分别代表写入tensorboard、日志），或包含它们的列表，或 ``None`` 。只有指定的参数会被监控，选择 'log' 时可能引入大量打印信息。设置为 ``None`` 以忽略该指标。默认值： ``None`` 。
          - ``"device_local_norm_format"`` - 决定device local norm的展示方式。必须是字符串'tensorboard'、'log'之一（分别代表写入tensorboard、日志），或包含它们的列表，或 ``None`` 。设置为 ``None`` 以忽略该指标。默认值： ``None`` 。
          - ``"local_loss_format"`` - 决定local loss的展示方式。必须是字符串'tensorboard'、'log'之一（分别代表写入tensorboard、日志），或包含它们的列表，或 ``None`` 。设置为 ``None`` 以忽略该指标。默认值： ``None`` 。
          - ``"device_local_loss_format"`` - 决定device local loss的展示方式。必须是字符串'tensorboard'、'log'之一（分别代表写入tensorboard、日志），或包含它们的列表，或 ``None`` 。设置为 ``None`` 以忽略该指标。默认值： ``None`` 。
          - ``"optimizer_state_format"`` - 决定优化器状态的展示方式。必须是字符串'tensorboard'、'log'之一（分别代表写入tensorboard、日志），或包含它们的列表，或 ``None`` 。只有指定参数的优化器状态会被监控，选择 'log' 时可能引入大量打印信息。设置为 ``None`` 以忽略该指标。默认值： ``None`` 。
          - ``"weight_state_format"`` - 决定权重L2-norm的展示方式。必须是字符串'tensorboard'、'log'之一（分别代表写入tensorboard、日志），或包含它们的列表，或 ``None`` 。设置为 ``None`` 以忽略该指标。默认值： ``None`` 。
          - ``"stable_rank_config"`` (dict, 可选) - 对有效秩指标落盘的配置信息字典。

            - ``"format"``: 决定权重的有效秩指标stable_rank和最大特征值指标max_eigenvalue的展示方式。必须是字符串'tensorboard'、'log'之一（分别代表写入tensorboard、日志），或包含它们的列表，或 ``None`` 。只有指定的参数会被监控。设置为 ``None`` 以忽略该指标。默认值： ``None`` 。
            - ``"step_interval"`` (int, 可选) - 设置记录指标stable_rank和max_eigenvalue的频率。必须为正整数。默认值： ``100`` .
            - ``"target"`` (list[str], 可选) - 设置指标stable_rank和max_eigenvalue要监控的参数的命名或正则表达式。必须是字符串列表，例如["layers.[01]", "attention"]。默认值： ``[".*"]`` ，即选择所有参数。
            - ``"do_aggregation"`` (bool, 可选) - 是否在权重参数被切分时先进行聚合，以计算指标stable_rank和max_eigenvalue。默认值： ``False`` .
            - ``"moe_show_mode"`` (Literal['full', 'statistics', 'all'], 可选) - 仅在为MoE模型计算指标stable_rank和max_eigenvalue时生效。设置为"full"会直接输出各专家的计算值；设置为"statistics"会输出各专家计算结果的统计值（min，max，mean）；设置为"all"会输出所有计算值和统计值。默认值： ``'all'`` .
            - ``"power_iteration_num"`` (int, 可选) - 设置幂迭代法计算max_eigenvalue的迭代次数。迭代次数越大，计算值越接近真实值，但相应计算开销会增大。必须为正整数。默认值： ``5`` .

          - ``"throughput_baseline"`` - 模型吞吐量的基线，用于计算线性度。必须为正数。会同时写入日志文件和tensorboard。设置为 ``None`` 以忽略该指标。默认值： ``None`` 。
          - ``"print_struct"`` - 是否打印模型结构。若是，则会在第一个step打印所有可训练参数的名字，并退出训练。默认值： ``False`` 。

        - **step_interval** (int, 可选) - 每多少次step对指标进行展示。默认值： ``1`` 。
        - **dataset_size** (int, 可选) - 数据下沉模式必选。训练的数据集数量。默认值： ``None`` 。
        - **initial_epoch** (int, 可选) - 训练开始的epoch数。默认值： ``0`` 。
        - **initial_step** (int, 可选) - 训练开始的step数。默认值： ``0`` 。
        - **micro_batch_num** (int, 可选) - 流水线并行时设置的MicroBatch大小。默认值： ``0`` 。
        - **global_batch_size** (int, 可选) - 总BatchSize大小。默认值： ``0`` 。
        - **tensor_model_parallel_size** (int, 可选) - 张量并行的切分数量。默认值： ``0`` 。
        - **check_for_nan_in_loss_and_grad** (bool, 可选) - 是否检查损失和梯度存在Nan。默认值： ``False`` 。
        - **use_skip_data_by_global_norm** (bool, 可选) - 是否开启通过global norm来进行数据跳过的功能。默认值： ``False`` 。
        - **embedding_size** (int, 可选) - 通过hidden_size * vocab_size来计算embedding norm的size。默认值： ``4096`` 。
        - **use_local_norm** (bool, 可选) - 是否打开local norm的开关。默认值： ``False`` 。