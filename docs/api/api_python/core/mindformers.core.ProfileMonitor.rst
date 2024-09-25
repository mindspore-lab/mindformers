mindformers.core.ProfileMonitor
===============================

.. py:class:: mindformers.core.ProfileMonitor(start_step=1, stop_step=10, output_path=None, start_profile=True, profile_communication=False, profile_memory=True, config=None, **kwargs)

    监控训练过程的性能分析回调函数。

    参数：
        - **start_step** (int) - 启动profiling的训练步数。默认值： ``1`` 。
        - **stop_step** (int) - 停止profiling的训练步数。默认值： ``10`` 。
        - **output_path** (str) - 保存profiling生成文件的文件夹路径。默认值： ``None`` 。
        - **start_profile** (str) - 是否打开profiling功能。默认值： ``True`` 。
        - **profile_communication** (str) - 在分布式训练期间是否收集通信性能数据。默认值： ``False`` 。
        - **profile_memory** (str) - 是否收集张量的内存数据。默认值： ``True`` 。
        - **config** (dict) - 配置项，用于对相关配置信息进行profiling，比如并行配置。默认值： ``None`` 。