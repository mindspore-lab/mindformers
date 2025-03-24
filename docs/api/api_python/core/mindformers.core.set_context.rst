mindformers.core.set_context
============================

.. py:function:: mindformers.core.set_context(run_mode=None, **kwargs)

    设置运行环境的context。

    在运行程序之前，应配置context。如果没有配置，默认情况下将根据设备目标进行自动设置。

    .. note::
        设置属性时，必须输入属性名称。目前只有run_mode参数属于MindFormers context，Kwargs参数会被传到MindSpore的set_context接口中。
        可以通过关键字参数开关训练或推理的确定性计算。训练和推理的确定性计算的关键字参数分别为：
        ``train_precision_sync`` 和 ``infer_precision_sync``。
        开和关对应的操作是布尔值，``True`` 表示打开、``False`` 表示关闭。
        该操作是单次的，重复操作不成功且会有告警日志提示。

    参数：
        - **run_mode** (str, 可选) - 运行模式，可以是['train', 'finetune', 'eval', 'predict']其中之一。默认值： ``None`` 。
        - **kwargs** - MindSpore的context参数。