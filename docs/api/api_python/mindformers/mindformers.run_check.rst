mindformers.run_check
=====================

.. py:function:: mindformers.run_check(log_level='info')

    检查CANN、固件与驱动、MindSpore和MindFormers安装版本是否匹配。

    参数:
        - **log_level** (str，可选) - 控制日志打印，大小写不敏感。可选值：
          - ``debug``: 打印所有信息。
          - ``info``: 只打印info级别及以上的信息（默认值）。
          - ``warning``: 只打印warning级别及以上的信息。
          - ``error``: 只打印error级别及以上的信息。
          - ``critical``: 只打印critical级别的信息。
