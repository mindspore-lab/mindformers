mindformers.run_check
=====================

.. py:function:: mindformers.run_check()

    检查CANN、固件与驱动、MindSpore和MindFormers安装版本是否匹配。

    VERSION_MAP.json的结构如下：

    .. code-block::

        {
            'mf': {
                'version1': {
                    'prefer': 'prefered ms version',
                    'support': [competible ms version list]
                },
            },
            'ms': {
                'version1': {
                    'prefer' : 'prefered cann version',
                    'support': [competible cann version list]
                },
            },
            'cann': {
                'version1': {
                    'prefer' : 'prefered driver version',
                    'support': [competible driver version list]
                },
            }
        }