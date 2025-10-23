mindformers.core.AdamW
======================

.. py:class:: mindformers.core.AdamW(params, learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, use_fused=False, amsgrad=False, maximize=False, swap=False)

    权重衰减Adam算法的实现。

    .. math::
        \begin{array}{l}
            &\newline
            &\hline \\
            &\textbf{Parameters}: \: 1^{\text {st }}\text {moment vector} \: m , \: 2^{\text {nd}} \:
             \text{moment vector} \: v , \\
            &\: gradients \: g, \: \text{learning rate} \: \gamma,
             \text {exponential decay rates for the moment estimates} \: \beta_{1} \: \beta_{2} , \\
            &\:\text {parameter vector} \: w_{0}, \:\text{timestep} \: t, \: \text{weight decay} \: \lambda \\
            &\textbf{Init}:  m_{0} \leftarrow 0, \: v_{0} \leftarrow 0, \: t \leftarrow 0, \:
             \text{init parameter vector} \: w_{0} \\[-1.ex]
            &\newline
            &\hline \\
            &\textbf{repeat} \\
            &\hspace{5mm} t \leftarrow t+1 \\
            &\hspace{5mm}\boldsymbol{g}_{t} \leftarrow \nabla f_{t}\left(\boldsymbol{w}_{t-1}\right) \\
            &\hspace{5mm}\boldsymbol{w}_{t} \leftarrow \boldsymbol{w}_{t-1}-\gamma\lambda\boldsymbol{w}_{t-1} \\
            &\hspace{5mm}\boldsymbol{m}_{t} \leftarrow \beta_{1} \boldsymbol{m}_{t-1}+\left(1-\beta_{1}\right)
             \boldsymbol{g}_{t} \\
            &\hspace{5mm}\boldsymbol{v}_{t} \leftarrow \beta_{2} \boldsymbol{v}_{t-1}+\left(1-\beta_{2}\right)
             \boldsymbol{g}_{t}^{2} \\
            &\hspace{5mm}\widehat{\boldsymbol{m}_{t}} \leftarrow \boldsymbol{m}_{t}/\big(1-\beta_{1}^{t} \big) \\
            &\hspace{5mm}\widehat{\boldsymbol{v}_{t}} \leftarrow \boldsymbol{v}_{t}/\big(1-\beta_{2}^{t} \big) \\
            &\hspace{5mm}\boldsymbol{w}_{t} \leftarrow \boldsymbol{w}_{t-1}-\gamma\widehat{\boldsymbol{m}_{t}}
             /\left(\sqrt{\widehat{\boldsymbol{v}_{t}}}+\epsilon\right) \\
            &\textbf{until}\text { stopping criterion is met } \\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
            &\textbf{return} \: \boldsymbol{w}_{t} \\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
        \end{array}

    :math:`m` 代表第一个动量矩阵 `moment1` ， :math:`v` 代表第二个动量矩阵 `moment2` ，:math:`\widehat{m}` 代表经过偏差修正的第一个动量矩阵， :math:`\widehat{v}` 代表经过偏差修正的第二个动量矩阵， :math:`g` 代表 `gradients` ，:math:`\gamma` 代表 `learning_rate` ，:math:`\beta_1, \beta_2` 代表 `beta1` 和 `beta2` ， :math:`t` 代表当前step，:math:`w` 代表 `params` ，:math:`\lambda` 代表 `weight_decay` 。

    参数：
        - **params** (Union[list[Parameter], list[dict]]) - 必须是 `Parameter` 组成的列表或字典组成的列表。当列表元素是字典时，字典的键可以是"params"、"lr"、"weight_decay"、和"order_params"：

          - ``"params"``：必选。当前组中的参数。值必须是 `Parameter` 的列表。
          - ``"lr"``：可选。如果键中包含 "lr"，则使用对应的学习率值。如果未包含，则使用优化器中的 `learning_rate`。支持固定和动态学习率。
          - ``"weight_decay"``：可选。如果键中包含 "weight_decay"，则使用对应的权重衰减值。如果未包含，则使用优化器中的 `weight_decay`。需要注意的是，权重衰减可以是常数值或 `Cell`。仅在应用动态权重衰减时才为 `Cell`。动态权重衰减类似于动态学习率，用户需要自定义仅以全局步数为输入的权重衰减计划，在训练过程中，优化器将调用 `WeightDecaySchedule` 实例来获取当前步的权重衰减值。
          - ``"order_params"``：可选。当参数被分组时，通常用于维护网络中出现的参数顺序以提高性能。值应为优化器中将遵循其顺序的参数。如果键中包含 `order_params`，其他键将被忽略，并且 'order_params' 的元素必须在 `params` 的一组中。

        - **learning_rate** (Union[float, int, Tensor, Iterable, LearningRateSchedule], 可选) - 默认值： ``1e-3`` 。

          - ``"float"``：固定学习率值。必须等于或大于 0。
          - ``"int"``：固定学习率值。必须等于或大于 0。将转换为浮点数。
          - ``"Tensor"``：其值应为标量或 1-D 向量。对于标量，将应用固定学习率。对于向量，学习率是动态的，第 i 步将使用第 i 个值作为学习率。
          - ``"Iterable"``：学习率是动态的。第 i 步将使用第 i 个值作为学习率。
          - ``"LearningRateSchedule"``：学习率是动态的。在训练过程中，优化器将调用 `LearningRateSchedule` 实例并以步数为输入来获取当前步的学习率。

        - **betas** (Union[list(float), tuple(float)], 可选) - `moment1`、 `moment2` 的指数衰减率。每一个参数范围（0.0,1.0）。默认值： ``(0.9, 0.999)`` 。
        - **eps** (float, 可选) - 将添加到分母中，以提高数值稳定性。必须大于0。默认值： ``1e-6`` 。
        - **weight_decay** (Union[float, int, Cell], 可选) - 权重衰减（L2 penalty）。默认值： ``0.0`` 。

          - ``"float"``：固定的权重衰减值。必须等于或大于 0。
          - ``"int"``：固定的权重衰减值。必须等于或大于 0。将被转换为浮点数。
          - ``"Cell"``：权重衰减是动态的。在训练过程中，优化器将调用 `Cell` 实例，并以步数为输入来获取当前步的权重衰减值。

        - **use_fused** (bool, 可选) - 是否启用融合算子。默认值： ``False`` 。
        - **amsgrad** (bool, 可选) - 是否使用 Adam 算法的 AMSGrad 变体。该变体会保留历史梯度平方的最大值，而非使用指数移动平均值。在某些情况下，这有助于改善模型的收敛性。为 `True` 时使用 AMSGrad 变体，仅支持 `use_fused=True` 的场景。默认值： ``False`` 。
        - **maximize** (bool, 可选) - 是否对目标函数执行最大化（而非最小化）操作。该选项适用于需要最大化奖励函数或效用函数的场景。为 `True` 时最大化目标函数，仅支持 `use_fused=True` 的场景。默认值： ``False`` 。
        - **swap** (bool, 可选) - 是否启用 swap_optimizer 特性，将优化器状态卸载到 CPU 存储，而非保留在 NPU 上。启用后需设置环境变量 `MS_DEV_RUNTIME_CONF="switch_inline:False"`。默认值： ``False`` 。

    输入：
        - **gradients** (tuple[Tensor]) - `params` 的梯度，shape与 `params` 相同。

    输出：
        tuple[bool]，所有元素都为True。

    异常：
        - **TypeError** - `learning_rate` 不是int、float、Tensor、Iterable或LearningRateSchedule。
        - **TypeError** - `parameters` 的元素不是Parameter或字典。
        - **TypeError** - `betas[0]` 、 `betas[1]` 或 `eps` 不是float。
        - **TypeError** - `weight_decay` 不是float或int。
        - **ValueError** - `eps` 小于等于0。
        - **ValueError** - `betas[0]` 、 `betas[1]` 不在（0.0,1.0）范围内。
        - **ValueError** - `weight_decay` 小于0。
