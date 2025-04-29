mindformers.core.WarmUpStableDecayLR
====================================

.. py:class:: mindformers.core.WarmUpStableDecayLR(learning_rate: float, lr_end: float = 1e-7, warmup_steps: int = None, warmup_lr_init: float = 0., warmup_ratio: float = None, total_steps: int = None, decay_start_steps: int = None, decay_start_ratio: float = None, **kwargs)

    带稳定衰减的预热学习率调度器。

    该学习率调度器分为三个阶段：
    1. **预热阶段**：学习率从初始值 `warmup_lr_init` 线性增长到基准学习率 `learning_rate`。
    2. **稳定阶段**：保持基准学习率不变。
    3. **衰减阶段**：学习率从 `learning_rate` 线性衰减到最终值 `lr_end`。

    **预热阶段公式**：

    .. math::
        \eta_t = \eta_{\text{warmup}} + t \times \frac{\eta_{\text{base}} - \eta_{\text{warmup}}}{\text{warmup\_steps}}

    其中：
    - :math:`\eta_{\text{warmup}}` 是初始预热学习率 (`warmup_lr_init`)
    - :math:`\eta_{\text{base}}` 是基准学习率 (`learning_rate`)
    - :math:`t` 是当前步数（不超过 `warmup_steps`）

    **衰减阶段公式**：

    .. math::
        \eta_t = \eta_{\text{base}} - (\eta_{\text{base}} - \eta_{\text{end}}) \times \frac{t - T_{\text{decay\_start}}}{T_{\text{decay\_steps}}}

    其中：
    - :math:`\eta_{\text{end}}` 是最终学习率 (`lr_end`)
    - :math:`T_{\text{decay\_start}}` 是衰减起始步数 (`decay_start_steps`)
    - :math:`T_{\text{decay\_steps}}` 是衰减总步数 (`total_steps - decay_start_steps`)

    参数：
        - **learning_rate** (float) - 基准学习率。
        - **lr_end** (float, 可选) - 学习率的最终值。默认值： ``1e-7`` 。
        - **warmup_steps** (int, 可选) - 预热阶段的步数。若未指定，将通过 `warmup_ratio` 计算。默认值： ``None`` 。
        - **warmup_lr_init** (float, 可选) - 预热阶段的初始学习率。默认值： ``0.0`` 。
        - **warmup_ratio** (float, 可选) - 预热步数占总训练步数的比例（覆盖 `warmup_steps`）。默认值： ``None`` 。
        - **total_steps** (int, 可选) - 总训练步数（必须在指定 `warmup_ratio` 或 `decay_start_ratio` 时需提供）。默认值： ``None`` 。
        - **decay_start_steps** (int, 可选) - 衰减阶段的起始步数。若未指定，将通过 `decay_start_ratio` 计算。默认值： ``None`` 。
        - **decay_start_ratio** (float, 可选) - 衰减起始步数占总训练步数的比例（覆盖 `decay_start_steps`）。默认值： ``None`` 。

    异常：
        - **ValueError** - 如果 `lr_end` 大于或等于初始学习率 `learning_rate`。

    输入：
        - **global_step** (int) - 当前全局步数。

    输出：
        学习率。
