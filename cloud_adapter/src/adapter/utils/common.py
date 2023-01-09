# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
功能: 适配模块通用工具函数
"""
from datetime import datetime, timezone, timedelta


def get_unique_name(prefix="job"):
    """
    功能：生成名字唯一的job_name
    """
    time_str = datetime.now(tz=timezone(timedelta(hours=8))).strftime('%Y-%m-%d-%H-%M-%S')  # 北京时间
    return f'{prefix}_fm_{time_str}'


def select_with_priority(cmd_value,
                         value_in_app_config,
                         default=None):
    """
        功能： 按优先级（cmd_value > value_in_app_config > default）返回任务对应参数的最终值
        参数：
            cmd_value: 用户命令行中的参数值
            value_in_app_config: 在app_config中预配置的参数值
            default: 默认参数值
        返回值：
            指定参数的实际生效值
    """
    return cmd_value if cmd_value else value_in_app_config if value_in_app_config else default
