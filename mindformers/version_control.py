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
"""MindSpore Version Control"""
import os
import inspect
from functools import wraps

import mindspore as ms
from mindspore import nn
import mindspore.ops.operations as P
import mindspore.ops.functional as F
from mindspore._c_expression import MSContext
from mindspore.communication.comm_func import barrier

from mindformers.tools.utils import get_predict_run_mode
from .tools.utils import is_version_ge
from .tools.logger import logger


def is_910a():
    device = MSContext.get_instance().get_ascend_soc_version()
    return device in ['910a', 'ascend910']


def is_910b():
    device = MSContext.get_instance().get_ascend_soc_version()
    return device in ['910b', 'ascend910b']


def is_310p():
    device = MSContext.get_instance().get_ascend_soc_version()
    return device in ['310p', 'ascend310p']


def need_nz():
    device = MSContext.get_instance().get_ascend_soc_version()
    return device in ['310p', 'ascend310p', '910a', 'ascend910']


def get_predict_lazy_inline(func):
    """Predict lazy inline decorator."""

    def decorator(*args, **kwargs):
        if get_predict_run_mode():
            from mindspore.common import lazy_inline
            lazy_inline(func)(*args, **kwargs)

            logger.info("Predict enable lazy inline.")
        else:
            func(*args, **kwargs)

    return decorator


def get_lazy_inline(func):
    """Lazy inline decorator."""
    @wraps(func)
    def decorator(*args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        disable_lazy_inline = bound_args.kwargs.get('disable_lazy_inline', False)
        model_config = kwargs.get('config')
        if model_config and hasattr(model_config, 'disable_lazy_inline'):
            disable_lazy_inline = model_config.disable_lazy_inline

        if disable_lazy_inline:
            logger.info("The Lazy Inline compilation acceleration feature has been called, "
                        "and the feature is disabled by default.")
            func(*args, **kwargs)
            return

        from mindspore.common import lazy_inline
        logger.info("The Lazy Inline compilation acceleration feature is turned on.")
        lazy_inline(func)(*args, **kwargs)

    return decorator


def get_dropout(dropout_prob):
    if is_version_ge(ms.__version__, '1.11.0'):
        dropout = nn.Dropout(p=dropout_prob)
    else:
        dropout = nn.Dropout(keep_prob=1 - dropout_prob)
    return dropout


def get_tril():
    if is_version_ge(ms.__version__, '1.11.0'):
        tril = P.Tril()
    else:
        tril = nn.Tril()
    return tril


def get_norm():
    """Return ops.norm"""

    def tensor_norm1(input_tensor, tensor_ord=None, dim=None, keepdim=False, dtype=None):
        return F.norm(input_tensor, ord=tensor_ord, dim=dim, keepdim=keepdim, dtype=dtype)

    def tensor_norm2(input_tensor, tensor_ord=2, dim=None, keepdim=False, dtype=None):
        if dtype is not None:
            logger.warning("The 'dtype' is not available when mindspore version < '1.11.0'")
        if not isinstance(tensor_ord, int):
            raise TypeError("The type of 'tensor_ord' should be int when mindspore version < '1.11.0'")
        return F.norm(input_tensor, dim, p=tensor_ord, keep_dims=keepdim)

    if is_version_ge(ms.__version__, '1.11.0'):
        return tensor_norm1
    return tensor_norm2


def get_dataset_map(dataset, operations, input_columns=None, output_columns=None, num_parallel_workers=None, **kwargs):
    if is_version_ge(ms.__version__, "1.11.0"):
        return dataset.map(operations,
                           input_columns=input_columns,
                           output_columns=output_columns,
                           num_parallel_workers=num_parallel_workers,
                           **kwargs)
    return dataset.map(operations,
                       input_columns=input_columns,
                       output_columns=output_columns,
                       column_order=output_columns,
                       num_parallel_workers=num_parallel_workers,
                       python_multiprocessing=kwargs.get('python_multiprocessing', False),
                       max_rowsize=kwargs.get('max_rowsize', 16),
                       cache=kwargs.get('cache', None),
                       callbacks=kwargs.get('callbacks', None),
                       offload=kwargs.get('offload', None))


def get_identity():
    if is_version_ge(ms.__version__, "1.11.0"):
        return nn.Identity()
    return F.identity


def check_valid_flash_attention(fa_type=None):
    """check mindspore version is valid for input flash attention"""
    if not is_910b() and fa_type in ["PromptFlashAttention"]:
        logger.warning(f"Current device {MSContext.get_instance().get_ascend_soc_version()} do not support {fa_type}, "
                       f"please use 910b device.")
        return False
    return True


def is_version_python(cur_ver, tar_ver):
    """
        return cur_ver >= tar_ver.
        Check whether the current version is higher than or equal to the base version.
        for cur_ver: 3.7.10, tar_ver: 3.9.0, it return False.
        you can get python cur_ver through:
            cur_py_ver = sys.version.split(' ')[0]
    """
    version_split_char = '.'
    if version_split_char not in tar_ver or version_split_char not in cur_ver:
        raise ValueError("The version string will contain the `.`."
                         "For example, cur_ver: 3.7.10, tar_ver: 3.9.0")
    for x, y in zip(cur_ver.split(version_split_char), tar_ver.split(version_split_char)):
        if not x.isdigit() or not y.isdigit():
            continue
        if int(x) != int(y):
            return int(x) >= int(y)
    return True


def check_rmsnorm_big_kernel_valid():
    """check whether rmsnorm big kernel is valid"""
    if not is_910a():
        return True
    return False


def check_valid_gmm_op(gmm_version=None):
    """check mindspore version is valid for groupedmatmul"""
    version_map = {"GroupedMatmulV4": "2.6.0"}
    version_info = ms.__version__.split('rc')
    version_valid = is_version_ge(version_info[0], version_map.get(gmm_version))
    if version_valid is None:
        raise ValueError(f"gmm_version should be in {list(version_map.keys())}, but get {gmm_version}")
    if not version_valid:
        logger.warning(f"Current MindSpore do not support {gmm_version}, "
                       f"please upgrade to {version_valid} or later version.")
        return False
    return True


def check_valid_mindspore_gs():
    """check mindspore golden-stick version is valid or not"""
    import mindspore_gs
    version_valid = is_version_ge(mindspore_gs.__version__, "0.6.0")
    if not version_valid:
        logger.warning(f"Current MindSpore Golden-Stick version does not match"
                       f"the MindFormers version, please upgrade to 0.6.0 or later version.")
        return False
    return True


def check_tft_valid():
    """check tft is valid"""
    env_enable = os.getenv("MS_ENABLE_TFT", "")
    remain_required_flags = ["TTP:1", "UCE:1", "ARF:1", "HCCE:1"]
    return any(flag in env_enable for flag in remain_required_flags)


def check_tre_valid():
    """check mindspore version is valid for tre"""
    version_valid = is_version_ge(ms.__version__, "2.6.0")
    if not version_valid:
        logger.warning("Current MindSpore version does not support tre, please upgrade to 2.6.0 or later version.")
        return False
    env_enable = os.getenv("MS_ENABLE_TFT", "")
    return "TRE:1" in env_enable


def check_tsp_valid():
    """check mindspore version is valid for tsp"""
    version_valid = is_version_ge(ms.__version__, "2.7.0")
    if not version_valid:
        logger.warning("Current MindSpore version does not support tsp, please upgrade to 2.7.0 or later version.")
        return False
    env_enable = os.getenv("MS_ENABLE_TFT", "")
    return "TSP:1" in env_enable


def check_arf_status(cb_params):
    """check arf flag when using ARF, make sure that the number of operators executed by all nodes is consistent"""
    if check_tft_valid() and ("ARF:1" in os.getenv("MS_ENABLE_TFT", "")):
        return cb_params.is_arf
    return False


def check_is_reboot_node():
    """check whether is the reboot node in arf"""
    version_valid = is_version_ge(ms.__version__, "2.6.0")
    if not version_valid:
        logger.warning(
            "Current MindSpore version does not support skip barrier, please upgrade to 2.6.0 or later version.")
        return False
    env_enable = os.getenv("MS_ENABLE_TFT", "")
    if "ARF:1" in env_enable:
        from mindspore._c_expression import is_reboot_node
        return is_reboot_node()
    return False


def skip_barrier_controller(times: int = 1):
    """whether to skip barrier or exec barrier controller"""
    skip_barrier_sig = check_is_reboot_node()

    if skip_barrier_sig:
        logger.warning("barrier is not enable.")
        return

    for i in range(times):
        logger.info(f"barrier {i + 1} started")
        barrier()
        logger.info(f"barrier {i + 1} completed")
    logger.info(f"barrier {times} completed")


def check_seqpp_fa_opt_support():
    """check mindspore version if sparse adaptive adjustment of fa with seqpipie"""
    return is_version_ge(ms.__version__, "2.6.0")


def check_moveto_op_support():
    """Check whether MindSpore supports the MoveTo operator."""
    return is_version_ge(ms.__version__, "2.6.0")


def check_moev3_valid():
    """Check whether the current MindSpore version is valid for MoeV3."""
    return is_version_ge(ms.__version__, "2.6.0")


def check_safetensors_addition_param_support():
    """check mindspore version if support safetensors name map and unify with choice func/max proc num"""
    return is_version_ge(ms.__version__, "2.6.0")


def set_ms_deterministic(deterministic):
    """Set deterministic computing through mindspore."""
    logger.debug("The version of MindSpore is %s, "
                 "set deterministic compution by set_deterministic()",
                 ms.__version__)
    ms.set_deterministic(deterministic)
