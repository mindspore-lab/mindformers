# Copyright 2024 The HuggingFace Inc. team.
# adapt to mindformers, add PtqConfig and SmoothQuantConfig
#       Huawei Technologies Co., Ltd
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
"""Quantization Configuration."""
import copy
import json
import os
from enum import Enum
from typing import Any, Dict, List, Union
from collections import OrderedDict
from dataclasses import dataclass, field

from mindspore import dtype as msdtype
from mindspore_gs.ptq import PTQConfig, PTQMode, OutliersSuppressionType, QuantGranularity, PrecisionRecovery
from mindspore_gs.common import BackendTarget

dtype_map = {"None": None,
             "bool": msdtype.bool_,
             "int": msdtype.int_,
             "int4": msdtype.qint4x2,
             "int8": msdtype.int8,
             "int16": msdtype.int16,
             "int32": msdtype.int32,
             "int64": msdtype.int64,
             "uint8": msdtype.uint8,
             "uint16": msdtype.uint16,
             "uint32": msdtype.uint32,
             "uint64": msdtype.uint64,
             "float": msdtype.float_,
             "float16": msdtype.float16,
             "float32": msdtype.float32,
             "float64": msdtype.float64,
             "bfloat16": msdtype.bfloat16,
             "complex64": msdtype.complex64,
             "complex128": msdtype.complex128}

outliers_map = {"None": OutliersSuppressionType.NONE,
                "smooth": OutliersSuppressionType.SMOOTH,
                "awq": OutliersSuppressionType.AWQ,
                "outlier-suppression+": OutliersSuppressionType.OUTLIER_SUPPRESSION_PLUS}

precision_recovery_map = {"none": PrecisionRecovery.NONE,
                          "gptq": PrecisionRecovery.GPTQ}

quant_granularity_map = {"per_tensor": QuantGranularity.PER_TENSOR,
                         "per_channel": QuantGranularity.PER_CHANNEL,
                         "per_token": QuantGranularity.PER_TOKEN,
                         "per_group": QuantGranularity.PER_GROUP}


class QuantizationMethod(str, Enum):
    RTN = "rtn"
    PTQ = "ptq"
    SMOOTH_QUANT = 'smooth_quant'


@dataclass
class QuantizationConfigMixin:
    """
    Mixin class for quantization config
    """

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        """
        Instantiates a [`QuantizationConfigMixin`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            return_unused_kwargs (`bool`,*optional*, defaults to `False`):
                Whether or not to return a list of unused keyword arguments. Used for `from_pretrained` method in
                `PreTrainedModel`.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`QuantizationConfigMixin`]: The configuration object instantiated from those parameters.
        """

        config = cls(**config_dict)

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        if return_unused_kwargs:
            return config, kwargs
        return config

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default
                `QuantizationConfig()` is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            writer.write(json_string)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return copy.deepcopy(self.__dict__)

    def __iter__(self):
        """allows `dict(obj)` for situations where obj may be a dict or QuantizationConfigMixin"""
        for attr, value in copy.deepcopy(self.__dict__).items():
            yield attr, value

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def update(self, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
        returning all the unused kwargs.

        Args:
            kwargs (`Dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `Dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # Remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs


@dataclass
class RtnConfig(QuantizationConfigMixin, PTQConfig):
    """Config for post trainning quantization.

    Args:

        mode (:class:`mindspore_gs.ptq.PTQMode`): Flag for ptq mode, ``QUANTIZATION`` for quantization mode,
            ``DEPLOY`` for deploy mode, MindFormers only supports deploy mode now.
        backend (:class:`mindspore_gs.ptq.BackendTarget`): Flag for backend target,
            ``NONE`` for no specific backend, ``ASCEND`` for ascend backend.
        weight_dtype (mindspore.dtype): Used to configure the quantization type of weight. mindspore.dtype.int8
            indicates that the weight is quantized by 8 bits, and None indicates that it is not quantized.
        activation_dtype (mindspore.dtype): Used to configure the quantization type of activation.
            mindspore.dtype.int8 indicates that the activation is quantized by 8 bits,
            and None indicates that it is not quantized.
        kvcache_dtype (mindspore.dtype): Used to configure the quantization type of kvcache. mindspore.dtype.int8
            indicates that the kvcache is quantized by 8 bits, and None indicates that it is not quantized.
        algorithm_args (Union[dict, dataclass]): Used to configure hyperparameters of algorithms such as RTN,
            SmoothQuant, and OmniQuant.
        modules_to_not_convert (List[str]): Blacklist of opname. Layers in network with name fuzzy matched with this
            blacklist will not being quanted.
        outliers_suppression (OutliersSuppressionType): the method of outliers suprression,
            support None and smooth currently.
        act_quant_granularity (QuantGranularity): the quant granularity of act,
            support QuantGranularity.PER_TENSOR and QuantGranularity.PER_TOKEN currently.
        kvcache_quant_granularity(QuantGranularity): the quant granularity of kvcache,
            support QuantGranularity.PER_CHANNEL and QuantGranularity.PER_TOKEN currently.

    Raises:
        ValueError: If `mode` is not PTQMode.QUANTIZE or PTQMode.DEPLOY.
        ValueError: If `backend` is not BackendTarget.NONE or BackendTarget.ASCEND.
        TypeError: if `modules_to_not_convert` is not a list of str.
        ValueError: If `weight_dtype` is not mindspore.dtype.int8 or None.
        ValueError: If `activation_dtype` is not mindspore.dtype.int8 or None.
        ValueError: If `kvcache_dtype` is not mindspore.dtype.int8 or None.
        ValueError: If `outliers_suppression` is not OutliersSuppressionType.NONE or OutliersSuppressionType.SMOOTH.
        ValueError: If `kvcache_quant_granularity` is QuantGranularity.PER_TOKEN but `kvcache_dtype` is
                    not mindspore.dtype.int8.
        ValueError: If `act_quant_granularity` is QuantGranularity.PER_TOKEN but `weight_dtype` is
                    not mindspore.dtype.int8 or `activation_dtype` is not mindspore.dtype.int8.

    Examples:
        >>> from mindformers.utils.quantization_config import RtnConfig
        >>> RtnConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, opname_blacklist=['layer0'])
        SmoothQuantConfig(mode=<PTQMode.DEPLOY: 'deploy'>, backend=<BackendTarget.ASCEND: 'ascend'>,
                            opname_blacklist=['layer0'], algo_args={})
    """
    def __init__(
            self,
            quant_method: QuantizationMethod.RTN,
            mode: PTQMode = PTQMode.DEPLOY,
            backend: BackendTarget = BackendTarget.ASCEND,
            weight_dtype: msdtype = msdtype.int8,
            activation_dtype: msdtype = None,
            kvcache_dtype: msdtype = None,
            modules_to_not_convert: List[str] = field(default_factory=list),
            outliers_suppression: OutliersSuppressionType = OutliersSuppressionType.NONE,
            algorithm_args: Union[dict, object] = field(default_factory=dict),
            act_quant_granularity: str = "per_tensor",
            kvcache_quant_granularity: str = "per_channel",
            **kwargs
    ):
        super().__init__()
        self.quant_method = quant_method
        self.mode = mode
        self.backend = backend
        self.opname_blacklist = modules_to_not_convert
        self.algo_args = algorithm_args
        self.weight_quant_dtype = dtype_map.get(weight_dtype)
        self.kvcache_quant_dtype = dtype_map.get(kvcache_dtype)
        self.act_quant_dtype = dtype_map.get(activation_dtype)
        self.outliers_suppression = outliers_map[outliers_suppression]
        self.act_quant_granularity = quant_granularity_map.get(act_quant_granularity)
        self.kvcache_quant_granularity = quant_granularity_map.get(kvcache_quant_granularity)
        self.init_check()

    def init_check(self):
        r"""
        Safety checker that arguments are correct
        """
        accepted_mode = [PTQMode.DEPLOY]
        accepted_backend = [BackendTarget.ASCEND]
        accepted_weights = [None, msdtype.int8]
        accepted_activations = [None, msdtype.int8]
        accepted_kvcache = [None, msdtype.int8]
        accepted_act_granularity = [QuantGranularity.PER_TENSOR, QuantGranularity.PER_TOKEN]
        accepted_kvcache_granularity = [QuantGranularity.PER_CHANNEL, QuantGranularity.PER_TOKEN]
        if self.mode not in accepted_mode:
            raise ValueError(f"Only support {accepted_mode} but found {self.mode}")
        if self.backend not in accepted_backend:
            raise ValueError(f"Only support {accepted_backend} but found {self.backend}")
        if self.backend not in accepted_backend:
            raise ValueError(f"Only support quant weights in {accepted_weights} but found {self.weight_quant_dtype}")
        if self.weight_quant_dtype not in accepted_weights:
            raise ValueError(f"Only support quant weights in {accepted_weights} but found {self.weight_quant_dtype}")
        if self.act_quant_dtype not in accepted_activations:
            raise ValueError(
                f"Only support activation weights in {accepted_activations} but found {self.act_quant_dtype}")
        if self.kvcache_quant_dtype not in accepted_kvcache:
            raise ValueError(f"Only support kvcache weights in {accepted_kvcache} but found {self.kvcache_quant_dtype}")
        if self.act_quant_dtype is not None or self.outliers_suppression != OutliersSuppressionType.NONE:
            raise ValueError(f"RTN algorithm only support A16W8、C8、A16W8C8, please set the correct configuration."
                             f"Now the configuration is act_quant_dtype={self.act_quant_dtype},"
                             f"weight_quant_dtype={self.weight_quant_dtype},"
                             f"kvcache_quant_dtype={self.kvcache_quant_dtype},"
                             f"outliers_suppression={self.outliers_suppression}")
        if self.act_quant_granularity not in accepted_act_granularity:
            raise ValueError(f"Only support act_quant_granularity in {accepted_act_granularity} but found "
                             f"{self.act_quant_granularity}")
        if self.kvcache_quant_granularity not in accepted_kvcache_granularity:
            raise ValueError(f"Only support kvcache_quant_granularity in {accepted_kvcache_granularity} but found "
                             f"{self.kvcache_quant_granularity}")
        if (self.weight_quant_dtype != msdtype.int8 or self.act_quant_dtype != msdtype.int8) and \
            self.act_quant_granularity is QuantGranularity.PER_TOKEN:
            raise ValueError("when self.act_quant_granularity is QuantGranularity.PER_TOKEN, self.weight_quant_dtype:"
                             f"{self.weight_quant_dtype} and self.act_quant_dtype: {self.act_quant_dtype} must be "
                             "mindspore.dtype.int8.")
        if self.kvcache_quant_dtype != msdtype.int8 and self.kvcache_quant_granularity is QuantGranularity.PER_TOKEN:
            raise ValueError("when self.kvcache_quant_granularity is QuantGranularity.PER_TOKEN, "
                             "self.kvcache_quant_dtype must be mindspore.dtype.int8.")


@dataclass
class PtqConfig(QuantizationConfigMixin, PTQConfig):
    """Config for post trainning quantization.

    Args:

        mode (:class:`mindspore_gs.ptq.PTQMode`): Flag for ptq mode, ``QUANTIZATION`` for quantization mode,
            ``DEPLOY`` for deploy mode, MindFormers only supports deploy mode now.
        backend (:class:`mindspore_gs.ptq.BackendTarget`): Flag for backend target,
            ``NONE`` for no specific backend, ``ASCEND`` for ascend backend.
        weight_dtype (mindspore.dtype): Used to configure the quantization type of weight. mindspore.dtype.int8
            indicates that the weight is quantized by 8 bits, and None indicates that it is not quantized.
        activation_dtype (mindspore.dtype): Used to configure the quantization type of activation.
            mindspore.dtype.int8 indicates that the activation is quantized by 8 bits,
            and None indicates that it is not quantized.
        kvcache_dtype (mindspore.dtype): Used to configure the quantization type of kvcache. mindspore.dtype.int8
            indicates that the kvcache is quantized by 8 bits, and None indicates that it is not quantized.
        algorithm_args (Union[dict, dataclass]): Used to configure hyperparameters of algorithms such as RTN,
            SmoothQuant, and OmniQuant.
        modules_to_not_convert (List[str]): Blacklist of opname. Layers in network with name fuzzy matched with this
            blacklist will not being quanted.
        outliers_suppression (OutliersSuppressionType): the method of outliers suprression.
            Choose from (`"None"`, `"smooth"`, `"awq"`, `"outlier-suppression+"`). Defaults: ``"None"``.
        precision_recovery (PrecisionRecovery): the method of precision recovery, used to precision compensation of
            weights during quantization, support None and GPTQ currently.
        act_quant_granularity (QuantGranularity): the quant granularity of act,
            support QuantGranularity.PER_TENSOR and QuantGranularity.PER_TOKEN currently.
        kvcache_quant_granularity(QuantGranularity): the quant granularity of kvcache,
            support QuantGranularity.PER_CHANNEL and QuantGranularity.PER_TOKEN currently.
        weight_quant_granularity(QuantGranularity): the quant granularity of weight,
            support QuantGranularity.PER_CHANNEL and QuantGranularity.PER_GROUP currently.
        group_size (int): group_size of per_group quantization, suggest using 64 or 128.
        layer_policies (dict, optional): quantization strategy for layers. Defaults: ``None``.
            The key of `layer_policies` is regular string to match the layer name,
            the value of `layer_policies` is the quantization strategy of current layer.

    Raises:
        ValueError: If `mode` is not PTQMode.QUANTIZE or PTQMode.DEPLOY.
        ValueError: If `backend` is not BackendTarget.NONE or BackendTarget.ASCEND.
        TypeError: if `modules_to_not_convert` is not a list of str.
        ValueError: If `weight_dtype` is not mindspore.dtype.int8 or None.
        ValueError: If `activation_dtype` is not mindspore.dtype.int8 or None.
        ValueError: If `kvcache_dtype` is not mindspore.dtype.int8 or None.
        ValueError: If `outliers_suppression` is not OutliersSuppressionType.NONE or OutliersSuppressionType.SMOOTH.
        ValueError: If `precision_recovery` is not PrecisionRecovery.NONE or PrecisionRecovery.GPTQ.
        ValueError: If `kvcache_quant_granularity` is QuantGranularity.PER_TOKEN but `kvcache_dtype` is
                    not mindspore.dtype.int8.
        ValueError: If `act_quant_granularity` is QuantGranularity.PER_TOKEN but `weight_dtype` is
                    not mindspore.dtype.int8 or `activation_dtype` is not mindspore.dtype.int8.
        ValueError: If `weight_quant_granularity` is QuantGranularity.PER_GROUP but `group_size` is not in [64, 128].
        TypeError: If `group_size` is not Int.
        TypeError: If `layer_policies` not dict, the key of `layer_policies` is not string or
            the value of `layer_policies` is not a regular expression.

    Examples:
        >>> from mindformers.utils.quantization_config import PtqConfig
        >>> PtqConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, opname_blacklist=['layer0'])
        SmoothQuantConfig(mode=<PTQMode.DEPLOY: 'deploy'>, backend=<BackendTarget.ASCEND: 'ascend'>,
                            opname_blacklist=['layer0'], algo_args={})
    """
    def __init__(
            self,
            quant_method: QuantizationMethod.PTQ,
            mode: PTQMode = PTQMode.DEPLOY,
            backend: BackendTarget = BackendTarget.ASCEND,
            weight_dtype: msdtype = msdtype.int8,
            activation_dtype: msdtype = None,
            kvcache_dtype: msdtype = None,
            modules_to_not_convert: List[str] = field(default_factory=list),
            outliers_suppression: str = "None",
            precision_recovery: str = "none",
            algorithm_args: Union[dict, object] = field(default_factory=dict),
            act_quant_granularity: str = "per_tensor",
            kvcache_quant_granularity: str = "per_channel",
            weight_quant_granularity: str = "per_channel",
            group_size: int = 0,
            layer_policies: Dict[str, Any] = None,
            **kwargs
    ):
        super().__init__()
        self.quant_method = quant_method
        self.mode = mode
        self.backend = backend
        self.opname_blacklist = modules_to_not_convert
        self.algo_args = algorithm_args
        self.weight_quant_dtype = dtype_map.get(weight_dtype)
        self.kvcache_quant_dtype = dtype_map.get(kvcache_dtype)
        self.act_quant_dtype = dtype_map.get(activation_dtype)
        self.act_quant_granularity = quant_granularity_map.get(act_quant_granularity)
        self.kvcache_quant_granularity = quant_granularity_map.get(kvcache_quant_granularity)
        self.weight_quant_granularity = quant_granularity_map.get(weight_quant_granularity)
        self.outliers_suppression = outliers_map[outliers_suppression]
        self.precision_recovery = precision_recovery_map[precision_recovery]
        self.group_size = group_size

        self.layer_policies = OrderedDict()
        if layer_policies:
            for key, value in layer_policies.items():
                self.layer_policies[key] = PtqConfig.from_dict(value)
        self.init_check()

    def init_check(self):
        r"""
        Safety checker that arguments are correct
        """
        accepted_mode = [PTQMode.DEPLOY]
        accepted_backend = [BackendTarget.ASCEND]
        accepted_weights = [None, msdtype.int8, msdtype.qint4x2]
        accepted_activations = [None, msdtype.int8]
        accepted_kvcache = [None, msdtype.int8]
        accepted_outliers_suppression = [OutliersSuppressionType.NONE, OutliersSuppressionType.SMOOTH,
                                         OutliersSuppressionType.AWQ, OutliersSuppressionType.OUTLIER_SUPPRESSION_PLUS]
        accepted_precision_recovery = [PrecisionRecovery.NONE, PrecisionRecovery.GPTQ]
        accepted_group_size = [0, 64, 128]
        if self.mode not in accepted_mode:
            raise ValueError(f"Only support {accepted_mode} but found {self.mode}")
        if self.backend not in accepted_backend:
            raise ValueError(f"Only support {accepted_backend} but found {self.backend}")
        if self.backend not in accepted_backend:
            raise ValueError(f"Only support quant weights in {accepted_weights} but found {self.weight_quant_dtype}")
        if self.weight_quant_dtype not in accepted_weights:
            raise ValueError(f"Only support quant weights in {accepted_weights} but found {self.weight_quant_dtype}")
        if self.act_quant_dtype not in accepted_activations:
            raise ValueError(
                f"Only support activation weights in {accepted_activations} but found {self.act_quant_dtype}")
        if self.kvcache_quant_dtype not in accepted_kvcache:
            raise ValueError(f"Only support kvcache weights in {accepted_kvcache} but found {self.kvcache_quant_dtype}")
        if self.outliers_suppression not in accepted_outliers_suppression:
            raise ValueError(f"Only support outliers suppression in {accepted_outliers_suppression} but found "
                             f"{self.outliers_suppression}")
        if self.precision_recovery not in accepted_precision_recovery:
            raise ValueError(f"Only support precision recovery in {accepted_precision_recovery} but found "
                             f"{self.precision_recovery}")
        if self.weight_quant_dtype is None and self.act_quant_dtype == msdtype.int8:
            raise ValueError("PTQ algorithm not support only quant activation.")
        if (self.weight_quant_dtype != msdtype.int8 or self.act_quant_dtype != msdtype.int8) and \
            self.act_quant_granularity is QuantGranularity.PER_TOKEN:
            raise ValueError("when self.act_quant_granularity is QuantGranularity.PER_TOKEN, self.weight_quant_dtype:"
                             f"{self.weight_quant_dtype} and self.act_quant_dtype: {self.act_quant_dtype} must be "
                             "mindspore.dtype.int8.")
        self._check_quant_granularity()
        if self.group_size not in accepted_group_size:
            raise ValueError(f"Only support group_size in {accepted_group_size} but found "
                             f"{self.group_size}")
        self._check_layer_policies()

    def _check_quant_granularity(self):
        """check quant granularity"""
        accepted_act_granularity = [QuantGranularity.PER_TENSOR, QuantGranularity.PER_TOKEN]
        accepted_kvcache_granularity = [QuantGranularity.PER_CHANNEL, QuantGranularity.PER_TOKEN]
        accepted_weight_granularity = [QuantGranularity.PER_CHANNEL, QuantGranularity.PER_GROUP]
        if self.act_quant_granularity not in accepted_act_granularity:
            raise ValueError(f"Only support act_quant_granularity in {accepted_act_granularity} but found "
                             f"{self.act_quant_granularity}")
        if self.kvcache_quant_granularity not in accepted_kvcache_granularity:
            raise ValueError(f"Only support kvcache_quant_granularity in {accepted_kvcache_granularity} but found "
                             f"{self.kvcache_quant_granularity}")
        if self.weight_quant_granularity not in accepted_weight_granularity:
            raise ValueError(f"Only support weight_quant_granularity in {accepted_weight_granularity} but found "
                             f"{self.weight_quant_granularity}")
        if self.kvcache_quant_dtype != msdtype.int8 and self.kvcache_quant_granularity is QuantGranularity.PER_TOKEN:
            raise ValueError("when self.kvcache_quant_granularity is QuantGranularity.PER_TOKEN, "
                             "self.kvcache_quant_dtype must be mindspore.dtype.int8.")

    def _check_layer_policies(self):
        """_check_layer_policies"""
        import re
        if not isinstance(self.layer_policies, OrderedDict):
            raise TypeError(f'layer_policies should be an OrderedDict, bug got {type(self.layer_policies)}.')
        if any(not isinstance(key, str) for key in self.layer_policies.keys()):
            raise TypeError(f'all key of layer_policies should be a string.')
        try:
            for key, _ in self.layer_policies.items():
                re.compile(key)
        except re.error as e:
            raise TypeError(f'The regular string of layer_policies not correct, please check and try again.') from e


@dataclass
class SmoothQuantConfig(QuantizationConfigMixin, PTQConfig):
    """Config for post trainning quantization.

    Args:

        mode (:class:`mindspore_gs.ptq.PTQMode`): Flag for ptq mode, ``QUANTIZATION`` for quantization mode,
            ``DEPLOY`` for deploy mode, MindFormers only supports deploy mode now.
        backend (:class:`mindspore_gs.ptq.BackendTarget`): Flag for backend target,
            ``NONE`` for no specific backend, ``ASCEND`` for ascend backend.
        weight_dtype (mindspore.dtype): Used to configure the quantization type of weight. mindspore.dtype.int8
            indicates that the weight is quantized by 8 bits, and None indicates that it is not quantized.
        activation_dtype (mindspore.dtype): Used to configure the quantization type of activation.
            mindspore.dtype.int8 indicates that the activation is quantized by 8 bits,
            and None indicates that it is not quantized.
        kvcache_dtype (mindspore.dtype): Used to configure the quantization type of kvcache. mindspore.dtype.int8
            indicates that the kvcache is quantized by 8 bits, and None indicates that it is not quantized.
        algorithm_args (Union[dict, dataclass]): Used to configure hyperparameters of algorithms such as RTN,
            SmoothQuant, and OmniQuant.
        modules_to_not_convert (List[str]): Blacklist of opname. Layers in network with name fuzzy matched with this
            blacklist will not being quanted.
        outliers_suppression (OutliersSuppressionType): the method of outliers suprression,
            support None and smooth currently.
        act_quant_granularity (QuantGranularity): the quant granularity of act,
            support QuantGranularity.PER_TENSOR currently.
        kvcache_quant_granularity(QuantGranularity): the quant granularity of kvcache,
            support QuantGranularity.PER_CHANNEL currently.

    Raises:
        ValueError: If `mode` is not PTQMode.QUANTIZE or PTQMode.DEPLOY.
        ValueError: If `backend` is not BackendTarget.NONE or BackendTarget.ASCEND.
        TypeError: if `modules_to_not_convert` is not a list of str.
        ValueError: If `weight_dtype` is not mindspore.dtype.int8 or None.
        ValueError: If `activation_dtype` is not mindspore.dtype.int8 or None.
        ValueError: If `kvcache_dtype` is not mindspore.dtype.int8 or None.
        ValueError: If `outliers_suppression` is not OutliersSuppressionType.NONE or OutliersSuppressionType.SMOOTH.
        ValueError: If `act_quant_granularity` is not QuantGranularity.PER_TENSOR.
        ValueError: If `outliers_suppression` is not QuantGranularity.PER_CHANNEL.

    Examples:
        >>> from mindformers.utils import SmoothQuantConfig
        >>> SmoothQuantConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, opname_blacklist=['layer0'])
        SmoothQuantConfig(mode=<PTQMode.DEPLOY: 'deploy'>, backend=<BackendTarget.ASCEND: 'ascend'>,
                            opname_blacklist=['layer0'], algo_args={})
    """
    def __init__(
            self,
            quant_method: QuantizationMethod.SMOOTH_QUANT,
            mode: PTQMode = PTQMode.DEPLOY,
            backend: BackendTarget = BackendTarget.ASCEND,
            weight_dtype: msdtype = msdtype.int8,
            activation_dtype: msdtype = None,
            kvcache_dtype: msdtype = None,
            modules_to_not_convert: List[str] = field(default_factory=list),
            outliers_suppression: OutliersSuppressionType = OutliersSuppressionType.NONE,
            algorithm_args: Union[dict, object] = field(default_factory=dict),
            act_quant_granularity: str = "per_tensor",
            kvcache_quant_granularity: str = "per_channel",
            **kwargs
    ):
        super().__init__()
        self.quant_method = quant_method
        self.mode = mode
        self.backend = backend
        self.opname_blacklist = modules_to_not_convert
        self.algo_args = algorithm_args
        self.weight_quant_dtype = dtype_map[weight_dtype]
        self.kvcache_quant_dtype = dtype_map[kvcache_dtype]
        self.act_quant_granularity = quant_granularity_map.get(act_quant_granularity)
        self.kvcache_quant_granularity = quant_granularity_map.get(kvcache_quant_granularity)
        self.act_quant_dtype = dtype_map[activation_dtype]
        self.outliers_suppression = outliers_map[outliers_suppression]
        self.init_check()

    def init_check(self):
        r"""
        Safety checker that arguments are correct
        """
        accepted_mode = [PTQMode.DEPLOY]
        accepted_backend = [BackendTarget.ASCEND]
        accepted_weights = [msdtype.int8]
        accepted_activations = [None, msdtype.int8]
        accepted_kvcache = [None, msdtype.int8]
        if self.mode not in accepted_mode:
            raise ValueError(f"Only support {accepted_mode} but found {self.mode}")
        if self.backend not in accepted_backend:
            raise ValueError(f"Only support {accepted_backend} but found {self.backend}")
        if self.backend not in accepted_backend:
            raise ValueError(f"Only support quant weights in {accepted_weights} but found {self.weight_quant_dtype}")
        if self.weight_quant_dtype not in accepted_weights:
            raise ValueError(f"Only support quant weights in {accepted_weights} but found {self.weight_quant_dtype}")
        if self.act_quant_dtype not in accepted_activations:
            raise ValueError(
                f"Only support activation weights in {accepted_activations} but found {self.act_quant_dtype}")
        if self.kvcache_quant_dtype not in accepted_kvcache:
            raise ValueError(f"Only support kvcache weights in {accepted_kvcache} but found {self.kvcache_quant_dtype}")
        if self.act_quant_granularity is not QuantGranularity.PER_TENSOR or self.kvcache_quant_granularity \
                is not QuantGranularity.PER_CHANNEL:
            raise ValueError("Only support act_quant_granularity is QuantGranularity.PER_TENSOR and "
                             "kvcache_quant_granularity is QuantGranularity.PER_CHANNEL but found "
                             f"act_quant_granularity is {self.act_quant_granularity} and "
                             f"kvcache_quant_granularity is {self.kvcache_quant_granularity}.")
        do_a8w8 = self.act_quant_dtype == msdtype.int8 and self.weight_quant_dtype == msdtype.int8 and \
                    self.outliers_suppression == OutliersSuppressionType.SMOOTH and \
                    self.kvcache_quant_dtype is None
        do_nothing = self.act_quant_dtype is None and self.weight_quant_dtype is None and \
                    self.outliers_suppression is None and \
                    self.kvcache_quant_dtype is None
        if not do_a8w8 and not do_nothing:
            raise ValueError("SmoothQuant algorithm only support A8W8, please set act_quant_dtype=int8,"
                             "weight_quant_dtype=int8 and outliers_suppression='smooth'."
                             f"Now the configuration is act_quant_dtype={self.act_quant_dtype},"
                             f"weight_quant_dtype={self.weight_quant_dtype},"
                             f"kvcache_quant_dtype={self.kvcache_quant_dtype},"
                             f"outliers_suppression={self.outliers_suppression}")
