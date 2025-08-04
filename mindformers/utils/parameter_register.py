#  Copyright 2025 Huawei Technologies Co., Ltd
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ============================================================================
"""utils of parameter register."""
import mindspore as ms
from mindspore import Parameter, Tensor
from mindspore.ops import function as F


class ParameterRegister:
    """Used to register and manage intermediate model output parameters, with extensible support for arbitrary keys."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ParameterRegister, cls).__new__(cls)
            # pylint: disable=W0212
            cls._instance._params = {}
        return cls._instance

    def __init__(self):
        pass

    def register(self, name, value, requires_grad=False, parallel_optimizer=False):
        if name in self._params:
            raise KeyError(f"Parameter '{name}' already registered.")
        if isinstance(value, Tensor):
            param = Parameter(value, name=name, requires_grad=requires_grad, parallel_optimizer=parallel_optimizer)
        else:
            param = Parameter(Tensor(value, ms.float32), name=name, requires_grad=requires_grad,
                              parallel_optimizer=parallel_optimizer)
        self._params[name] = param
        return param

    def get(self, name):
        if name in self._params:
            return self._params[name]
        raise KeyError(f"{name} not registered.")

    def clear(self, name):
        """assign param to 0."""
        if name in self._params:
            param = self._params[name]
            zero_tensor = F.zeros_like(param, dtype=param.dtype)
            F.assign(param, zero_tensor)
        else:
            raise KeyError(f"{name} not registered.")

    def items(self):
        return self._params.items()

    def keys(self):
        return self._params.keys()

    def values(self):
        return self._params.values()


parameter_register = ParameterRegister()
