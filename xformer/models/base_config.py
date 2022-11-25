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

'''
BaseConfig class,
which is all model configs' base class
'''
from ..tools import logger
from ..xformer_book import print_path_or_list


class BaseConfig(dict):
    '''
    Base Config for all models' config
    '''
    _support_list = []

    def __init__(self, **kwargs):
        super(BaseConfig, self).__init__()
        self.update(kwargs)

    def __getattr__(self, key):
        if key not in self:
            return None
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def to_dict(self):
        '''
        for yaml dump,
        transform from Config to a strict dict class
        '''
        return_dict = {}
        for key, val in self.items():
            if isinstance(val, BaseConfig):
                val = val.to_dict()
            return_dict[key] = val
        return return_dict

    @classmethod
    def show_support_list(cls):
        '''show support list of config'''
        logger.info("support list of %s is:", cls.__name__)
        print_path_or_list(cls._support_list)
