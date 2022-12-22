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

"""Build Feature Extractor API."""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from ..mindformer_book import MindFormerBook

def check_and_add_vocab_file_path(config, **kwargs):
    """And the vocab file path to the config if there is not vocab file in the config"""
    if 'vocab_file' in config:
        return
    class_name = config['type']
    dynamic_class = MindFormerRegister.get_cls(module_type='tokenizer', class_name=class_name)
    # If the tokenizer does not require the vocab_file, just stop
    if not dynamic_class.VOCAB_FILES:
        return
    name_or_path = class_name.lower().rstrip("tokenizer")
    path = kwargs.pop('lib_path', None)
    remote_tokenizer_support_list = MindFormerBook.get_tokenizer_support_list().keys()
    if name_or_path not in remote_tokenizer_support_list and path:
        read_vocab_file_dict, read_tokenizer_file_dict = \
            dynamic_class.read_files_according_specific_by_tokenizer(name_or_path=path)
        config.update(read_vocab_file_dict)
        config.update(read_tokenizer_file_dict)
    else:
        vocab_file = dynamic_class.cache_vocab_files(name_or_path=name_or_path)
        config['vocab_file'] = vocab_file


def build_tokenizer(
        config: dict = None, default_args: dict = None,
        module_type: str = 'tokenizer', class_name: str = None, **kwargs):
    """Build trainer API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        check_and_add_vocab_file_path(config, **kwargs)
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.TOKENIZER, default_args=default_args)

    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)
