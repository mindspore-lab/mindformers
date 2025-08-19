# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Base Dataset Handler."""

from abc import ABC, abstractmethod

from mindformers.tools.logger import logger
from mindformers.models.build_tokenizer import build_tokenizer
from mindformers.core.context import is_legacy_model


class BaseInstructDataHandler(ABC):
    """
    Base class for instruction-based data handlers.

    This class provides common utilities for handling and tokenizing
    instruction-style datasets. It manages tokenizer initialization,
    padding, and ignore token IDs used for label masking during training.
    """

    def __init__(self, **kwargs):
        self.use_legacy = is_legacy_model()
        tokenize_args = kwargs.get('tokenizer')
        self.tokenizer = self.build_tokenizer(tokenize_args)

        if self.tokenizer is not None and getattr(self.tokenizer, 'pad_token_id', None) is not None:
            self.pad_token_id = self.tokenizer.pad_token_id
        else:
            logger.info(f"tokenizer not set or it have no pad_token_id, set 0 as `pad_token_id`.")
            self.pad_token_id = kwargs.get('pad_token_id', 0)

        self.ignore_token_id = kwargs.get('ignore_token_id', -100)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Process a dataset."""
        raise NotImplementedError

    def build_tokenizer(self, config):
        """
        Build a tokenizer based on the provided configuration.

        Args:
            config (dict or None): Tokenizer configuration. May include 'name' for pretrained tokenizers,
                                   or parameters for building a tokenizer from scratch.

        Returns:
            tokenizer instance or None: Initialized tokenizer.
        """
        if config is None:
            return None

        pretrained_model_dir = config.pop("pretrained_model_dir", None)
        trust_remote_code = config.pop("trust_remote_code", False)
        tokenizer = build_tokenizer(config,
                                    use_legacy=self.use_legacy,
                                    pretrained_model_dir=pretrained_model_dir,
                                    trust_remote_code=trust_remote_code)
        if hasattr(tokenizer, 'padding_side'):
            tokenizer.padding_side = config.get("padding_side", "right")
        return tokenizer
