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
import os

_PROTOCOL = 'obs'
_PROTOCOL_S3 = 's3'


class Validator:
    """validator for checking input parameters"""

    @staticmethod
    def check_type(arg_value, arg_type):
        """Check int."""
        if not isinstance(arg_value, arg_type):
            raise TypeError('{} should be {} type, but get {}'.format(arg_value, arg_type, type(arg_value)))

    @staticmethod
    def is_obs_url(url):
        """Check obs url."""
        return url.startswith(_PROTOCOL + '://') or url.startswith(_PROTOCOL_S3 + '://')


def check_obs_url(url):
    """Check obs url."""
    if not (url.startswith(_PROTOCOL + '://') or url.startswith(_PROTOCOL_S3 + '://')):
        raise TypeError("obs url should be start with obs:// or s3://, but get {}".format(url))
    return True


def check_in_modelarts():
    """Check if the training is on modelarts.
    Returns:
        (bool): If it is True, it means ModelArts environment.
    """
    return 'MA_HOME' in os.environ or \
           'MA_JOB_DIR' in os.environ or \
           'BATCH_GROUP_NAME' in os.environ or \
           'MA_LOCAL_LOG_PATH' in os.environ


def format_path(path):
    """Check path."""
    return os.path.realpath(path)
