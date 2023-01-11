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
import yaml
from distutils.core import setup
from setuptools import find_packages

VERSION_YAML_FILE = '../mindxsdk/build/conf/version.yaml'

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

cmd_class = {}


def get_version():
    if os.path.exists(VERSION_YAML_FILE):
        with open(VERSION_YAML_FILE, 'rb') as file:
            config_output = yaml.safe_load(file)
            version = config_output['mindxsdk']
    else:
        version = '1.0.1'
    return version


def do_setup(packages_data):
    setup(
        name='Ascend-mindxsdk-mxFoundationModel',

        version=get_version(),

        description='Huawei Ascend Research for Mindxsdk-mxFoundationModel Toolkit',

        keywords='ascend mindxsdk-mxFoundationModel finetune toolkit',

        # 详细的程序描述
        long_description=readme,
        long_description_content_type='text/markdown',

        # 编包依赖
        setup_requires=[
            'python_version>=3.7', 'setuptools>=61.2.0', 'pyyaml>=6.0', 'wheel'
        ],

        # 项目依赖的Python库
        install_requires=[
            'click', 'pyyaml', 'psutil', 'setproctitle', 'esdk-obs-python', 'tabulate'
        ],

        # 需要打包的目录列表
        packages=find_packages(
            # 不需要打包的目录列表
            exclude=[
                'test', 'test.*'
            ]
        ),

        package_data=packages_data,
        entry_points={
            'console_scripts': [
                'fm = fm.main:cli_wrapper'
            ],
        },
        cmdclass=cmd_class
    )


if __name__ == '__main__':
    # 待补充package data
    package_data = {
        'fm': [
            '*.py',
            '*/*.py',
            '*/*/*.py',
            'kmc/kmc_lib/*',
            'kmc/kmc_lib/*/*',
        ]
    }
    do_setup(package_data)
