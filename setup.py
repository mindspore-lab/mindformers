#!/usr/bin/env python3
# encoding: utf-8
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
"""setup package."""
import os
import stat
import shutil
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py


commit_id = os.getenv('COMMIT_ID').replace("\n", "")
build_path = os.getenv('BUILD_PATH')
pkg_dir = os.path.join(build_path, 'lib')
pwd = os.path.dirname(os.path.realpath(__file__))


class BuildPy(build_py):
    """BuildPy."""

    @staticmethod
    def _write_commit_file(file):
        """ Write commit info into `file`. """
        file.write("__commit_id__ = '{}'\n".format(commit_id))

    @staticmethod
    def _write_extra_info():
        """ Write extra info into `file` such as commit info. """
        mindspore_dir = os.path.join(pkg_dir, 'mindtransformer')
        BuildPy._update_permissions(mindspore_dir)
        mindspore_dir = os.path.join(pwd, 'mindtransformer')
        BuildPy._update_permissions(mindspore_dir)
        mindspore_dir = os.path.join(pkg_dir, 'mindtransformer', '.commit_id')
        BuildPy._update_permissions(mindspore_dir)
        mindspore_dir = os.path.join(pwd, 'mindtransformer', '.commit_id')
        BuildPy._update_permissions(mindspore_dir)
        commit_file = os.path.join(pkg_dir, 'mindtransformer', '.commit_id')
        with open(commit_file, 'w') as f:
            BuildPy._write_commit_file(f)
        commit_file = os.path.join(pwd, 'mindtransformer', '.commit_id')
        with open(commit_file, 'w') as f:
            BuildPy._write_commit_file(f)

    @staticmethod
    def _update_permissions(path):
        """
        Update permissions of `path`.

        Args:
            path (str): Target directory path.
        """

        for dirpath, dirnames, filenames in os.walk(path):
            for dirname in dirnames:
                dir_fullpath = os.path.join(dirpath, dirname)
                os.chmod(dir_fullpath, stat.S_IREAD | stat.S_IWRITE |
                         stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)
            for filename in filenames:
                file_fullpath = os.path.join(dirpath, filename)
                os.chmod(file_fullpath, stat.S_IREAD | stat.S_IWRITE)

    @staticmethod
    def _copy_ops_files():
        """Copy ops files to pkg."""
        src_path = os.path.join(pwd, 'mindtransformer')
        for dirpath, dirnames, _ in os.walk(src_path):
            if 'ccsrc' in dirnames:
                src_dir_path = os.path.join(dirpath, 'ccsrc')
                dst_dir_path = os.path.join(pkg_dir, 'mindtransformer', dirpath.split('mindtransformer/')[-1], 'ccsrc')
                if os.path.exists(dst_dir_path):
                    shutil.rmtree(dst_dir_path)
                shutil.copytree(src_dir_path, dst_dir_path)

    def run(self):
        super().run()
        BuildPy._copy_ops_files()
        BuildPy._write_extra_info()


setup(
    name='mindtransformer',
    version='0.1.0',
    author='The MindSpore Authors',
    author_email='contact@mindspore.cn',
    description='A MindSpore pretrained large model package.',
    url='https://www.mindspore.cn',
    download_url='https://gitee.com/mindspore/transformer/tags',
    packages=find_packages(include=['mindtransformer*']),
    cmdclass={'build_py': BuildPy},
    install_requires=[
        'numpy >= 1.17.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.7',
    license='Apache 2.0',
    keywords='mindspore optimization quantization golden-stick',
)
