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
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
"""Configuration file for the Sphinx documentation builder."""

import os
import glob
import shutil
from sphinx.util import logging

# -- Project information -----------------------------------------------------

project = 'mindformers'
# pylint: disable=W0622
copyright = '2022, mindformers contributors'
author = 'mindformers contributors'

# The full version, including alpha/beta/rc tags
release = 'master'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_markdown_tables',
    'myst_parser',
    'nbsphinx',
    'sphinx.ext.mathjax',
    'IPython.sphinxext.ipython_console_highlighting'
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

pygments_style = 'sphinx'

autodoc_inherit_docstrings = False

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------

# Reconstruction of sphinx auto generated document translation.
language = 'zh_CN'
gettext_compact = False

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'sphinx_rtd_theme'

# Copy source files of chinese python api from mindscience repository.
logger = logging.getLogger(__name__)

# src_dir_mfl = os.path.join(os.getenv("MFM_PATH"), 'docs/api_python')
file_path = os.path.abspath('__ file __')

readme_path = os.path.realpath(os.path.join(file_path, '../../../../README.md'))
api_path = os.path.realpath(os.path.join(file_path, '../../../api_python'))
model_cards_path = os.path.realpath(os.path.join(file_path, '../../../model_cards'))
task_cards_path = os.path.realpath(os.path.join(file_path, '../../../task_cards'))

shutil.copy(readme_path, './README.md')
shutil.copytree(api_path, './api_python')
shutil.copytree(model_cards_path, './model_cards')
shutil.copytree(task_cards_path, './task_cards')

rst_files = {i.replace('.rst', '') for i in glob.glob('./**/*.rst', recursive=True)}


def setup(app):
    app.add_config_value('rst_files', set(), False)
