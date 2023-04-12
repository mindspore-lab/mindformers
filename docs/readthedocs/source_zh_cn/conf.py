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
import re
import glob
import shutil
from sphinx.ext.autosummary import generate as g
from sphinx.ext import autodoc as sphinx_autodoc
from sphinx.ext.autosummary import Autosummary
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


# Modify regex for sphinx.ext.autosummary.generate.find_autosummary_in_lines.
gfile_abs_path = os.path.abspath(g.__file__)
autosummary_re_line_old = r"autosummary_re = re.compile(r'^(\s*)\.\.\s+autosummary::\s*')"
autosummary_re_line_new = r"autosummary_re = re.compile(r'^(\s*)\.\.\s+(ms[a-z]*)?autosummary::\s*')"
with open(gfile_abs_path, "r+", encoding="utf8") as f:
    data = f.read()
    data = data.replace(autosummary_re_line_old, autosummary_re_line_new)
    # pylint: disable=W0122
    exec(data, g.__dict__)

# Modify default signatures for autodoc.
autodoc_source_path = os.path.abspath(sphinx_autodoc.__file__)
autodoc_source_re = re.compile(r'stringify_signature\(.*?\)')
get_param_func_str = r"""\
import re
import inspect as inspect_

def get_param_func(func):
    try:
        source_code = inspect_.getsource(func)
        if func.__doc__:
            source_code = source_code.replace(func.__doc__, '')
        all_params_str = re.findall(r"def [\w_\d\-]+\(([\S\s]*?)(\):|\) ->.*?:)", source_code)
        all_params = re.sub("(self|cls)(,|, )?", '', all_params_str[0][0].replace("\n", "").replace("'", "\""))
        return all_params
    except:
        return ''

def get_obj(obj):
    if isinstance(obj, type):
        return obj.__init__

    return obj
"""

with open(autodoc_source_path, "r+", encoding="utf8") as f:
    code_str = f.read()
    code_str = autodoc_source_re.sub('"(" + get_param_func(get_obj(self.object)) + ")"', code_str, count=0)
    # pylint: disable=W0122
    exec(get_param_func_str, sphinx_autodoc.__dict__)
    # pylint: disable=W0122
    exec(code_str, sphinx_autodoc.__dict__)

# Copy source files of chinese python api from mindscience repository.
logger = logging.getLogger(__name__)

# src_dir_mfl = os.path.join(os.getenv("MFM_PATH"), 'docs/api_python')
file_path = os.path.abspath('__ file __')
src_dir_mfl = os.path.realpath(os.path.join(file_path, '../../../api_python'))

for i in os.listdir(src_dir_mfl):
    if os.path.isfile(os.path.join(src_dir_mfl, i)):
        if os.path.exists('./'+i):
            os.remove('./'+i)
        shutil.copy(os.path.join(src_dir_mfl, i), './'+i)
    else:
        if os.path.exists('./'+i):
            shutil.rmtree('./'+i)
        shutil.copytree(os.path.join(src_dir_mfl, i), './'+i)

rst_files = {i.replace('.rst', '') for i in glob.glob('./**/*.rst', recursive=True)}


def setup(app):
    app.add_directive('msplatformautosummary', MsPlatformAutoSummary)
    app.add_config_value('rst_files', set(), False)


class MsAutosummary(Autosummary):
    """
    Inherited from sphinx's autosummary, add titles and a column for the generated table.
    """

    def init(self):
        """
        init method
        """
        self.find_doc_name = ""
        self.third_title = ""
        self.default_doc = ""

    def extract_env_summary(self, doc) -> str:
        """Extract env summary from docstring."""
        env_sum = self.default_doc
        for index, piece in enumerate(doc):
            if piece.startswith(self.find_doc_name):
                env_sum = doc[index+1][4:]
        return env_sum


class MsPlatformAutoSummary(MsAutosummary):
    """
    Inherited from MsAutosummary. Add a third column about `Supported Platforms` to the table.
    """
    def init(self):
        """
        init method
        """
        self.find_doc_name = "Supported Platforms:"
        self.third_title = "**{}**".format(self.find_doc_name[:-1])
        self.default_doc = "``Ascend`` ``GPU`` ``CPU``"
