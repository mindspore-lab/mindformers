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
Test Module for testing functions of XFormerBook class

How to run this:
windows:  pytest .\\tests\\ut\\test_xformer_book.py
linux:  pytest ./tests/ut/test_xformer_book.py
'''
from mindformers import XFormerBook
from mindformers.tools import logger


def test_xformer_book_show():
    '''
    Feature: mindformers book class, show content
    Description: Test to show content in XFormerBook
    Exception: None
    '''
    XFormerBook.show_trainer_support_task_list()
    XFormerBook.show_pipeline_support_task_list()
    XFormerBook.show_model_support_list()
    XFormerBook.show_model_ckpt_url_list()
    XFormerBook.show_model_config_url_list()
    XFormerBook.show_project_path()
    XFormerBook.show_default_checkpoint_download_folder()
    XFormerBook.show_default_checkpoint_save_folder()

def test_xformer_book_get():
    '''
    Feature: mindformers book class, get content
    Description: Test to get content of XFormerBook
    Exception: None
    '''
    logger.info(XFormerBook.get_trainer_support_task_list())
    logger.info(XFormerBook.get_pipeline_support_task_list())
    logger.info(XFormerBook.get_model_support_list())
    logger.info(XFormerBook.get_model_ckpt_url_list())
    logger.info(XFormerBook.get_model_config_url_list())
    logger.info(XFormerBook.get_project_path())
    logger.info(XFormerBook.get_default_checkpoint_download_folder())
    logger.info(XFormerBook.get_default_checkpoint_save_folder())

# default path is settable
def test_xformer_book_set():
    '''
    Feature: mindformers book class, get content
    Description: Test to set content for XFormerBook
    Exception: None
    '''
    path = XFormerBook.get_project_path()
    XFormerBook.set_default_checkpoint_download_folder(path)
    XFormerBook.set_default_checkpoint_save_folder(path)
    logger.info(XFormerBook.get_default_checkpoint_download_folder())
    logger.info(XFormerBook.get_default_checkpoint_save_folder())
