import os
import sys
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)

from xformer import XFormerBook
from xformer.tools import logger

'''
套件内容列表  XFormerBook 测试
'''
def test_xformer_book():

    XFormerBook.show_trainer_support_task_list()
    XFormerBook.show_pipeline_support_task_list()
    XFormerBook.show_model_support_list()
    XFormerBook.show_model_ckpt_url_list()
    XFormerBook.show_model_config_url_list()
    XFormerBook.show_project_path()
    XFormerBook.show_default_checkpoint_download_folder()
    XFormerBook.show_default_checkpoint_save_folder()

    logger.info(XFormerBook.get_trainer_support_task_list())
    logger.info(XFormerBook.get_pipeline_support_task_list())
    logger.info(XFormerBook.get_model_support_list())
    logger.info(XFormerBook.get_model_ckpt_url_list())
    logger.info(XFormerBook.get_model_config_url_list())
    logger.info(XFormerBook.get_project_path())
    logger.info(XFormerBook.get_default_checkpoint_download_folder())
    logger.info(XFormerBook.get_default_checkpoint_save_folder())

    # 两个默认路径可设置
    XFormerBook.set_default_checkpoint_download_folder(path)
    XFormerBook.set_default_checkpoint_save_folder(path)
    logger.info(XFormerBook.get_default_checkpoint_download_folder())
    logger.info(XFormerBook.get_default_checkpoint_save_folder())

test_xformer_book()