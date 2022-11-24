import os
import sys
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)

from xformer import XFormerBook


'''
第一部分： 套件内容列表  XFormerBook
'''
XFormerBook.show_trainer_support_task_list()
XFormerBook.show_pipeline_support_task_list()
XFormerBook.show_model_support_list()
XFormerBook.show_model_ckpt_url_list()
XFormerBook.show_model_config_url_list()
XFormerBook.show_project_path()
XFormerBook.show_default_checkpoint_download_folder()
XFormerBook.show_default_checkpoint_save_folder()

print(XFormerBook.get_trainer_support_task_list())
print(XFormerBook.get_pipeline_support_task_list())
print(XFormerBook.get_model_support_list())
print(XFormerBook.get_model_ckpt_url_list())
print(XFormerBook.get_model_config_url_list())
print(XFormerBook.get_project_path())
print(XFormerBook.get_default_checkpoint_download_folder())
print(XFormerBook.get_default_checkpoint_save_folder())

XFormerBook.set_default_checkpoint_download_folder(path)
XFormerBook.set_default_checkpoint_save_folder(path)
