# -*- coding: utf-8 -*-
"""
功能: file transporter based on MoXing
版权信息: 华为技术有限公司, 版权所有(C) 2022-2023
"""

import os
import hashlib
from importlib import import_module

from lk_utils import is_valid_path
from ma.constants import NODE_NUM

MD5_KEYWORD_LIST = ('md5chksum', 'contentmd5', 'content_md5', 'content-md5')
MOX_COPY_THREAD_NUM = (16 // int(NODE_NUM) + 1) if NODE_NUM.isdigit() and int(NODE_NUM) >= 1 else 1


class FileTransporter(object):
    """
        file transporter by using MoXing
    """

    def __init__(self, path_pair, is_folder, existence_detection, info):
        self.info = info
        self.is_folder = is_folder
        self.existence_detection = existence_detection
        self.path_pair = path_pair

        try:
            self.mox_module = import_module('moxing')
        except ModuleNotFoundError:
            raise RuntimeError("Module MoXing (required by scenario modelarts) is not found.")

        self.src_path, self.dst_path = path_pair.src_path, path_pair.dst_path
        if not self.check_path(self.src_path, existence_detection=existence_detection):
            raise ValueError(f"Source path of {self.info} is not valid.")
        if not self.check_path(self.dst_path):
            raise ValueError(f"Destination path of {self.info} is not valid.")

    def transport(self):
        if not self.path_pair.is_same():
            if self.is_folder:
                self.mox_module.file.copy_parallel(self.src_path, self.dst_path, threads=MOX_COPY_THREAD_NUM)
                if check_obs_path(self.src_path) and not check_obs_path(self.dst_path):
                    self._validate_folder_integrity(self.mox_module, self.src_path, self.dst_path)
            else:
                self.mox_module.file.copy(self.src_path, self.dst_path)
                if check_obs_path(self.src_path) and not check_obs_path(self.dst_path):
                    self._validate_file_integrity(self.src_path, self.dst_path)
        return self.dst_path

    def synchronize(self):
        if not self.path_pair.is_same():
            if not self.is_folder:
                raise ValueError("Only folder synchronization is available at present.")
        self.mox_module.file.sync_copy(self.src_path, self.dst_path)

    def check_path(self, path, existence_detection=False):
        if path is None:
            return False
        is_obs_path = check_obs_path(path)
        if is_obs_path and len(path.split("/", 3)) <= 3:
            return False
        if not is_valid_path(path[5:] if is_obs_path else path, self.is_folder):
            return False
        if existence_detection:
            if not self._is_existed_path(path, is_obs_path):
                return False
        return True

    def _is_existed_path(self, path, is_obs_path):
        if is_obs_path:
            return self.mox_module.file.exists(path)
        else:
            return os.path.exists(path)

    def _validate_folder_integrity(self, mox, src_path, dst_path):
        file_list_directory = mox.file.list_directory(src_path, recursive=True)
        for file in file_list_directory:
            file_path_in_obs = os.path.join(src_path, file)
            if mox.file.is_directory(file_path_in_obs):
                continue
            self._validate_file_integrity(file_path_in_obs, os.path.join(dst_path, file))

    def _validate_file_integrity(self, src_path, dst_path):
        def extract_bucket_name_and_object_key(obs_path):
            # 拆分路径为['obs:', '', bucket_name, object_key]
            path_split_result = str(obs_path).split('/', 3)
            if len(path_split_result) < 3:
                raise ValueError(f"Source path of {self.info} is not valid.")
            elif len(path_split_result) == 3:
                return path_split_result[2], None
            else:
                return path_split_result[2], path_split_result[3]

        def get_file_md5_from_obs(obs_object):
            md5_o = None
            header_list = obs_object.header
            for item in header_list:
                if str.lower(item[0]) in MD5_KEYWORD_LIST:
                    md5_o = item[1]
            return md5_o

        def calculate_file_md5(file_path):
            dig = hashlib.md5()
            with open(file_path, 'rb') as f:
                for data in iter(lambda: f.read(1024), b''):
                    dig.update(data)
            return dig.hexdigest()

        """validate the integrity of the file downloaded by moxing """
        obs_client = self.mox_module.file.file_io._create_or_get_obs_client()
        bucket_name, object_key = extract_bucket_name_and_object_key(src_path)
        metadata = obs_client.getObjectMetadata(bucket_name, object_key)
        md5_origin = get_file_md5_from_obs(metadata)
        if md5_origin is None:
            return
        file_md5 = calculate_file_md5(dst_path)
        if md5_origin != file_md5:
            raise RuntimeError(f"Integrity validation of {self.info} is failed.")


def check_obs_path(path):
    if not isinstance(path, str):
        return False
    return path.startswith("obs://")
