#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能:
版权信息: 华为技术有限公司, 版权所有(C) 2010-2022
"""
import hashlib
import json
import time
import io
from flask import Flask, request
from werkzeug.utils import secure_filename
from lenet import Mnist

APP = Flask(__name__)

MODEL_NAME = 'lenet.ckpt'
MODEL_PATH = '/home/mind/model'

SERVICE = Mnist(MODEL_NAME, MODEL_PATH)


@APP.route('/health', methods=['GET'])
def health_func():
    return json.dumps({'health': 'true'}, indent=4)


@APP.route('/infer/image', methods=['POST'])
def infer_image_func():
    file_data = request.files['file']
    byte_stream = io.BytesIO(file_data.read())
    input_data = {'1': {'mnist': byte_stream}}
    preprocessed_result = Mnist.preprocess(input_data)
    inference_result = SERVICE.inference(preprocessed_result)
    inference_result = Mnist.postprocess(inference_result)

    res_data = {
        'inference_result': str(inference_result)
    }

    return json.dumps(res_data, indent=4)


# Host IP应该为"0.0.0.0", port可以为8080
if __name__ == '__main__':
    # 若ModelArts当前版本不支持配置协议与端口，去除ssl_context参数配置，port需为8080
    APP.run(host="0.0.0.0", port=8080, ssl_context='adhoc')
