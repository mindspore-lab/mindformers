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
"""
功能: 开启服务示例
"""
import json
import sys

from flask import Flask, request
from fm.fm_sdk import finetune

from fm.src.aicc_tools.ailog.log import service_logger

sys.path.append('./')
app = Flask(__name__)


@app.route('/example/path1/', methods=['POST'])
def finetune_api():
    headers = request.headers

    kwargs = {'scenario': headers.get('Scenario'), 'cert': headers.get('Cert'), 'app_config': headers.get('Appconfig')}

    output = finetune(**kwargs)

    return json.dumps(output)


@app.route('/example/path2/', methods=['GET', 'POST'])
def test():
    service_logger.info("test")
    name_ = request.values["name"]
    service_logger.info("param is " + name_)
    return name_


def run():
    service_logger.info("Init log system of aicc_tools tools.")
    # 若ModelArts当前版本不支持配置协议与端口，去除ssl_context参数配置，port需为8080
    app.run(host='xxx.xxx.xxx.xxx', port='xxx', ssl_context='adhoc')


if __name__ == '__main__':
    run()
