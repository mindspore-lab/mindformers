import io
import json
import os
from flask import Flask, request
from caption import opt_caption_inference

APP = Flask(__name__)

# 加载模型
model_path = os.path.split(__file__)[0]
model_name = "opt_caption_graph.mindir"
vocab_name = "vocab.json"
service_object = opt_caption_inference(model_path, model_name, vocab_name)


@APP.route('/infer/image', methods=['POST'])
def infer_image_func():
    print("begin infer image")
    file_data = request.files['file']
    byte_stream = io.BytesIO(file_data.read())
    print("get infer image data")
    input_data = {"instances": {"image": byte_stream}}
    inference_result = service_object.inference(input_data)
    print("get infer output", inference_result)
    res_data = {
        'inference_result': inference_result
    }
    return json.dumps(res_data, indent=4, ensure_ascii=False)


# Host IP应该为"0.0.0.0", port可以为8080
if __name__ == '__main__':
    # 若ModelArts当前版本不支持配置协议与端口，去除ssl_context参数配置，port需为8080
    APP.run(host="0.0.0.0", port=8080, ssl_context='adhoc')
