# Copyright 2025 Huawei Technologies Co., Ltd
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
"""mcore deepseek modelslim w8a8 model ST of inference"""
import argparse
import os

from transformers import AutoTokenizer

from mindspore.nn.utils import no_init_parameters
from mindformers import MindFormerBook
from mindformers import AutoModel, MindFormerConfig, build_context
from mindformers.core.parallel_config import build_parallel_config


def test_deepseek_a8w8_predict_mcore():
    """
    Feature: Infer interface of deepseek a8w8.
    Description: Test mcore interface for prediction.
    Expectation: AssertionError
    """
    config_path = os.path.join(MindFormerBook.get_project_path(), "configs", "deepseek3",
                               "predict_deepseek3_671b.yaml")
    load_safetensors = "/home/workspace/mindspore_dataset/weight/DeepSeek-R1-W8A8"

    max_decode_length = 10

    config = MindFormerConfig(config_path)
    config.context.max_device_memory = '28GB'
    config.pretrained_model_dir = load_safetensors
    config.parallel_config.model_parallel = 2
    build_context(config)
    build_parallel_config(config)

    config.load_checkpoint = load_safetensors
    with no_init_parameters():
        network = AutoModel.from_config(config)
        network.load_weights(config.load_checkpoint)

    build_context(config)
    tokenizer = AutoTokenizer.from_pretrained(load_safetensors)

    batch_datas = [{"prompt": "介绍下北京故宫", "answer": "博物院ODాలు SER비스"}]
    for batch_data in batch_datas:
        input_ids = tokenizer(batch_data["prompt"])["input_ids"]
        outputs = network.generate(input_ids,
                                   max_length=max_decode_length,
                                   do_sample=False)
        generated_text = tokenizer.decode(outputs[0])
        print("predict answer is:", generated_text, flush=True)
        assert batch_data["answer"] in generated_text

def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--quant_algo', '-a', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    uargs = get_args()
    quant_algo = uargs.quant_algo
    if quant_algo == "A8W8":
        os.environ['MS_INTERNAL_ENABLE_NZ_OPS'] = "QuantBatchMatmul,MlaPreprocess,Mla"
        test_deepseek_a8w8_predict_mcore()
