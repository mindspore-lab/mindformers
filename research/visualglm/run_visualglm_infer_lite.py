# Copyright 2023 Huawei Technologies Co., Ltd
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
"""visualglm inference runner with mindspore lite. """

import argparse
import re
import time

import numpy as np
import mindspore as ms
from mindspore.dataset import vision
from mindspore.dataset.vision.utils import Inter
from mindformers.inference import InferConfig, InferTask
from mindformers.tools.utils import str2bool
from mindformers.tools.image_tools import load_image

from visualglm import VisualGLMImageToTextGeneration
from visualglm_config import VisualGLMConfig
from visualglm_processor import VisualGLMProcessor
from visualglm_text_generator_infer import register_task


def init_context(device_id):
    ms.set_context(mode=0, device_target="Ascend", device_id=device_id, max_device_memory="40GB")  # Ascend, CPU


def build_text_input(prompts, templates):
    text_input = []
    for i in range(len(prompts)):
        text_input.append(templates[i].format(prompts[i]))
    return text_input


def process_response(response_list):
    """ get standard response"""
    handled_response = []
    for response in response_list:
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            [r"\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        response = response.split('答：')[-1].strip()
        handled_response.append(response)
    return handled_response


DEFAULT_IMAGE_TEXT_PAIR = [
    ("./examples/titanic.jpg", "这部电影的导演是谁？")
]


def generate_glm_prompt(unhandled_prompts, history=None, english=False):
    """ generate glm prompt from raw prompt. """
    if history is None:
        history = []
    post_prompts, image_positions = [], []
    for query in unhandled_prompts:
        prompt = "</img>"
        if english:
            for _, (old_query, response) in enumerate(history):
                prompt += f"Q:{old_query}\nA:{response}\n"
            prompt += f"Q:{query}\nA:"
        else:
            for _, (old_query, response) in enumerate(history):
                prompt += f"问：{old_query}\n答：{response}\n"
            prompt += f"问：{query}\n答："
        post_prompts.append(prompt)
    pre_prompts = ["<img>"] * len(post_prompts)
    image_positions = [len("<img>")] * len(post_prompts)
    return pre_prompts, post_prompts, image_positions


def handle_prompt(args):
    """ handle prompt from args """
    if args.image_path is None:
        image_filepath = [pair[0] for pair in DEFAULT_IMAGE_TEXT_PAIR]
    else:
        image_filepath = args.image_path.split(',')

    if args.prompt is None:
        if args.image_path is not None:
            raw_prompts = [""] * len(image_filepath)
        else:
            raw_prompts = [pair[1] for pair in DEFAULT_IMAGE_TEXT_PAIR]
    else:
        raw_prompts = args.prompt.split(',')

    if len(raw_prompts) != len(image_filepath):
        raise ValueError("prompts length do not equal to image_path length, please check the args.")

    # handle prompt using chatglm type
    pre_prompts, post_prompts, image_positions = generate_glm_prompt(raw_prompts)

    return image_filepath, pre_prompts, post_prompts, image_positions


def infer_with_lite(model, lite_model, tokenizer, input_images, pre_input_ids, post_input_ids):
    """ inference with lite model """
    start_pre_time = time.time()
    text_input_ids, projected_qformer_output, pre_input_ids, post_input_ids = model(input_images,
                                                                                    pre_input_ids,
                                                                                    post_input_ids)
    model_kwargs = {}
    model.llm_model.update_model_kwargs_before_generate(input_ids=text_input_ids.asnumpy(),
                                                        model_kwargs=model_kwargs)

    pre_inputs = model.llm_model.prepare_inputs_for_generation(input_ids=text_input_ids.asnumpy(),
                                                               image_embeds=projected_qformer_output,
                                                               pre_input_ids=pre_input_ids.asnumpy(),
                                                               post_input_ids=post_input_ids.asnumpy(),
                                                               **model_kwargs)
    pre_inputs.pop("input_ids")
    text_input_ids = text_input_ids.asnumpy()
    valid_length = np.max(np.argwhere(np.array(text_input_ids[0]) != tokenizer.pad_token_id)) + 1
    end_pre_time = time.time()
    print(f"preprocess time: {end_pre_time - start_pre_time}")
    output, output_ids, generate_time = lite_model.infer(inputs=text_input_ids[:, :valid_length],
                                                         pad_token_id=tokenizer.pad_token_id,
                                                         eos_token_id=tokenizer.eos_token_id,
                                                         **pre_inputs)

    response = process_response(output)
    print("Response:")
    print(response)
    print("\n")
    total_time = generate_time + (end_pre_time - start_pre_time)
    generate_len = sum(len(output_ids[i]) - text_input_ids[i].tolist().index(tokenizer.bos_token_id)
                       for i in range(len(output_ids)))
    print(f"visualglm generate time: {generate_time},"
          f" total time: {total_time}, generate_len: {generate_len}",
          f" total rate: {generate_len / total_time}")


def load_glm_model(args, tokenizer):
    """ load glm lite model """
    lite_config = InferConfig(
        prefill_model_path=args.full_model_path,
        increment_model_path=args.inc_model_path,
        model_type="mindir",
        model_name="visualglm",
        ge_config_path=args.ge_config,
        device_id=args.device_id,
        infer_seq_length=args.seq_length + 32,
    )
    print("start load >>>>>>>>>>>>>>>>>>>")
    infer_task = InferTask.get_infer_task("visualglm_generation", lite_config, tokenizer=tokenizer)
    print("end load >>>>>>>>>>>>>>>>>>>")

    return infer_task


def main(args):
    init_context(device_id=args.device_id)
    model_config = VisualGLMConfig.from_pretrained(args.config_path)
    model_config.max_txt_len = args.seq_length

    register_task()

    if args.checkpoint is not None:
        print(f"checkpoint: {args.checkpoint}")
        model_config.checkpoint_name_or_path = args.checkpoint

    image_filepath, pre_prompts, post_prompts, image_positions = handle_prompt(args)

    if args.batch_size > 1:
        model_config.batch_size = args.batch_size

        diff = model_config.batch_size - len(image_filepath)
        if diff > 0:
            extend_filepath = [image_filepath[-1]] * diff
            extend_pre_prompt = [pre_prompts[-1]] * diff
            extend_post_prompt = [post_prompts[-1]] * diff
            extend_positions = [image_positions[-1]] * diff
            image_filepath.extend(extend_filepath)
            pre_prompts.extend(extend_pre_prompt)
            post_prompts.extend(extend_post_prompt)
            image_positions.extend(extend_positions)
        else:
            image_filepath = image_filepath[:model_config.batch_size]
            pre_prompts = pre_prompts[:model_config.batch_size]
            post_prompts = post_prompts[:model_config.batch_size]
            image_positions = image_positions[:model_config.batch_size]
    else:
        model_config.batch_size = 1

    model_config.text_config.batch_size = model_config.batch_size
    model_config.text_config.seq_length = args.seq_length + model_config.qformer_config.query_length
    model_config.text_config.do_sample = args.do_sample
    model_config.text_config.top_p = args.top_p
    model_config.text_config.top_k = args.top_k
    model_config.text_config.use_past = args.use_past

    model = VisualGLMImageToTextGeneration(model_config)
    processor = VisualGLMProcessor.from_pretrained(args.config_path)
    processor.image_processor.resize.resize = vision.transforms.Resize((224, 224), Inter.BICUBIC)

    tokenizer = processor.tokenizer

    glm_model = load_glm_model(args, tokenizer)

    for _ in range(args.generate_repeat_time):
        if model_config.batch_size > 1:
            input_images = processor.image_processor([load_image(filepath) for filepath in image_filepath])
            pre_input_ids = tokenizer(pre_prompts, add_special_tokens=False, return_tensors="ms")["input_ids"]
            post_input_ids = tokenizer(post_prompts,
                                       max_length=args.seq_length - len(pre_input_ids[0]),
                                       padding="max_length",
                                       return_tensors="ms")["input_ids"]
            infer_with_lite(model, glm_model, tokenizer, input_images, pre_input_ids, post_input_ids)

        else:
            batch_size = len(image_filepath)
            for index in range(batch_size):
                pil_image = load_image(image_filepath[index])
                input_image = processor.image_processor(pil_image)
                pre_input_ids = tokenizer(pre_prompts[index], add_special_tokens=False, return_tensors="ms")[
                    "input_ids"]
                post_input_ids = tokenizer(post_prompts[index],
                                           max_length=args.seq_length - len(pre_input_ids),
                                           padding="max_length",
                                           return_tensors="ms")["input_ids"]
                infer_with_lite(model, glm_model, tokenizer, input_image, pre_input_ids, post_input_ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="visualglm_6b", type=str, required=False, help='model type')
    parser.add_argument('--config_path', default="run_visualglm_6b_image_to_text_generation.yaml",
                        type=str, required=False, help='config path')
    parser.add_argument('--device_id', type=int, default=0, required=False, help='device id')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch_size')
    parser.add_argument('--checkpoint', type=str, default=None, required=False, help='checkpoint path')
    parser.add_argument('--generate_repeat_time', type=int, default=2, required=False, help='generate repeat time')
    parser.add_argument('--use_past', type=str2bool, default=True, required=False, help='whether use past')
    parser.add_argument('--do_sample', type=str2bool, default=False, required=False, help='whether do sample')
    parser.add_argument('--top_p', type=float, default=1, required=False, help='top p')
    parser.add_argument('--top_k', type=int, default=0, required=False, help='top k')
    parser.add_argument('--seq_length', type=int, default=32, required=False, help='seq length')
    parser.add_argument('--image_path', type=str, default=None, required=False, help='image path')
    parser.add_argument('--prompt', type=str, default='', required=False, help='prompt content')
    parser.add_argument('--full_model_path', default=None, type=str, help="load mindir full checkpoint")
    parser.add_argument('--inc_model_path', default=None, type=str, help="load mindir inc checkpoint")
    parser.add_argument('--ge_config', type=str, required=False, help="Path to GE config")
    args_ = parser.parse_args()
    print(args_)
    main(args_)
