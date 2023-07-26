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
"""
Run chat web demo.

If running on remote server, please run this command before:
export GRADIO_SERVER_NAME=0.0.0.0
"""
import argparse
import re
from threading import Thread
import mdtex2html
import mindspore as ms
import gradio as gr

from mindformers import AutoModel, AutoTokenizer, TextIteratorStreamer, AutoConfig, logger

parser = argparse.ArgumentParser()
parser.add_argument('--device_target', default="Ascend", type=str,
                    help='The target device to run, support "Ascend" and "CPU". Default: Ascend.')
parser.add_argument('--device_id', default=0, type=int,
                    help='Which device to run service. Default: 0.')
parser.add_argument('--model', default='glm_6b_chat', type=str,
                    help='Which model to generate text. Default: glm_6b_chat.')
parser.add_argument('--tokenizer', default='glm_6b', type=str,
                    help='Which tokenizer to tokenize text. Default: glm_6b.')
parser.add_argument('--max_length', default=1024, type=int,
                    help='Max output length. Default: 1024')
parser.add_argument('--use_past', default=False, type=bool,
                    help='Whether to enable incremental inference. Default: False')
parser.add_argument('--port', default=None, type=int,
                    help='Which port to run the service. Default: None')
args = parser.parse_args()

ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
config = AutoConfig.from_pretrained(args.model)
config.use_past = args.use_past
model = AutoModel.from_config(config)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True)
# pre-build the network
sample_input = tokenizer("hello")
sample_output = model.generate(sample_input["input_ids"], max_length=10)


def process_response(response):
    """process response"""
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
    return response


class Chatbot(gr.Chatbot):
    """Chatbot with overrode postprocess method"""
    def postprocess(self, y):
        """postprocess"""
        if y is None:
            return []
        for i, (message, response) in enumerate(y):
            y[i] = (
                None if message is None else mdtex2html.convert(message),
                None if response is None else mdtex2html.convert(response),
            )
        return y


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(inputs, bot, history):
    """predict"""
    logger.info("Received user input: %s", inputs)
    bot.append((parse_text(inputs), ""))

    if inputs == "":
        inputs = "你好"

    input_tokens = tokenizer(inputs)
    logger.info("Convert user input to tokens: %s", input_tokens)
    generation_kwargs = dict(input_ids=input_tokens["input_ids"],
                             streamer=streamer,
                             max_length=args.max_length,
                             top_k=1,
                             repetition_penalty=1.5)
    if args.model.startswith("glm"):
        generation_kwargs.pop("repetition_penalty")
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    output = ""
    for response in streamer:
        output += response
        new_history = history + [(inputs, output)]

        bot[-1] = (parse_text(inputs), parse_text(output))

        yield bot, new_history
    logger.info("Generate output: %s", output)


def reset_user_input():
    """reset user input"""
    return gr.update(value='')


def reset_state():
    """reset state"""
    return [], []


with gr.Blocks() as demo:
    gr.HTML(f"""<h1 align="center">Chat Web Demo powered by {args.model}</h1>""")
    chatbot = Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            empty_btn = gr.Button("Clear")

    chat_history = gr.State([])

    submit_btn.click(predict, [user_input, chatbot, chat_history], [chatbot, chat_history],
                     show_progress=True)
    submit_btn.click(reset_user_input, [], [user_input])

    empty_btn.click(reset_state, outputs=[chatbot, chat_history], show_progress=True)

demo.queue().launch(server_port=args.port)
