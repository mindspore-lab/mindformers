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
"""
import argparse
import time
from threading import Thread
import mdtex2html
import mindspore as ms
import gradio as gr

from mindformers import AutoModel, AutoTokenizer, TextIteratorStreamer, AutoConfig, logger


def get_model_and_tokenizer(model_config, tokenizer_name):
    """Get model and tokenizer instance"""
    return AutoModel.from_config(model_config), AutoTokenizer.from_pretrained(tokenizer_name)


parser = argparse.ArgumentParser()
parser.add_argument('--device_target', default="Ascend", type=str, choices=['Ascend', 'CPU'],
                    help='The target device to run, support "Ascend" and "CPU". Default: Ascend.')
parser.add_argument('--device_id', default=0, type=int, help='Which device to run service. Default: 0.')
parser.add_argument('--model', type=str, help='Which model to generate text.')
parser.add_argument('--tokenizer', type=str, help='Which tokenizer to tokenize text.')
parser.add_argument('--checkpoint_path', type=str, default=None, help='The path of model checkpoint.')
parser.add_argument('--seq_length', default="512", type=int, help="Sequence length of the model. Default: 512.")
parser.add_argument('--use_past', default=False, type=bool,
                    help='Whether to enable incremental inference. Default: False.')
parser.add_argument('--host', default="127.0.0.1", type=str,
                    help="Which host ip to run the service. Default: 127.0.0.1.")
parser.add_argument('--port', default=None, type=int, help='Which port to run the service. Default: None.')
args = parser.parse_args()

ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
config = AutoConfig.from_pretrained(args.model)
config.seq_length = args.seq_length
config.use_past = args.use_past
if args.checkpoint_path:
    config.checkpoint_name_or_path = args.checkpoint_path
logger.info("Config: %s", config)
model, tokenizer = get_model_and_tokenizer(config, args.tokenizer)
streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True)

# pre-build the network
sample_input = tokenizer("hello")
sample_output = model.generate(sample_input["input_ids"], max_length=10)

prompt_examples = [["Below is an instruction that describes a task. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n{}\n\n### Response:"],
                   ["A chat between a curious user and an artificial intelligence assistant. The assistant gives "
                    "helpful,"
                    "detailed, and polite answers to user\'s questions. USER: {} ASSISTANT: "],
                   ["Assume you are a dog, you must response \"Woofs\" at first whatever any instruction\n\n"
                    "### Instruction:\n{}\n\n### Response:"],
                   ["<human>: {} \n<bot>: "],
                   ["问：{}\n\n答："]]

is_multi_round = args.model.startswith("glm2")
default_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the " \
                 "request.\n\n### Instruction:\n{}\n\n### Response:" if args.model.startswith("llama_7b_lora") else ""
default_prompt = "问：{}\n\n答：" if args.model.startswith("glm2") else ""

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
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def build_prompt(inputs, prompt):
    """Build prompt for inputs"""
    if prompt == "":
        return inputs
    if prompt.find("{}") != -1:
        return prompt.format(inputs)
    raise gr.Error("The prompt is invalid! Please make sure your prompt contains placeholder '{}' to replace user "
                   "input.")


def build_multi_round(inputs, history):
    """Build multi-round prompt for inputs"""
    prev_rounds = ""
    if args.model.startswith("glm2"):
        for i, (query, response) in enumerate(history):
            prev_rounds += "[Round {}]\n\n{}{}\n\n".format(i, query, response)
        prev_rounds += "[Round {}]\n\n".format(len(history))
    return prev_rounds + inputs


def generate(**kwargs):
    """generate function with timer and exception catcher"""
    gen_model = kwargs.pop("model")
    try:
        start_time = time.time()
        gen_model.generate(**kwargs)
        end_time = time.time()
        logger.info("Total time: %.2f s", (end_time-start_time))

    # pylint: disable=W0703
    except Exception as e:
        logger.error(repr(e))
        streamer.text_queue.put("<ERROR>")


def predict(inputs, bot, history, do_sample, top_k, top_p, temperature, repetition_penalty, max_length, prompt):
    """predict"""
    if inputs == "":
        raise gr.Error("Input cannot be empty!")

    logger.info("Received user input: %s", inputs)
    bot.append((parse_text(inputs), ""))

    prompted_input = build_prompt(inputs, prompt)
    logger.info("Prompt: %s", prompt)
    logger.info("User input after prompted: %s", prompted_input)

    round_prompted_input = build_multi_round(prompted_input, history)
    logger.info("User input after multi-round prompted: %s", round_prompted_input)

    input_tokens = tokenizer(round_prompted_input)
    if max_length > args.seq_length:
        raise gr.Error("Max length must be set smaller than the sequence length of model! The model sequence length "
                       "is {}.".format(args.seq_length))
    if max_length <= len(input_tokens["input_ids"]):
        raise gr.Error("Max length must be larger than the length of input tokens! The current length of input "
                       "tokens is {}.".format(len(input_tokens["input_ids"])))
    generation_kwargs = dict(model=model,
                             input_ids=input_tokens["input_ids"],
                             streamer=streamer,
                             do_sample=do_sample,
                             top_k=top_k,
                             top_p=top_p,
                             temperature=temperature,
                             repetition_penalty=repetition_penalty,
                             max_length=max_length)
    thread = Thread(target=generate, kwargs=generation_kwargs)
    logger.info("Start to generate text. Generate args: {input_ids: %s, do_sample: %s, top_k: %s, top_p: %s, "
                "temperature: %s, repetition_penalty: %s, max_length: %s}",
                input_tokens["input_ids"], do_sample, top_k, top_p, temperature, repetition_penalty, max_length)
    thread.start()

    interval_time = 0.
    output = ""
    for response in streamer:
        if response == "<ERROR>":
            raise gr.Error("An error occurred! Please make sure all the settings are correct!")
        output += response
        new_history = history + [(prompted_input, output)]

        bot[-1] = (parse_text(inputs), parse_text(output))

        interval_start = time.time()
        yield bot, new_history
        interval_end = time.time()
        interval_time = interval_time + interval_end - interval_start

    logger.info("Generate output: %s", output)
    logger.info("Web communication time: %.2f s", interval_time)


def reset_user_input():
    """reset user input"""
    return gr.update(value='')


def reset_state():
    """reset state"""
    return [], []


def set_do_sample_args(do_sample):
    return {top_k_slider: gr.update(visible=do_sample),
            top_p_slider: gr.update(visible=do_sample),
            temp_number: gr.update(visible=do_sample)}


with gr.Blocks() as demo:
    gr.HTML(f"""<h1 align="center">Chat Web Demo powered by MindFormers</h1>""")
    with gr.Row():
        with gr.Column(scale=10):
            with gr.Group():
                chatbot = gr.Chatbot(label=fr"Chatbot {args.model}")
                user_input = gr.Textbox(show_label=False, placeholder="Ask something...", lines=5)
            with gr.Row():
                with gr.Column(scale=6):
                    submit_btn = gr.Button("Submit", variant="primary")
                with gr.Column(scale=6):
                    empty_btn = gr.Button("Clear")
        with gr.Column(scale=2):
            with gr.Group():
                do_sample_checkbox = gr.Checkbox(value=config.do_sample if config.do_sample else False,
                                                 label="sampling", info="Whether to sample on the candidate ids")
                top_k_slider = gr.Slider(value=config.top_k if config.top_k else 1,
                                         label="top k", maximum=10, minimum=0, step=1,
                                         visible=config.do_sample if config.do_sample else False,
                                         info="Determine the topK numbers token id as candidate")
                top_p_slider = gr.Slider(value=config.top_p if config.top_p else 1,
                                         label="top p", maximum=1, minimum=0.01, step=0.01,
                                         visible=config.do_sample if config.do_sample else False,
                                         info="The accumulation probability of the candidate token ids, "
                                              "below the top_p will be select as the candidate ids")
                temp_number = gr.Number(value=config.temperature if config.temperature else 1,
                                        label="temperature",
                                        visible=config.do_sample if config.do_sample else False,
                                        info="The value used to modulate the next token probabilities")
                rp_number = gr.Number(value=config.repetition_penalty if config.repetition_penalty else 1,
                                      label="repetition penalty", minimum=0,
                                      info="The penalty factor of the frequency that generated words")
                max_len_number = gr.Number(value=config.max_length if config.max_length else 512, minimum=0,
                                           label="max length", info="The maximum length of the generated words")
                prompt_input = gr.Textbox(value=default_prompt, label="prompt", placeholder="No prompt...",
                                          info="Add prompt to input")
                gr.Examples(prompt_examples, inputs=prompt_input, label="Prompt example")

    chat_history = gr.State([])

    submit_btn.click(predict,
                     [user_input, chatbot, chat_history, do_sample_checkbox, top_k_slider, top_p_slider, temp_number,
                      rp_number, max_len_number, prompt_input],
                     [chatbot, chat_history],
                     show_progress=True)
    submit_btn.click(reset_user_input, [], [user_input])

    empty_btn.click(reset_state, outputs=[chatbot, chat_history], show_progress=True)

    do_sample_checkbox.change(set_do_sample_args, [do_sample_checkbox], [top_k_slider, top_p_slider, temp_number])

if __name__ == "__main__":
    demo.queue().launch(server_name=args.host, server_port=args.port)
