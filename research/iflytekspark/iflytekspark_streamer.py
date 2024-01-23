# Copyright 2024 Huawei Technologies Co., Ltd
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
# This file was refer to project:
# https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py
# ============================================================================
"""Streamers for text generation."""
from multiprocessing import Queue
from typing import Optional
from mindformers.models.base_tokenizer import BaseTokenizer
from mindformers.generation.streamers import TextIteratorStreamer


__all__ = ['IFlytekSparkStreamer']


class IFlytekSparkStreamer(TextIteratorStreamer):
    """iFlytekSpark inference streamer."""
    def __init__(self,
                 tokenizer: Optional[BaseTokenizer] = None,
                 skip_prompt: bool = False,
                 timeout: Optional[float] = None,
                 **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        # variables used in the streaming process
        self.token_cache = []
        self.cache_index = 0
        self.text_index = 0
        self.real_text = ""
        self.real_len = []

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        token_list = value.copy()
        if not self.token_cache:
            self.token_cache = [[] for _ in range(len(token_list))]
            self.ready = [False for _ in range(len(token_list))]

        for i in range(len(token_list)): # batch size
            if token_list[i]:
                self.token_cache[i].append(token_list[i][0]) # 每个iteration过来都会append进来，token_cache为batch size个[id123]
                if token_list[i][0] > 294 or token_list[i][0] < 39: # can decode, recode last token is ready or not
                    self.ready[i] = True
                else:
                    self.ready[i] = False
            else:
                self.ready[i] = True

        if sum(self.ready) == len(self.ready):
            for i in range(len(self.token_cache)):
                if len(self.token_cache[i]) > 1 and self.token_cache[i][-1] == 5: # eos_token_id
                    while len(self.token_cache[i]) > 1:
                        if 39 <= self.token_cache[i][-2] <= 294:
                            self.token_cache[i].pop(-2)
                        else:
                            break

            generated_text = self.tokenizer.batch_decode(self.token_cache, skip_special_tokens=True)
            self.on_finalized_text(generated_text, stream_end=False)
            self.token_cache = []

    def end(self):
        self.on_finalized_text("", stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)
        else:
            self.text_queue.put(text, timeout=self.timeout)
