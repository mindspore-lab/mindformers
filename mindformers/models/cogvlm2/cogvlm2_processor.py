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
# ============================================================================
"""
CogVLM2Processor
"""
from typing import Optional, Union, List, Dict, Any

import PIL
import PIL.Image
import cv2
import numpy as np
import mindspore as ms

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.dataset.transforms.vision_transforms import (
    BatchPILize,
    BatchToTensor,
    BatchNormalize, BatchCenterCrop
)
from mindformers.models.image_processing_utils import BaseImageProcessor

from ..multi_modal import ModalContentTransformTemplate, BaseTextContentBuilder
from ..multi_modal.base_multi_modal_processor import BatchResizeV2
from ..multi_modal.modal_content import ModalContentBuilder
from ..multi_modal.utils import DataRecord

video_format_list = ('.mp4', '.avi', '.flv', '.mpeg', '.f4v', '.mkv')


class VideoProcessor:
    """
    Process input video data with opencv-python.
    """

    def __init__(self,
                 num_frames=1,
                 pad_frames=False,
                 frames_per_sec=None):
        self.num_frames = num_frames
        self.pad_frames = pad_frames
        self.frames_per_sec = frames_per_sec

    def __call__(self, video_file_path):
        frames = self.cv_video_to_image(video_file_path)
        return frames

    # pylint: disable=W0640
    def cv_video_to_image(self, video_path):
        """Extract input video frames with strategy."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_nums = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        seconds = [_ / fps for _ in range(frame_nums)]
        max_second = round(max(seconds)) + 1

        if self.frames_per_sec is None:
            frame_ids = []
            for sec in range(max_second):
                closest_num = min(seconds, key=lambda x: abs(x - sec))
                index = seconds.index(closest_num)
                frame_ids.append(index)
                if len(frame_ids) >= self.num_frames:
                    break

        elif isinstance(self.frames_per_sec, int) and self.frames_per_sec > 0:
            sec_idx = []
            for sec in range(max_second):
                closest_num = min(seconds, key=lambda x: abs(x - sec))
                index = seconds.index(closest_num)
                sec_idx.append(index)
            frame_ids = []
            is_full = False
            for idx in range(len(sec_idx) - 1):
                start_idx = sec_idx[idx]
                end_idx = sec_idx[idx + 1]
                for _ in range(self.frames_per_sec):
                    fetch_idx = (_ / self.frames_per_sec) * (end_idx - start_idx) + start_idx
                    frame_ids.append(fetch_idx)
                    if len(frame_ids) >= self.num_frames:
                        is_full = True
                        break
                if is_full:
                    break

        else:
            raise ValueError(f"unsupported input strategy frames_per_sec: {self.frames_per_sec}.")

        frames = []
        for idx in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError("video frame corruption.")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) < self.num_frames and self.pad_frames:
            last_frame = frames[-1]
            frames += [last_frame for _ in range(self.num_frames - len(frames))]

        return np.array(frames)


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class CogVLM2ImageProcessor(BaseImageProcessor):
    """
    CogVLM2 ImageProcessor.

    Args:
        image_size (int): The target size.
    """

    def __init__(self,
                 image_size: Optional[int] = 224,
                 interpolation: Optional[str] = 'bicubic',
                 mean=None,
                 std=None,
                 is_hwc=False,
                 **kwargs):
        self.pilize = BatchPILize()
        super().__init__(**kwargs)
        if isinstance(image_size, int):
            image_size = (image_size,) * 2
        self.resize = BatchResizeV2(image_size[0], interpolation=interpolation)
        self.to_tensor = BatchToTensor()
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = BatchNormalize(mean, std, is_hwc)
        self.center_crop = BatchCenterCrop(image_size)

    def preprocess(self, images: Union[ms.Tensor, PIL.Image.Image, np.ndarray, List[PIL.Image.Image]], **kwargs):
        r"""
        Preprocess Required By Base Processor.

        Args:
            images (ms.Tensor, PIL.Image, numpy.array, List[PIL.Image]): A batch of images.

        Return:
            A 4-rank tensor for a batch of images.
        """
        images = self.pilize(images)
        images = self.resize(images)
        images = self.center_crop(images)
        images = self.to_tensor(images)
        images = self.normalize(images)

        kwargs.pop("other", None)
        if isinstance(images, list):
            return ms.Tensor(np.row_stack([np.expand_dims(item, axis=0) for item in images]))
        if len(images.shape) == 4:
            return ms.Tensor(images)
        return ms.Tensor(np.expand_dims(images, axis=0))

    @staticmethod
    def _bhwc_check(image_batch: Union[ms.Tensor, PIL.Image.Image, np.ndarray, List[PIL.Image.Image]]):
        r"""Bhwc_check"""
        if isinstance(image_batch, np.ndarray):
            if image_batch.shape[-1] == 3:
                return True
        if isinstance(image_batch, ms.Tensor):
            if image_batch.asnumpy().shape[-1] == 3:
                return True
        if isinstance(image_batch, (list, PIL.Image.Image)):
            return True
        return False


class CogVLM2VideoContentBuilder(ModalContentBuilder):
    """
    CogVLM2 Video Content Builder.
    """

    def __init__(
            self,
            context_pad_token,
            context_length,
            use_custom_token=True,
            start_token=None,
            end_token=None,
            tokenizer=None,
            num_frames=24,
            pad_frames=False,
            frames_per_sec=None,
            image_size=224
    ):
        super(CogVLM2VideoContentBuilder, self).__init__(
            type_="video",
            context_pad_token=context_pad_token,
            context_length=context_length,
            use_custom_token=use_custom_token,
            start_token=start_token,
            end_token=end_token,
            tokenizer=tokenizer
        )

        self.num_frames = 0
        self.vision_token_type = 0
        self.language_token_type = 1

        self.video_reader = VideoProcessor(num_frames=num_frames,
                                           pad_frames=pad_frames,
                                           frames_per_sec=frames_per_sec)
        self.image_processor = CogVLM2ImageProcessor(image_size)

    # pylint: disable=W0613
    def regular_input_for_predict(self, inputs: Dict, result_recorder: DataRecord = None, **kwargs):
        """regular input for predict."""
        video_path = inputs["video"]
        images = self.video_reader(video_path)
        images = self.image_processor(images)
        self.num_frames = images.shape[0]
        text = self.start_token + self.end_token
        result_recorder.put("images", images)
        return text

    def regular_input_for_train(self, inputs, result_recorder: DataRecord = None, **kwargs):
        """regular input for train."""
        raise NotImplementedError

    # pylint: disable=W0613
    def padding_context(self, input_ids, result_recorder: DataRecord = None, **kwargs):
        """Pad text input with pad_token_id."""
        start_position = np.where(input_ids == self.start_token_id)[0]

        if start_position.size == 0:
            return input_ids

        context_input_ids = [self.tokenizer.bos_token_id]
        token_type_ids = [self.language_token_type]
        position_ids = [0]

        video_token_ids = []
        types = []
        cur_position = 1
        for time_index in range(self.num_frames):
            video_token_ids += [self.context_pad_token_id] * self.context_length

            types += [self.vision_token_type] * self.context_length

            time_indices = self.tokenizer(str(time_index), add_special_tokens=False)["input_ids"]
            len_time_indices = len(time_indices)

            video_token_ids += time_indices
            types += [self.language_token_type] * len_time_indices

            position_ids += [cur_position] + [cur_position + 1] * (self.context_length - 2) + \
                            [cur_position + 2] + [cur_position + 3 + i for i in range(len_time_indices)]
            cur_position += 3 + len_time_indices

        context_input_ids += video_token_ids
        token_type_ids += types

        input_ids = np.insert(input_ids, [start_position[0]], context_input_ids)

        input_ids = np.delete(input_ids, np.where(input_ids == self.start_token_id))
        input_ids = np.delete(input_ids, np.where(input_ids == self.end_token_id))

        position_ids = np.array(position_ids)
        padding_length = len(input_ids) - len(position_ids)
        position_ids = np.pad(position_ids, (0, padding_length), 'linear_ramp',
                              end_values=(0, cur_position + padding_length - 1))

        result_recorder.put("position_ids", position_ids)

        return input_ids


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class CogVLM2ContentTransformTemplate(ModalContentTransformTemplate):
    """
    CogVLM2 Modal Content Transform Template.
    """

    # pylint: disable=W1113
    def __init__(self, output_columns, tokenizer, mode="predict", signal_type="base", *args, **kwargs):
        super().__init__(output_columns=output_columns, tokenizer=tokenizer, mode=mode, *args, **kwargs)
        self.signal_type = signal_type

        self.modal_builders = {
            "video": CogVLM2VideoContentBuilder("<|reserved_special_token_2|>", 66, use_custom_token=False,
                                                start_token="<|reserved_special_token_3|>",
                                                end_token="<|reserved_special_token_4|>"),
            "text": BaseTextContentBuilder(),
        }

    # pylint: disable=W0613
    def build_conversation_input_text(self, raw_inputs, result_recorder):
        """Build conversion input text."""
        if self.signal_type == "base":
            return "".join(raw_inputs)

        if self.signal_type == "vqa":
            answer_format = "Short answer:"
        elif self.signal_type == "chat":
            answer_format = "Answer:"
        else:
            raise ValueError(f"Unknown signal type {self.signal_type}")

        prompt = ""
        for _, (old_query, response) in enumerate(self.history):
            prompt += f"Question: {old_query} {answer_format} {response}\n"

        video_context_placeholder = (f"{self.modal_builders['video'].start_token}"
                                     f"{self.modal_builders['video'].end_token}")
        if video_context_placeholder in raw_inputs:
            video_query_index = raw_inputs.index(video_context_placeholder)
            video_query = raw_inputs.pop(video_query_index)
            prompt = f"{video_query}{prompt}"

        query = "".join(raw_inputs)
        prompt += f"Question: {query} {answer_format}"
        return prompt

    def get_need_update_output_items(self, result: DataRecord) -> Dict[str, Any]:
        """Update result before output."""
        update_items = {}
        if not result.has_key("position_ids"):
            position_ids = [i for i in range(len(result.get("input_ids")))]
            update_items["position_ids"] = position_ids

        return update_items

    def batch(self, data_list, token_padding_length, **kwargs):
        """Get batched input data."""
        batched_data = super().batch(data_list, token_padding_length, **kwargs)

        position_ids_list = batched_data.get("position_ids")
        padded_position_ids_list = []
        for position_ids in position_ids_list:
            padding_length = token_padding_length - len(position_ids)
            if padding_length > 0:
                padding_start = position_ids[-1]
                position_ids = np.pad(position_ids, (0, padding_length), 'linear_ramp',
                                      end_values=(0, padding_start + padding_length))
            else:
                position_ids = position_ids[:token_padding_length]

            padded_position_ids_list.append(position_ids)
        batched_data["position_ids"] = ms.Tensor(np.stack(padded_position_ids_list, axis=0), ms.int32)
        return batched_data

    # pylint: disable=W0613
    def post_process(self, output_ids, **kwargs):
        """Post process model output."""
        output = []
        for output_ids_item in output_ids:
            decoded = self.tokenizer.decode(output_ids_item, skip_special_tokens=True)
            if self.signal_type == "vqa":
                processed = decoded.split("Short answer:")[-1]
            elif self.signal_type == "chat":
                processed = decoded.split("Answer:")[-1]
            else:
                processed = decoded
            output.append(processed)
        return output
