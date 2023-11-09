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
"""SAM Predict by Image"""
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

import mindspore as ms

from mindformers.pipeline import pipeline

sys.path.insert(0, os.getcwd())

ms.set_context(device_target="Ascend", device_id=0, mode=0)

def show_anns(anns):
    """show annos"""
    if not anns:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

image = cv2.imread("images/dog.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pipeline_task = pipeline("segment_anything", model='sam_vit_h')

# 9.全图分割: 默认参数, image传入cv2图像
masks = pipeline_task({"image": image}, seg_image=True)

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig('examples/9-全图分割_1.png')
print("9-全图分割: image传入cv2图像")

# 9.全图分割: 默认参数, image传入图像路径
masks = pipeline_task({"image": "images/dog.jpg"}, seg_image=True)

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig('examples/9-全图分割_2.png')
print("9-全图分割: image传入图像路径")

# 10.全图分割：调整参数
masks = pipeline_task({"image": image},
                      seg_image=True,
                      points_per_side=32,
                      pred_iou_thresh=0.86,
                      stability_score_thresh=0.92,
                      crop_n_layers=1,
                      crop_n_points_downscale_factor=2,
                      min_mask_region_area=100)

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig('examples/10-全图分割.png')
print("10-全图分割")
