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
"""SAM Predict by Prompt"""
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

import mindspore as ms

from mindformers.pipeline import pipeline

sys.path.insert(0, os.getcwd().split('research')[0])

ms.set_context(device_target="Ascend", device_id=0, mode=0)

def show_mask(input_mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = input_mask.shape[-2:]
    mask_image = input_mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(bbox, ax):
    x0, y0 = bbox[0], bbox[1]
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

pipeline_task = pipeline("segment_anything", model='sam_vit_h')

image = cv2.imread("images/truck.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 0.使用pipeline_task.set_image提前抽取图像特征
pipeline_task.set_image(image)

# 1. 单点确定一个物体
input_point = np.array([[500, 375]])
input_label = np.array([1])
outputs = pipeline_task({"points": input_point,
                         "labels": input_label},
                        multimask_output=True)
masks = outputs["masks"]
scores = outputs["iou_predictions"]
logits_single = outputs["low_res_masks"]

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.savefig(f"examples/1-单点确定一个物体_{i}.png")
print(f"1-单点确定一个物体")

# 2.两点确定相同物体
input_point = np.array([[500, 375], [1125, 625]])
input_label = np.array([1, 1])
outputs = pipeline_task({"points": input_point,
                         "labels": input_label},
                        multimask_output=False)
masks = outputs["masks"]
scores = outputs["iou_predictions"]
logits = outputs["low_res_masks"]

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.savefig("examples/2-两点确定相同物体.png")
print(f"2-两点确定相同物体")

# 3.两点确定不同物体
input_point = np.array([
        [[500, 375]],
        [[1125, 625]],
    ])
input_label = np.array([[1], [1]])
outputs = pipeline_task({"points": input_point,
                         "labels": input_label},
                        multimask_output=False)
masks = outputs["masks"]
scores = outputs["iou_predictions"]
logits = outputs["low_res_masks"]

plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    show_mask(mask, plt.gca(), random_color=True)
show_points(input_point.reshape(-1, 2), input_label.reshape(-1), plt.gca())
plt.axis('off')
plt.savefig("examples/3-两点确定不同物体.png")
print(f"3-两点确定不同物体")

# 4.一个前景点和背景点
input_point = np.array([[500, 375], [1125, 625]])
input_label = np.array([1, 0])
outputs = pipeline_task({"points": input_point,
                         "labels": input_label},
                        multimask_output=False)
masks = outputs["masks"]
scores = outputs["iou_predictions"]
logits = outputs["low_res_masks"]

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.savefig("examples/4-一个前景点和背景点.png")
print(f"4-一个前景点和背景点")

# 5.单框确定一个物体
input_box = np.array([425, 600, 700, 875])
outputs = pipeline_task({"boxes": input_box},
                        multimask_output=False)
masks = outputs["masks"]
scores = outputs["iou_predictions"]
logits = outputs["low_res_masks"]

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks[0], plt.gca())
show_box(input_box, plt.gca())
plt.axis('off')
plt.savefig("examples/5-单框确定一个物体.png")
print(f"5-单框确定一个物体")

# 6.框和背景点确定物体
input_box = np.array([425, 600, 700, 875])
input_point = np.array([[575, 750]])
input_label = np.array([0])
outputs = pipeline_task({"points": input_point,
                         "labels": input_label,
                         "boxes": input_box},
                        multimask_output=False)
masks = outputs["masks"]
scores = outputs["iou_predictions"]
logits = outputs["low_res_masks"]

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks[0], plt.gca())
show_box(input_box, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.savefig("examples/6-框和背景点确定物体.png")
print(f"6-框和背景点确定物体")

# 7.多组框和点确定不同物体
input_boxes = np.array([[425, 600, 700, 875],
                        [1360, 525, 1680, 780]])
input_points = np.array([[[575, 750]],
                         [[1525, 670]]])
input_labels = np.array([[1], [1]])
outputs = pipeline_task({"points": input_points,
                         "labels": input_labels,
                         "boxes": input_boxes},
                        multimask_output=False)
masks = outputs["masks"]
scores = outputs["iou_predictions"]
logits = outputs["low_res_masks"]

plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    show_mask(mask, plt.gca(), random_color=True)
for box in input_boxes:
    show_box(box, plt.gca())
for point, label in zip(input_points, input_labels):
    show_points(point, label, plt.gca())
plt.axis('off')
plt.savefig("examples/7-多组框和点确定不同物体.png")
print(f"7-多组框和点确定不同物体")

# 8.多个框确定不同物体
input_boxes = np.array([
        [75, 275, 1725, 850],
        [425, 600, 700, 875],
        [1375, 550, 1650, 800],
        [1240, 675, 1400, 750],
    ])
outputs = pipeline_task({"boxes": input_boxes},
                        multimask_output=False)
masks = outputs["masks"]
scores = outputs["iou_predictions"]
logits = outputs["low_res_masks"]

plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    show_mask(mask, plt.gca(), random_color=True)
for box in input_boxes:
    show_box(box, plt.gca())
plt.axis('off')
plt.savefig("examples/8-多个框确定不同物体.png")
print(f"8-多个框确定不同物体")

# 单点确定一个物体: 传入cv2图像和prompt
image = cv2.imread("images/truck.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

input_point = np.array([[500, 375]])
input_label = np.array([1])
outputs = pipeline_task({"image": image,
                         "points": input_point,
                         "labels": input_label},
                        multimask_output=True)
masks = outputs["masks"]
scores = outputs["iou_predictions"]
logits_single = outputs["low_res_masks"]

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.savefig(f"examples/9-单点确定一个物体_{i}.png")
print(f"9-单点确定一个物体: 传入cv2图像和prompt")

# 单点确定一个物体: 传入图像路径和prompt
input_point = np.array([[500, 375]])
input_label = np.array([1])
outputs = pipeline_task({"image": "images/truck.jpg",
                         "points": input_point,
                         "labels": input_label},
                        multimask_output=True)
masks = outputs["masks"]
scores = outputs["iou_predictions"]
logits_single = outputs["low_res_masks"]

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.savefig(f"examples/10-单点确定一个物体_{i}.png")
print(f"10-单点确定一个物体: 传入图像路径和prompt")
