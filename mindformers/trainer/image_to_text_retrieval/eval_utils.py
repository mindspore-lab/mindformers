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
"""Image-to-text Retrieval Trainer Utils."""
import numpy as np
from mindspore import Tensor

from mindformers.tools.logger import logger
from mindformers.generation.utils import topk
from mindformers.dataset.dataloader.multi_image_cap_dataloader import MultiImgCapDataLoader

def extract_image_text_mapping(eval_dataloader, score_i2t, score_t2i):
    """extract_image_text_mapping from eval_dataloader.

    Args:
        eval_dataloader: evaluation dataloader
        score_i2t, score_t2i: two score matrix (I2T, T2I)

    Returns:
        img2txt, txt2img: ground truth image-text mapping
    """
    dataloader = eval_dataloader
    while hasattr(dataloader, "children") and dataloader.children is not None:
        dataloader = dataloader.children[0]
    if isinstance(dataloader, MultiImgCapDataLoader):
        dataset = dataloader.source
        return dataset.img2txt, dataset.txt2img
    logger.warning("expect the eval dataset to be generate \
        from MultiImgCapDataLoader, but is %s, will generate \
        image-text mapping with accumulate indexes by default.", type(dataloader))
    assert (score_i2t.shape[0], score_i2t.shape[1]) == (score_t2i.shape[1], score_t2i.shape[0])
    image_num = score_i2t.shape[0]
    text_num = score_t2i.shape[0]
    if image_num == text_num:
        inds = [i for i in range(image_num)]
        return inds, inds
    bigger = image_num if image_num > text_num else text_num
    smaller = text_num if bigger == image_num else image_num
    factor = bigger // smaller
    smaller_inds = [[min(i + k * factor, bigger) for i in range(factor)] for k in range(smaller)]
    bigger_inds = []
    for i in range(smaller - 1):
        bigger_inds += [i] * factor
    last_num = bigger - (smaller - 1) * factor
    bigger_inds += [smaller - 1] * last_num
    if bigger == image_num:
        return bigger_inds, smaller_inds
    return smaller_inds, bigger_inds

def compute_itm_scores(network, eval_inputs, k_test=128, add_extra_itm_score=False):
    """
    compute image-text matching scores, in matrix format.

    Args:
        network (BaseModel): network for evaluate ITM
        eval_inputs (tuple): inputs for evaluate itm scores.
        k_test (int, optional): k_test num, Defaults to 128.
        add_extra_itm_score (bool, optional): whether to add extra scores (model decides), Defaults to False.

    Returns:
        score_matrix_i2t, score_matrix_t2i: two score matrix (I2T, T2I)
    """
    logger.info("========= k_text num: %d =========", k_test)
    sims_matrix = []
    image_feats, text_feats, extra_args = eval_inputs[0], eval_inputs[1], eval_inputs[2:]
    image_feats = image_feats.asnumpy()
    text_feats = text_feats.asnumpy()
    for image_feat in image_feats:
        print(image_feat.shape, image_feat.dtype, text_feats.T.shape, text_feats.dtype)
        sim_q2t = np.matmul(image_feat, text_feats.T)
        sim_i2t = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = np.stack(sims_matrix, axis=0)

    score_matrix_i2t = np.full(
        (image_feats.shape[0], text_feats.shape[0]), -100.0
    )

    for i, sims in enumerate(sims_matrix):
        if i % 50 == 0:
            print(f"I2T: {i}/{sims_matrix.shape[0]} - sims.shape: {sims.shape}")
        topk_sim, topk_idx = topk(sims.T, k_test)
        topk_sim = topk_sim.T
        topk_idx = topk_idx.T
        score_matrix_i2t[i, topk_idx] = topk_sim # .astype(mstype.float32)
        if add_extra_itm_score:
            topk_idx_ms = Tensor.from_numpy(topk_idx)
            score = network.compute_extra_itm(extra_args, i, k_test, topk_idx_ms, i2t=True)
            score_matrix_i2t[i, topk_idx] += score.asnumpy()

    sims_matrix = sims_matrix.T
    score_matrix_t2i = np.full(
        (text_feats.shape[0], image_feats.shape[0]), -100.0
    )

    for i, sims in enumerate(sims_matrix):
        if i % 50 == 0:
            print(f"T2I: {i}/{sims_matrix.shape[0]} - sims.shape: {sims.shape}")
        topk_sim, topk_idx = topk(sims.T, k_test)
        topk_sim = topk_sim.T
        topk_idx = topk_idx.T
        score_matrix_t2i[i, topk_idx] = topk_sim
        if add_extra_itm_score:
            topk_idx_ms = Tensor.from_numpy(topk_idx)
            score = network.compute_extra_itm(extra_args, i, k_test, topk_idx_ms, i2t=False)
            score_matrix_t2i[i, topk_idx] += score.asnumpy()

    return score_matrix_i2t, score_matrix_t2i
