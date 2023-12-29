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
"""eval longbench metrics"""
import json
import argparse
import jieba
from rouge import Rouge


def rouge_score(prediction, ground_truth):
    """compute the rouge score"""
    rouge = Rouge()
    scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth):
    """compute chinese version rouge score"""
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    score = rouge_score(prediction, ground_truth)
    return score


def scorer(predictions, answers):
    r"""
    compute the rouge result

    Args:
       predictions (List): the predicted result
       answers (List): the ground truth
    """
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        for ground_truth in ground_truths:
            score = max(score, rouge_zh_score(prediction, ground_truth))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def compute_metrics(input_file):
    r"""
    compute the metrics

    Args:
       input_file (str): The input dataset file
    """
    predictions, answers, lengths = [], [], []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            predictions.append(data["pred"])
            answers.append(data["answers"])
            if "length" in data:
                lengths.append(data["length"])
    score = scorer(predictions, answers)
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_file',
                        default='/path/merged/dureader.jsonl',
                        type=str, help="predict_file")
    opt_para = parser.parse_args()

    res_score = compute_metrics(opt_para.predict_file)
    print("evaluate score is: ", res_score)
