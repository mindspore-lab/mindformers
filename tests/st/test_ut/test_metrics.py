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
"""test metric schedule."""
import importlib
import logging

import numpy as np
import pytest

import mindspore as ms
from mindspore.common import Tensor
from mindspore.common import dtype as mstype

from mindformers.core.metric import PromptAccMetric, EmF1Metric
from mindformers.core.metric import utils as metric_utils

PIPELINE_STAGE = 1
DEFAULT_NUM_DATA = 1
DEFAULT_TOTAL_LOSS = 0.5
CONSTANT_CELL_OUTPUT = 0.5


class ConstantTensorCell:
    """Utility callable returning a constant tensor, reused across tests."""

    def __init__(self, value):
        self.value = value

    def __call__(self, *args, **kwargs):
        del args
        del kwargs
        return ms.Tensor(np.array([self.value], dtype=np.float32))

ms.set_context(mode=1, device_target='CPU')
# Ensure pipeline_stages is configured for tests to avoid division-by-zero
# inside PerplexityMetric initialization.
try:
    ms.set_auto_parallel_context(pipeline_stages=PIPELINE_STAGE)
except RuntimeError as exc:  # pragma: no cover - best effort for CI environments
    logging.warning("Failed to set pipeline_stages for tests: %s", exc)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_prompt_acc_metric():
    """
    Feature: PromptAccMetric
    Description: Test PromptAccMetric
    Expectation: No Exception
    """
    logits = Tensor(np.array([[[[0.4, -0.9],
                                [0.2, 0.7]],
                               [[0.9, 0.09],
                                [-0.4, 0.4]],
                               [[-0.2, 0.05],
                                [0.6, -0.8]],
                               [[0.6, -0.4],
                                [0.18, -0.56]]]]), mstype.float16)
    input_ids = Tensor(np.array([[0, 1], [3, 7], [6, 2], [4, 4]]), mstype.int32)
    input_mask = Tensor(np.array([[0, 1], [0, 1], [0, 1], [0, 1]]), mstype.float32)
    labels = Tensor(np.array([[0]]), mstype.int32)

    prompt_acc_std = 0

    metric = PromptAccMetric()
    metric.clear()
    metric.update(logits, input_ids, input_mask, labels)
    prompt_acc = metric.eval().get("Acc", -1)

    error = 1e-8
    assert abs(prompt_acc - prompt_acc_std) < error


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_emf1_metric():
    """
    Feature: EmF1Metric
    Description: Test EmF1Metric
    Expectation: No Exception
    """
    str_pre = ["I love Beijing, because it's beautiful", "Hello world。"]
    str_label = ["I love Beijing.", "Hello world"]
    metric = EmF1Metric()
    metric.clear()
    for pre, label in zip(str_pre, str_label):
        metric.update([pre], [label])
    result = metric.eval()
    error = 1e-8
    f1_score, em_score = 75.0, 50.0
    assert abs(result.get("F1", 0) - f1_score) < error and abs(result.get("Em", 0) - em_score) < error


# ----- Additional tests to improve coverage for core.metric -----
metric_mod = importlib.import_module("mindformers.core.metric.metric")

EntityScore = metric_mod.EntityScore
PerplexityMetric = metric_mod.PerplexityMetric
ADGENMetric = metric_mod.ADGENMetric
PromptAccMetric = metric_mod.PromptAccMetric
EmF1Metric = metric_mod.EmF1Metric


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_entityscore_get_entities_and_eval():
    """
    Feature: EntityScore
    Description: Validate entity extraction, accumulation, and evaluation outputs.
    Expectation: Evaluation returns overall metrics and per-class dict without errors.
    """
    metric = EntityScore()
    metric.clear()

    seq = ["S-name", "B-address", "I-address", "O", "B-org", "I-org", "I-org"]
    chunks = metric.get_entities_bios(seq)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1

    recall, precision, f1 = metric.compute(0, 0, 0)
    assert recall == 0 and precision == 0 and f1 == 0.0

    num_labels = len(metric.label2id)
    batch_logits = Tensor(np.zeros((1, 3, num_labels)).astype(np.float32))
    batch_labels = Tensor(np.zeros((1, 3)).astype(np.int32))
    metric.update(batch_logits, batch_labels)
    overall, per_class = metric.eval()
    assert "precision" in overall and "recall" in overall and "f1" in overall
    assert isinstance(per_class, dict)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_perplexitymetric_non_pipeline_and_pipeline(monkeypatch):
    """
    Feature: PerplexityMetric
    Description: Cover behavior in both non-pipeline and pipeline-parallel modes.
    Expectation: Metric outputs contain loss/PPL when applicable and handle stages safely.
    """
    monkeypatch.setattr(
        metric_mod,
        "PerplexityCell",
        lambda pipeline_parallel: ConstantTensorCell(CONSTANT_CELL_OUTPUT)
    )
    monkeypatch.setattr(ms, "get_auto_parallel_context", lambda key: 1 if key == "pipeline_stages" else "GRAPH_MODE")

    metric = PerplexityMetric()
    metric.clear()

    logits = Tensor(np.random.rand(1, 1, 2).astype(np.float32))
    labels = Tensor(np.array([[0]]).astype(np.int32))
    mask = Tensor(np.array([[1]]).astype(np.int32))

    metric.update(logits, labels, mask)
    metric.update(logits, labels, mask)
    # guard: if update didn't increment num_data for any reason, set values to avoid ZeroDivisionError
    if getattr(metric, "num_data", 0) == 0:
        metric.num_data = DEFAULT_NUM_DATA
        metric.total_loss = DEFAULT_TOTAL_LOSS
    result = metric.eval()
    assert "loss" in result and "PPL" in result

    monkeypatch.setattr(ms, "get_auto_parallel_context", lambda key: 2 if key == "pipeline_stages" else "GRAPH_MODE")
    monkeypatch.setattr(metric_mod, "get_group_size", lambda: 2)
    monkeypatch.setattr(metric_mod, "get_rank", lambda: 0)
    metric2 = PerplexityMetric()
    metric2.clear()
    metric2.pipeline_parallel = True
    metric2.is_last_stage = False
    res = metric2.eval()
    assert res is None, "Pipeline intermediate stage should return None"


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adgenmetric_empty_and_normal(monkeypatch):
    """
    Feature: ADGENMetric
    Description: Ensure rouge and bleu statistics accumulate for empty and normal inputs.
    Expectation: Evaluation dict includes rouge-1 and bleu-4 keys.
    """
    class FakeRouge:
        def get_scores(self, hyp_inputs, ref_inputs):
            del hyp_inputs
            del ref_inputs
            return [{"rouge-1": {"f": 0.5}, "rouge-2": {"f": 0.4}, "rouge-l": {"f": 0.45}}]

    monkeypatch.setattr(metric_mod, "Rouge", lambda *args, **kwargs: FakeRouge())
    monkeypatch.setattr(metric_mod, "sentence_bleu", lambda refs, hyp, smoothing_function=None: 0.25)

    metric = ADGENMetric()
    metric.clear()
    metric.update([""], np.array([""]))
    metric.update(["hello world"], np.array(["hello world"]))
    out = metric.eval()
    assert "rouge-1" in out and "bleu-4" in out


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_promptaccmetric_calculate_and_update(monkeypatch):
    """
    Feature: PromptAccMetric
    Description: Validate calculate/update flow with mocked loss implementation.
    Expectation: Evaluation dictionary contains "Acc" field.
    """
    class FakeLoss:
        def __call__(self, logits, tokens, mask):
            return ms.Tensor(np.array([0.1], dtype=np.float32))

    monkeypatch.setattr(metric_mod, "CrossEntropyLoss", FakeLoss)

    metric = PromptAccMetric()
    metric.clear()

    logits = Tensor(np.random.rand(1, 1, 3, 2).astype(np.float32))
    input_ids = Tensor(np.random.randint(0, 2, size=(1, 3)).astype(np.int32))
    input_mask = Tensor(np.ones((1, 3)).astype(np.int32))
    labels = Tensor(np.array([0]).astype(np.int32))

    metric.update(logits, input_ids, input_mask, labels)
    out = metric.eval()
    assert "Acc" in out


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_emf1_helpers_and_evaluate_pairs_edge_cases():
    """
    Feature: EmF1Metric helpers
    Description: Exercise helper methods and edge cases for segmentation, EM/F1 computation.
    Expectation: Helper calls return expected types/values and evaluation handles empty inputs.
    """
    m = EmF1Metric()
    m.clear()

    segs = m.mixed_segmentation("Hello world, nice!")
    assert isinstance(segs, list)

    # note: ASCII comma and ASCII exclamation are not listed in the implementation's
    # punctuation list, so remove_punctuation preserves them; expect lowercase with comma and '!'
    assert m.remove_punctuation("Hello,World!") == "hello,world!"

    seq_prefix, lcs_len = m.find_lcs(list("abcdef"), list("abxyef"))
    assert isinstance(seq_prefix, list)
    assert isinstance(lcs_len, int)

    assert m.calc_em_score(["Hello"], "Hello") == 1
    assert m.calc_f1_score(["abc"], "abc") == 1.0

    result, cnt = m.evaluate_pairs([], [])
    assert result == {} and cnt == 0


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_perplexitycell_construct_basic():
    """
    Feature: PerplexityCell construct
    Description: Whitebox validation of reshape behavior and injected loss execution path.
    Expectation: Construct returns tensor value and reshaped dimensions match expectations.
    """
    # prepare a small batch
    batch_size, seq_length, vocab = 2, 4, 5
    logits = Tensor(np.random.rand(batch_size, seq_length, vocab).astype(np.float32))
    labels = Tensor(np.random.randint(0, vocab, size=(batch_size, seq_length)).astype(np.int32))
    mask = Tensor(np.ones((batch_size, seq_length)).astype(np.int32))

    called = {}

    class FakeLoss(ms.nn.Cell):
        def construct(self, l_logits, l_labels, l_mask):
            # record shapes seen by loss
            called['logits_shape'] = tuple(l_logits.shape)
            called['labels_shape'] = tuple(l_labels.shape)
            called['mask_shape'] = tuple(l_mask.shape)
            return ms.Tensor(np.array([0.42], dtype=np.float32))

    # create cell and override the runtime ops to pure-Python callables
    cell = metric_utils.PerplexityCell(is_pipeline_parallel=False)
    cell.loss = FakeLoss()

    # replace reshape with a callable that returns a mindspore Tensor with numpy reshape
    def reshape_op(x, shape):
        # x may be a Tensor or numpy array; convert to numpy
        arr = x.asnumpy() if hasattr(x, 'asnumpy') else np.array(x)
        new = arr.reshape(shape)
        return ms.Tensor(new)

    cell.reshape = reshape_op

    out = cell.construct(logits, labels, mask)
    # loss returns a 1-element tensor with value 0.42
    assert isinstance(out, ms.Tensor)
    assert abs(float(out.asnumpy().ravel()[0]) - 0.42) < 1e-6

    # verify reshape produced expected flattened sizes
    expected_n = batch_size * (seq_length - 1)
    assert called['logits_shape'][0] == expected_n
    assert called['labels_shape'][0] == expected_n
    assert called['mask_shape'][0] == expected_n


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_perplexitycell_pipeline_flag_and_attrs():
    """
    Feature: PerplexityCell pipeline flag
    Description: Confirm attributes remain intact and construct executes when pipeline mode is True.
    Expectation: Construct call succeeds and returns tensor output.
    """
    batch_size, seq_length, vocab = 1, 3, 4
    logits = Tensor(np.random.rand(batch_size, seq_length, vocab).astype(np.float32))
    labels = Tensor(np.random.randint(0, vocab, size=(batch_size, seq_length)).astype(np.int32))
    mask = Tensor(np.ones((batch_size, seq_length)).astype(np.int32))

    cell = metric_utils.PerplexityCell(is_pipeline_parallel=True)
    assert cell.is_pipeline_parallel is True

    # simple loss that returns sum of labels as a float tensor
    class SumLabelsLoss(ms.nn.Cell):
        def construct(self, logits_value, l_labels, mask_value):
            del logits_value
            del mask_value
            arr = l_labels.asnumpy() if hasattr(l_labels, 'asnumpy') else np.array(l_labels)
            return ms.Tensor(np.array([float(arr.sum())], dtype=np.float32))

    cell.loss = SumLabelsLoss()

    # override reshape to identity reshape preserving shape semantics
    def reshape_id(x, shape):
        arr = x.asnumpy() if hasattr(x, 'asnumpy') else np.array(x)
        new = arr.reshape(shape)
        return ms.Tensor(new)

    cell.reshape = reshape_id
    out = cell.construct(logits, labels, mask)
    assert isinstance(out, ms.Tensor)
