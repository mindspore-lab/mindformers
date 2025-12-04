"""Unit tests for the base pipeline helpers."""

import pytest

from mindformers.pipeline.base_pipeline import Pipeline
from mindformers.pipeline import base_pipeline as base_pipeline_module


class _SaveableComponent:
    """Simple helper capturing save_pretrained calls."""

    def __init__(self):
        self.calls = []

    def save_pretrained(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class DummyPipeline(Pipeline):
    """Lightweight Pipeline implementation for white-box tests."""

    def __init__(self):
        # Skip the heavy super().__init__ and wire up the minimal attributes needed
        # by the helper methods under test.
        self.model = _SaveableComponent()
        self.tokenizer = None
        self.feature_extractor = None
        self.image_processor = None
        self.network = None
        self._preprocess_params = {"base_pre": 0}
        self._forward_params = {"base_fw": 0}
        self._postprocess_params = {"base_post": 0}
        self.call_count = 0
        self._batch_size = None
        self.records = []

    def _sanitize_parameters(self, **pipeline_parameters):
        # Mirror the real contract: return tuple of (preprocess, forward, postprocess) overrides
        return (
            pipeline_parameters.get("preprocess_params", {}),
            pipeline_parameters.get("forward_params", {}),
            pipeline_parameters.get("postprocess_params", {}),
        )

    def preprocess(self, inputs, **preprocess_params):
        payload = {"kind": "preprocess", "inputs": inputs, "params": preprocess_params}
        self.records.append(payload)
        return payload

    def _forward(self, model_inputs, **forward_params):
        payload = {"kind": "forward", "inputs": model_inputs, "params": forward_params}
        self.records.append(payload)
        return payload

    def postprocess(self, model_outputs, **postprocess_params):
        payload = {"kind": "postprocess", "inputs": model_outputs, "params": postprocess_params}
        self.records.append(payload)
        # run_multi expects run_single to return an iterable so wrap the payload in a list
        return [payload]


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_call_with_scalar_merges_params_and_updates_call_count():
    """
    Feature: __call__ merging logic
    Description: Verify scalar inputs merge default/sanitized params and bump call_count.
    Expectation: Preprocess receives merged dict, forward/postprocess mirror sanitized data.
    """
    pipe = DummyPipeline()
    result = pipe(
        "single",
        preprocess_params={"extra": 1},
        forward_params={"fw_extra": 2},
        postprocess_params={"post_extra": 3},
    )

    assert pipe.call_count == 1
    assert pipe.records[0]["params"] == {"base_pre": 0, "extra": 1}
    assert pipe.records[1]["params"] == {"base_fw": 0, "fw_extra": 2}
    assert pipe.records[2]["params"] == {"base_post": 0, "post_extra": 3}
    assert result[0]["kind"] == "postprocess"


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_call_with_list_invokes_run_multi_with_default_batch_size():
    """
    Feature: __call__ list dispatch
    Description: Ensure list inputs route through run_multi using configured batch_size.
    Expectation: run_multi observes provided inputs and default batch size when none passed.
    """

    class SpyPipeline(DummyPipeline):
        """Capture arguments passed to run_multi for verification."""

        def __init__(self):
            super().__init__()
            self.multi_args = None

        def run_multi(self, inputs, batch_size, preprocess_params, forward_params_unused, postprocess_params_unused):
            del forward_params_unused
            del postprocess_params_unused
            self.multi_args = {
                "inputs": inputs,
                "batch_size": batch_size,
                "preprocess": preprocess_params,
            }
            return ["multi"]

    pipe = SpyPipeline()
    pipe.batch_size = 3
    outcome = pipe([1, 2, 3], preprocess_params={"l": 1})

    assert outcome == ["multi"]
    assert pipe.multi_args == {
        "inputs": [1, 2, 3],
        "batch_size": 3,
        "preprocess": {"base_pre": 0, "l": 1},
    }
    assert pipe.call_count == 1


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_run_single_executes_pipeline_stages():
    """
    Feature: Pipeline stage orchestration
    Description: Ensure `run_single` invokes preprocess/forward/postprocess sequentially.
    Expectation: Stage records follow the expected order and final output originates from postprocess.
    """
    pipe = DummyPipeline()
    result = pipe.run_single(
        inputs="sample",
        preprocess_params={"prep": 1},
        forward_params={"fw": 2},
        postprocess_params={"post": 3},
    )

    assert result[0]["kind"] == "postprocess"
    kinds = [entry["kind"] for entry in pipe.records]
    assert kinds == ["preprocess", "forward", "postprocess"]


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_run_multi_batches_and_validates_length():
    """
    Feature: Batch execution logic
    Description: Validate `run_multi` chunks inputs correctly and enforces batch divisibility.
    Expectation: Outputs accumulate per batch and ValueError raised when length mismatch occurs.
    """
    pipe = DummyPipeline()
    outputs = pipe.run_multi(
        inputs=[1, 2, 3, 4],
        batch_size=2,
        preprocess_params={},
        forward_params={},
        postprocess_params={},
    )
    assert len(outputs) == 2

    with pytest.raises(ValueError):
        pipe.run_multi([1, 2, 3], batch_size=2, preprocess_params={}, forward_params={}, postprocess_params={})


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_call_with_dataset_handles_batching(monkeypatch):
    """
    Feature: Dataset path handling
    Description: Simulate GeneratorDataset inputs to ensure batching and iteration logic execute.
    Expectation: Dataset batches with provided size, outputs cover all batches, tqdm is bypassed for speed.
    """

    class FakeBatchDataset:
        def __init__(self, data, batch_size):
            self.data = data
            self.batch_size = batch_size

        def create_dict_iterator(self, do_copy=False):
            del do_copy
            for idx in range(0, len(self.data), self.batch_size):
                yield {"chunk": tuple(self.data[idx:idx + self.batch_size])}

    class FakeDataset:
        def __init__(self, data):
            self.data = data
            self.batched_with = None

        def batch(self, batch_size):
            self.batched_with = batch_size
            return FakeBatchDataset(self.data, batch_size)

    monkeypatch.setattr(base_pipeline_module, "GeneratorDataset", FakeDataset)
    monkeypatch.setattr(base_pipeline_module, "BatchDataset", FakeBatchDataset)
    monkeypatch.setattr(base_pipeline_module, "RepeatDataset", FakeBatchDataset)
    monkeypatch.setattr(base_pipeline_module, "tqdm", lambda data, **_: data)

    dataset = FakeDataset([1, 2, 3, 4])
    pipe = DummyPipeline()
    outputs = pipe(dataset, batch_size=2)

    assert len(outputs) == 2
    assert dataset.batched_with == 2


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batch_size_property_validation():
    """
    Feature: Batch size setter guardrails
    Description: `batch_size` must accept positive integers only.
    Expectation: Valid integers are stored; invalid types or negatives raise ValueError.
    """
    pipe = DummyPipeline()
    pipe.batch_size = 4
    assert pipe.batch_size == 4

    with pytest.raises(ValueError):
        pipe.batch_size = -1
    with pytest.raises(ValueError):
        pipe.batch_size = 1.5


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_save_pretrained_invokes_all_components(tmp_path):
    """
    Feature: save_pretrained delegation
    Description: Ensure model/tokenizer/image helpers receive save requests and file paths are created.
    Expectation: Each component's `save_pretrained` is called once; early exit occurs when path is a file.
    """
    pipe = DummyPipeline()
    pipe.tokenizer = _SaveableComponent()
    pipe.feature_extractor = _SaveableComponent()
    pipe.image_processor = _SaveableComponent()

    target_dir = tmp_path / "pipeline"
    pipe.save_pretrained(str(target_dir), save_name="custom")

    assert pipe.model.calls[0][0][0] == str(target_dir)
    assert pipe.tokenizer.calls[0][0][0] == str(target_dir)
    assert target_dir.exists()

    # When a file path is provided, the helper should short-circuit without invoking save_pretrained.
    file_path = tmp_path / "model.bin"
    file_path.write_text("stub")
    pipe.model.calls.clear()
    pipe.save_pretrained(str(file_path))
    assert not pipe.model.calls


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_transform_and_predict_delegate_to_call():
    """
    Feature: Compatibility helpers
    Description: Ensure `transform` and `predict` are thin proxies over `__call__`.
    Expectation: Calls are forwarded verbatim and return value propagated.
    """

    class ProxyPipeline(DummyPipeline):
        def __init__(self):
            super().__init__()
            self.calls = []

        def __call__(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "delegated"

    pipe = ProxyPipeline()
    assert pipe.transform("x") == "delegated"
    assert pipe.predict(data="y") == "delegated"
    assert pipe.calls == [(("x",), {}), ((), {"data": "y"})]
