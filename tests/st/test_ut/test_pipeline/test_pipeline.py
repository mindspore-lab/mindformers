"""Unit tests covering the public pipeline entry points."""

import importlib
import sys
import types

import pytest

# Import the pipeline module object (the file `mindformers/pipeline/pipeline.py`).
# The previous line `from mindformers.pipeline import pipeline` imports the
# `pipeline` symbol (a function) exported by the package, not the module file.
pipeline_module = importlib.import_module("mindformers.pipeline.pipeline")


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_experimental_mode_various_cases(tmp_path):
    """
    Feature: experimental mode detection
    Description: Verify `is_experimental_mode` correctly distinguishes between
                 string repo identifiers, local directories, non-string models,
                 and raises when experimental-only kwargs are present.
    Expectation: Returns True for repo-like strings and directories, False for
                 non-string model instances, and raises ValueError when illegal
                 experimental kwargs are passed with a plain model string.
    """
    # non-string model instance -> not experimental
    assert pipeline_module.is_experimental_mode(model=123) is False

    # a string with a slash (repo name) and not starting with 'mindspore' -> experimental
    assert pipeline_module.is_experimental_mode("owner/repo") is True

    # a local directory path -> experimental
    d = tmp_path / "model_dir"
    d.mkdir()
    assert pipeline_module.is_experimental_mode(str(d)) is True

    # model string without slash + experimental-only kw should raise
    with pytest.raises(ValueError):
        pipeline_module.is_experimental_mode("model_name", config=1)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_clean_custom_task_transform_ms_attribute():
    """
    Feature: custom task cleaning
    Description: Verify `clean_custom_task` transforms a string entry in the
                 `ms` field into the corresponding attribute/object from the
                 `mindformers` package.
    Expectation: Returns a cleaned dict where `ms` is a tuple of resolved
                 attributes from the injected `mindformers` module.
    """
    # prepare a fake mindformers module with an attribute we can resolve
    dummy_mod = types.SimpleNamespace()

    class DummyClass:
        pass

    setattr(dummy_mod, "MyDummy", DummyClass)

    # inject into sys.modules so that import inside function will pick it up
    sys.modules["mindformers"] = dummy_mod

    try:
        task_info = {"impl": "irrelevant", "ms": "MyDummy"}
        cleaned, _ = pipeline_module.clean_custom_task(task_info)
        assert isinstance(cleaned, dict)
        assert isinstance(cleaned["ms"], tuple)
        # The resolved item should be the DummyClass we provided
        assert cleaned["ms"][0] is DummyClass
    finally:
        # remove our injected module to avoid side effects for other tests
        del sys.modules["mindformers"]


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_clean_custom_task_missing_impl_raises():
    """
    Feature: custom task validation
    Description: `clean_custom_task` must fail when `impl` key is missing.
    Expectation: Raises RuntimeError.
    """
    with pytest.raises(RuntimeError):
        pipeline_module.clean_custom_task({})


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_load_model_success_and_failure_paths():
    """
    Feature: dynamic model loading
    Description: Exercise `load_model` code paths where the first candidate
                 model class succeeds, and where all candidates fail.
    Expectation: Returns instantiated model for successful class; raises
                 ValueError if all classes fail to load.
    """

    # Successful loader class
    class SuccessModel:
        def __init__(self):
            self._ok = True

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    cfg = types.SimpleNamespace(architectures=[])
    model = pipeline_module.load_model("some-id", cfg, model_classes=(SuccessModel,), task="t")
    assert getattr(model, "_ok", False) is True

    # All failing loader classes -> ValueError expected
    class Fail1:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise OSError("fail1")

    class Fail2:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise ValueError("fail2")

    with pytest.raises(ValueError):
        pipeline_module.load_model("some-id", cfg, model_classes=(Fail1, Fail2), task="t")


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_ms_pipeline_invalid_task_raises_keyerror():
    """
    Feature: pipeline task validation
    Description: `get_ms_pipeline` should raise if task is not registered in
                 `SUPPORT_PIPELINES`.
    Expectation: Raises KeyError for invalid task names.
    """
    with pytest.raises(KeyError):
        pipeline_module.get_ms_pipeline("nonexistent_task", None, None, None, None)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_ms_experimental_pipeline_requires_task_and_model():
    """
    Feature: experimental pipeline preconditions
    Description: `get_ms_experimental_pipeline` must raise a RuntimeError when
                 task or model is not provided.
    Expectation: Raises RuntimeError when task and model are both None.
    """
    with pytest.raises(RuntimeError):
        pipeline_module.get_ms_experimental_pipeline(task=None, model=None)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pipeline_dispatches_between_standard_and_experimental(monkeypatch):
    """
    Feature: pipeline entry point
    Description: Ensure `pipeline` routes to experimental or standard builders based on model input.
    Expectation: Correct helper is invoked and invalid backend raises ValueError.
    """

    called = {"standard": 0, "experimental": 0}

    def fake_is_experimental_mode(model, **_):
        return isinstance(model, str) and model.startswith("exp/")

    def fake_get_ms_pipeline(*args, **kwargs):
        called["standard"] += 1
        return ("standard", args, kwargs)

    def fake_get_ms_experimental_pipeline(*args, **kwargs):
        called["experimental"] += 1
        return ("experimental", args, kwargs)

    monkeypatch.setattr(pipeline_module, "is_experimental_mode", fake_is_experimental_mode)
    monkeypatch.setattr(pipeline_module, "get_ms_pipeline", fake_get_ms_pipeline)
    monkeypatch.setattr(pipeline_module, "get_ms_experimental_pipeline", fake_get_ms_experimental_pipeline)

    result_exp = pipeline_module.pipeline(task="text-generation", model="exp/repo")
    assert result_exp[0] == "experimental"
    result_std = pipeline_module.pipeline(task="text-generation", model=object())
    assert result_std[0] == "standard"
    assert called == {"standard": 1, "experimental": 1}

    with pytest.raises(ValueError):
        pipeline_module.pipeline(task="text-generation", model="m", backend="unknown")


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_ms_pipeline_constructs_components(monkeypatch):
    """
    Feature: MindSpore pipeline builder
    Description: Ensure `get_ms_pipeline` builds network, processors, tokenizer, and pipeline call.
    Expectation: Helper functions receive expected arguments and output from build_pipeline is returned.
    """

    monkeypatch.setattr(pipeline_module, "SUPPORT_PIPELINES", {"demo-task": {"foo": "cfg.yaml"}})

    def fake_config(path_value):
        del path_value
        return types.SimpleNamespace(
            model="MODEL_CFG",
            processor=types.SimpleNamespace(
                tokenizer="TOK_CFG", image_processor="IMG_CFG", audio_processor="AUD_CFG"
            ),
        )

    monkeypatch.setattr(pipeline_module, "MindFormerConfig", fake_config)

    calls = {}

    def fake_build_network(model_cfg, default_args):
        calls["build_network"] = (model_cfg, default_args)
        return "MODEL_OBJ"

    def fake_build_processor(argument):
        calls.setdefault("build_processor", []).append(argument)
        return f"PROC({argument})"

    def fake_build_tokenizer(argument, tokenizer_name):
        calls["build_tokenizer"] = (argument, tokenizer_name)
        return "TOKENIZER_OBJ"

    def fake_build_pipeline(**kwargs):
        calls["build_pipeline"] = kwargs
        return "PIPELINE_OBJ"

    monkeypatch.setattr(pipeline_module, "build_network", fake_build_network)
    monkeypatch.setattr(pipeline_module, "build_processor", fake_build_processor)
    monkeypatch.setattr(pipeline_module, "build_tokenizer", fake_build_tokenizer)
    monkeypatch.setattr(pipeline_module, "build_pipeline", fake_build_pipeline)

    output = pipeline_module.get_ms_pipeline(
        "demo-task",
        "foo",
        tokenizer=None,
        image_processor=None,
        audio_processor=None,
        batch_size=4,
        use_past=True,
    )

    assert output == "PIPELINE_OBJ"
    assert calls["build_network"] == ("MODEL_CFG", {"batch_size": 4, "use_past": True})
    assert calls["build_tokenizer"] == ("TOK_CFG", "foo")
    # image/audio processors are both constructed
    assert calls["build_processor"] == ["IMG_CFG", "AUD_CFG"]
    assert calls["build_pipeline"]["model"] == "MODEL_OBJ"
    assert calls["build_pipeline"]["tokenizer"] == "TOKENIZER_OBJ"
    assert calls["build_pipeline"]["image_processor"].startswith("PROC")


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_ms_pipeline_invalid_inputs(monkeypatch):
    """
    Feature: MindSpore pipeline validation
    Description: Validate errors for unsupported tasks and model names.
    Expectation: Raises KeyError in both cases.
    """

    monkeypatch.setattr(pipeline_module, "SUPPORT_PIPELINES", {"valid": {"foo": "cfg"}})
    with pytest.raises(KeyError):
        pipeline_module.get_ms_pipeline("missing-task", None, None, None, None)
    with pytest.raises(KeyError):
        pipeline_module.get_ms_pipeline("valid", "bar", None, None, None)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_ms_experimental_pipeline_builds_components(monkeypatch):
    """
    Feature: Experimental pipeline builder
    Description: Exercise a happy path covering tokenizer/image processor loading and context setup.
    Expectation: Returns instantiated pipeline class with propagated kwargs.
    """

    class DummyPipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    dummy_config = types.SimpleNamespace(custom_pipelines={}, _commit_hash="cfg-hash")

    class DummyModelConfig:
        _commit_hash = "model-hash"
        tokenizer_class = None

    class DummyModel:
        def __init__(self):
            self.config = DummyModelConfig()
            self._eval_called = False

        def eval(self):
            self._eval_called = True
            return self

    def fake_check_task(task):
        return task, {"impl": DummyPipeline, "ms": (object,)}, None

    created = {}

    def fake_load_model(model, **kwargs):
        created["load_model"] = (model, kwargs)
        return DummyModel()

    def fake_auto_tokenizer(identifier, **_):
        created["tokenizer"] = identifier
        return "TOKENIZER"

    def fake_auto_image_processor(identifier, **_):
        created["image_processor"] = identifier
        return "IMAGE_PROCESSOR"

    monkeypatch.setattr(pipeline_module, "check_task", fake_check_task)
    monkeypatch.setattr(pipeline_module, "load_model", fake_load_model)
    monkeypatch.setattr(pipeline_module.AutoTokenizer, "from_pretrained", fake_auto_tokenizer)
    monkeypatch.setattr(pipeline_module.AutoImageProcessor, "from_pretrained", fake_auto_image_processor)
    monkeypatch.setattr(pipeline_module, "cached_file", lambda *_, **__: "cfg")
    monkeypatch.setattr(pipeline_module, "extract_commit_hash", lambda *_: "commit")
    monkeypatch.setattr(pipeline_module, "set_context", lambda **kwargs: created.setdefault("context", kwargs))

    monkeypatch.setattr(pipeline_module, "TOKENIZER_MAPPING", {DummyModelConfig: None})
    monkeypatch.setattr(pipeline_module, "IMAGE_PROCESSOR_MAPPING", {DummyModelConfig: None})
    monkeypatch.setattr(pipeline_module, "NO_IMAGE_PROCESSOR_TASKS", set())

    pipeline_obj = pipeline_module.get_ms_experimental_pipeline(
        task="text-generation",
        model="repo/model",
        config=dummy_config,
        tokenizer=None,
        image_processor=None,
        device_id=1,
        device_target="Ascend",
        model_kwargs={"foo": "bar"},
        pipeline_class=DummyPipeline,
        device_map="cpu",
        torch_dtype="fp16",
    )

    assert isinstance(pipeline_obj, DummyPipeline)
    assert pipeline_obj.kwargs["tokenizer"] == "TOKENIZER"
    assert pipeline_obj.kwargs["image_processor"] == "IMAGE_PROCESSOR"
    assert created["context"] == {"mode": 0, "device_id": 1, "device_target": "Ascend"}


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_load_model_prefers_successful_class():
    """
    Feature: Model loading helper
    Description: Ensure `load_model` iterates over candidate classes until one succeeds.
    Expectation: Returns first successful model and raises when all fail.
    """

    class FailingModel:
        called = 0

        @classmethod
        def from_pretrained(cls, *_, **__):
            cls.called += 1
            raise OSError("fail")

    class SuccessfulModel:
        called = 0

        @classmethod
        def from_pretrained(cls, *_, **__):
            cls.called += 1
            return cls()

    config = types.SimpleNamespace(architectures=None)
    model = pipeline_module.load_model(
        "id",
        config=config,
        model_classes=(FailingModel, SuccessfulModel),
        task="text-generation",
    )
    assert isinstance(model, SuccessfulModel)
    assert FailingModel.called == 1 and SuccessfulModel.called == 1

    class AlwaysFail:
        @classmethod
        def from_pretrained(cls, *_, **__):
            raise ValueError("bad")

    with pytest.raises(ValueError):
        pipeline_module.load_model(
            "id",
            config=config,
            model_classes=(AlwaysFail,),
            task="text-generation",
        )
