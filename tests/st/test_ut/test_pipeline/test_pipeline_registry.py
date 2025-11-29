"""Unit tests for the pipeline registry helpers."""

import pytest

from mindformers.pipeline import pipeline_registry as pipeline_registry_module
from mindformers.pipeline.pipeline_registry import PipelineRegistry


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_supported_tasks_and_to_dict():
    """Get supported tasks and verify dict output.

    Feature:
        Test behavior of `PipelineRegistry.get_supported_tasks` and `to_dict`.

    Description:
        Initialize a registry with one task and one alias, verify the returned
        task list contains both entries (sorted), and verify `to_dict` returns
        the original mapping object.

    Expectation:
        `tasks` contains two items and `to_dict` returns the original dict.
    """
    supported = {"task1": {"impl": object}}
    aliases = {"alias1": "task1"}
    reg = PipelineRegistry(supported, aliases)
    tasks = reg.get_supported_tasks()
    # two entries sorted
    assert tasks in (["alias1", "task1"], ["task1", "alias1"])
    # to_dict returns the underlying mapping
    assert reg.to_dict() is supported


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_task_direct_and_alias():
    """Validate direct task name and alias resolution.

    Feature:
        Ensure `check_task` returns the normalized task name and the
        corresponding implementation when given a direct task name or an alias.

    Description:
        Provide a mapping that contains `t1` and `translation` and an alias
        that maps to `t1`. Call `check_task` with both types and assert the
        returned values.

    Expectation:
        Both direct name and alias resolve to the same target implementation
        and options are None.
    """
    supported = {"t1": {"impl": object}, "translation": {"impl": object}}
    aliases = {"alias": "t1"}
    reg = PipelineRegistry(supported, aliases)

    name, targeted, opts = reg.check_task("t1")
    assert name == "t1"
    assert targeted is supported["t1"]
    assert opts is None

    # alias should map to target
    name2, targeted2, opts2 = reg.check_task("alias")
    assert name2 == "t1"
    assert targeted2 is supported["t1"]
    assert opts2 is None


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_task_translation_valid_and_invalid():
    """Test parsing and error handling for translation tasks.

    Feature:
        Verify that parameterized translation tasks of the form
        `translation_XX_to_YY` are parsed correctly and invalid formats raise
        a KeyError.

    Description:
        Use a supported mapping that only contains `translation`. Test a valid
        translation task and an invalid format that should raise.

    Expectation:
        Valid string returns language tuple; invalid format raises KeyError.
    """
    supported = {"translation": {"impl": object}}
    reg = PipelineRegistry(supported, {})

    # valid: translation_en_to_de
    name, targeted, options = reg.check_task("translation_en_to_de")
    assert name == "translation"
    assert targeted is supported["translation"]
    assert options == ("en", "de")

    # invalid format should raise KeyError
    with pytest.raises(KeyError):
        reg.check_task("translation_en_de")


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_task_unknown_raises():
    """Ensure unknown task names raise an informative KeyError.

    Feature:
        The registry should raise a KeyError for unknown tasks and include
        'Unknown task' in the error message.

    Description:
        Initialize an empty registry and call `check_task` with a non-existent
        task name. Verify that a KeyError is raised and the error message
        contains the expected string.

    Expectation:
        KeyError is raised and message contains 'Unknown task'.
    """
    reg = PipelineRegistry({}, {})
    with pytest.raises(KeyError) as ei:
        reg.check_task("noexist")
    assert "Unknown task" in str(ei.value)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_register_pipeline_overwrite_and_defaults_and_task_type(monkeypatch):
    """Test pipeline registration, default wrapping, and task type field.

    Feature:
        Ensure `register_pipeline` correctly registers implementations,
        wraps a default that only contains 'ms' into {'model': ...}, and logs
        a warning when overwriting an existing registration.

    Description:
        Register a custom pipeline class into an empty supported mapping and
        verify stored values and class-level `registered_impl`. Then register
        again and assert a warning was emitted.

    Expectation:
        The supported mapping contains the correct entries, the pipeline class
        has `registered_impl`, and a warning is logged on overwrite.
    """
    supported = {}
    reg = PipelineRegistry(supported, {})

    class MyPipeline:
        pass

    # register with default that has only 'ms' key -> should be wrapped under 'model'
    reg.register_pipeline("t", MyPipeline, ms_model=(int,), default={"ms": "value"}, task_type="typeA")
    assert "t" in supported
    impl = supported["t"]
    assert impl["impl"] is MyPipeline
    assert impl["ms"] == (int,)
    # default should be wrapped into {'model': {...}}
    assert impl["default"] == {"model": {"ms": "value"}}
    assert impl["type"] == "typeA"
    # The pipeline_class should have registered_impl attribute set
    assert hasattr(MyPipeline, "registered_impl")
    assert "t" in MyPipeline.registered_impl

    # registering again should produce a warning about overwriting
    warnings = []

    def fake_warning(msg, *_args, **_kwargs):
        warnings.append(msg)

    monkeypatch.setattr(pipeline_registry_module.logger, "warning", fake_warning)

    reg.register_pipeline("t", MyPipeline, ms_model=(int,))
    assert warnings, "Expected warning to be emitted"
    assert any("is already registered" in msg for msg in warnings)
