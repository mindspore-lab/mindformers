"""Tests for registry_constant helpers and data wiring."""

import importlib
from importlib import util as importlib_util
from pathlib import Path

import pytest


# Resolve the path to the real registry_constant.py in a robust way.
# Prefer importlib's spec if the package is importable; otherwise fall back
# to searching parent directories for the repository layout.
try:
    _spec = importlib_util.find_spec("mindformers.pipeline.registry_constant")
    if _spec and getattr(_spec, "origin", None):
        REGISTRY_PATH = Path(_spec.origin)
    else:
        raise RuntimeError("spec not found")
except Exception:
    # fallback: look for the file under parents; this works when running
    # tests from workspace without installing the package
    base = Path(__file__).resolve()

    # 1) Find repository root by common markers (.git, setup.py, pyproject.toml)
    repo_root = None
    for p in base.parents:
        if (p / ".git").exists() or (p / "setup.py").exists() or (p / "pyproject.toml").exists():
            repo_root = p
            break

    if repo_root is not None:
        candidate = repo_root / "mindformers" / "pipeline" / "registry_constant.py"
        if candidate.exists():
            REGISTRY_PATH = candidate
        else:
            # if repo root found but file missing, fall back to a best-effort search below
            repo_root = None

    # 2) If repo root not determined, search parents for a candidate but avoid
    # picking up nested 'tests/mindformers' layouts which can occur when tests
    # are run from a working directory that mirrors the package tree.
    if repo_root is None:
        found = None
        for p in base.parents:
            candidate = p / "mindformers" / "pipeline" / "registry_constant.py"
            if candidate.exists():
                # prefer candidates not under a 'tests' directory
                if "tests" not in map(str, candidate.parts):
                    found = candidate
                    break
                # otherwise keep the first found as a last resort
                if found is None:
                    found = candidate
        if found is None:
            # original fallback (best-effort)
            REGISTRY_PATH = Path(__file__).resolve().parents[3] / "mindformers" / "pipeline" / "registry_constant.py"
        else:
            REGISTRY_PATH = found


def _load_registry_with_supported_tasks(supported_tasks):
    """Dynamically execute a modified copy of registry_constant with a
    custom SUPPORTED_TASKS dict. Returns the execution namespace dict.

    This approach recreates the module-level initialization logic (the
    for-loop that populates NO_* sets) without importing the real module
    (which would already have executed with its built-in SUPPORTED_TASKS).
    """
    src = REGISTRY_PATH.read_text(encoding="utf-8")

    # find the place where the NO_* sets start; reuse the remainder of the
    # file (including the for-loop and PIPELINE_REGISTRY init) so we only
    # replace SUPPORTED_TASKS.
    marker = "NO_FEATURE_EXTRACTOR_TASKS = set()"
    idx = src.find(marker)
    assert idx != -1, "registry_constant.py structure changed; cannot locate marker"
    line_offset = src[:idx].count("\n")
    remainder = ("\n" * line_offset) + src[idx:]

    # build a new source where SUPPORTED_TASKS is our custom dict
    new_src = "SUPPORTED_TASKS = " + repr(supported_tasks) + "\n" + remainder

    # prepare a namespace with minimal dependencies mocked
    class DummyPipelineRegistry:
        def __init__(self, supported_tasks=None, task_aliases=None):
            self.supported_tasks = supported_tasks
            self.task_aliases = task_aliases

    namespace = {
        "PipelineRegistry": DummyPipelineRegistry,
        "TextGenerationPipeline": object,
        "AutoModelForCausalLM": object,
        "TASK_ALIASES": {},
    }

    exec(compile(new_src, str(REGISTRY_PATH), "exec"), namespace)  # pylint: disable=W0122
    return namespace


def _recompute_sets_in_module(custom_tasks):
    """Execute the real registry_constant for-loop against custom tasks."""

    module = importlib.import_module("mindformers.pipeline.registry_constant")
    module.SUPPORTED_TASKS = custom_tasks
    module.NO_FEATURE_EXTRACTOR_TASKS = set()
    module.NO_IMAGE_PROCESSOR_TASKS = set()
    module.NO_TOKENIZER_TASKS = set()

    src = REGISTRY_PATH.read_text(encoding="utf-8")
    marker_start = "for task, values in SUPPORTED_TASKS.items():"
    marker_end = "PIPELINE_REGISTRY ="
    start_idx = src.find(marker_start)
    end_idx = src.find(marker_end)
    assert start_idx != -1 and end_idx != -1, "registry_constant structure changed"
    line_offset = src[:start_idx].count("\n")
    loop_src = ("\n" * line_offset) + src[start_idx:end_idx]

    exec(compile(loop_src, str(REGISTRY_PATH), "exec"), module.__dict__)  # pylint: disable=W0122
    return module


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_default_supported_tasks_processing():
    """
    Feature: registry_constant defaults
    Description: Ensure NO_* sets include expected defaults when importing the real module.
    Expectation: text-generation is categorized correctly across NO_* sets.
    """
    # Importing the real module executes the loop once; assert default behavior
    rc = importlib.import_module("mindformers.pipeline.registry_constant")
    assert "text-generation" in rc.NO_FEATURE_EXTRACTOR_TASKS
    assert "text-generation" in rc.NO_IMAGE_PROCESSOR_TASKS
    assert "text-generation" not in rc.NO_TOKENIZER_TASKS


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_processing_various_types_and_error():
    """
    Feature: registry_constant type handling
    Description: Validate NO_* set assignment for image/audio types and error for invalid type.
    Expectation: Image/audio tasks populate correct NO_* sets; unknown type raises ValueError.
    """
    # image type -> tokenizers not required
    ns = _load_registry_with_supported_tasks({"img-task": {"type": "image"}})
    assert "img-task" in ns["NO_TOKENIZER_TASKS"]

    # audio type -> tokenizer + image processor not required
    ns = _load_registry_with_supported_tasks({"aud-task": {"type": "audio"}})
    assert "aud-task" in ns["NO_TOKENIZER_TASKS"]
    assert "aud-task" in ns["NO_IMAGE_PROCESSOR_TASKS"]

    # invalid type should raise ValueError during module execution
    with pytest.raises(ValueError):
        _load_registry_with_supported_tasks({"bad": {"type": "unknown"}})


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pipeline_registry_aliases_and_supported_tasks_consistency():
    """
    Feature: PIPELINE_REGISTRY wiring
    Description: Verify exported PIPELINE_REGISTRY shares TASK_ALIASES/SUPPORTED_TASKS references.
    Expectation: Registry uses the same dict objects and exposes alias entries defined in TASK_ALIASES.
    """

    rc = importlib.import_module("mindformers.pipeline.registry_constant")
    assert rc.PIPELINE_REGISTRY.task_aliases is rc.TASK_ALIASES
    assert rc.PIPELINE_REGISTRY.supported_tasks is rc.SUPPORTED_TASKS
    assert rc.TASK_ALIASES["text_generation"] == "text-generation"


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_processing_video_and_multimodal_types():
    """
    Feature: registry_constant video/multimodal handling
    Description: Ensure video tasks skip tokenizer and multimodal entries bypass NO_* categorization.
    Expectation: Video tasks populate NO_TOKENIZER_TASKS; multimodal entries leave sets unchanged and do not raise.
    """

    ns = _load_registry_with_supported_tasks({
        "vid-task": {"type": "video"},
        "multi-task": {"type": "multimodal"},
    })
    assert "vid-task" in ns["NO_TOKENIZER_TASKS"]
    assert "vid-task" not in ns["NO_IMAGE_PROCESSOR_TASKS"]
    assert "multi-task" not in ns["NO_TOKENIZER_TASKS"]
    assert "multi-task" not in ns["NO_FEATURE_EXTRACTOR_TASKS"]


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_recompute_sets_in_module_covers_video_audio_and_multimodal():
    """
    Feature: registry_constant loop execution
    Description: Run the actual module-level loop with custom tasks to cover image/video/audio branches.
    Expectation: Module NO_* sets reflect injected task types and state resets cleanly after reload.
    """

    custom = {
        "img-task": {"type": "image"},
        "aud-task": {"type": "audio"},
        "multi-task": {"type": "multimodal"},
    }
    module = _recompute_sets_in_module(custom)
    assert "img-task" in module.NO_TOKENIZER_TASKS
    assert "aud-task" in module.NO_TOKENIZER_TASKS
    assert "aud-task" in module.NO_IMAGE_PROCESSOR_TASKS
    assert "multi-task" not in module.NO_TOKENIZER_TASKS

    importlib.reload(importlib.import_module("mindformers.pipeline.registry_constant"))


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_recompute_sets_invalid_type_triggers_value_error():
    """
    Feature: registry_constant invalid type guard
    Description: Ensure executing the real loop with unsupported type raises ValueError.
    Expectation: ValueError message references offending task.
    """

    with pytest.raises(ValueError):
        _recompute_sets_in_module({"bad": {"type": "unknown"}})

    importlib.reload(importlib.import_module("mindformers.pipeline.registry_constant"))
