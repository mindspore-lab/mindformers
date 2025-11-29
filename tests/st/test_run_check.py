"""Test for run_check function"""
import pytest
from mindformers import run_check


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_run_check():
    """
    Feature: Test run_check function
    Description: Call run_check to check if MindSpore, MindFormers, CANN and driver versions are compatible
    Expectation: No exceptions raised, all checks pass
    """
    run_check()
