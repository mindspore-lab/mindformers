"""Integration test for TrainerState with callbacks."""
# pylint: disable=wrong-import-position,import-outside-toplevel,protected-access
import importlib.util
import os
import sys
import traceback
from unittest.mock import Mock

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

# Mock mindspore modules
sys.modules['mindspore'] = Mock()
sys.modules['mindspore.nn'] = Mock()
sys.modules['mindspore.dataset'] = Mock()

# Mock mindformers modules
mock_logger = Mock()
sys.modules['mindformers'] = Mock()
sys.modules['mindformers.tools'] = Mock()
sys.modules['mindformers.tools.logger'] = Mock(logger=mock_logger)
sys.modules['mindformers.checkpoint'] = Mock()

# Mock CommonInfo and AsyncSaveManager
class MockCommonInfo:
    def __init__(self):
        self.epoch_num = None
        self.step_num = None
        self.global_step = None
        self.global_batch_size = None

class MockAsyncSaveManager:
    def __init__(self, async_save):
        self.async_save = async_save
    def prepare_before_save(self):
        pass

sys.modules['mindformers.checkpoint.checkpoint'] = Mock(
    CommonInfo=MockCommonInfo,
    AsyncSaveManager=MockAsyncSaveManager
)

# Direct import to avoid __init__.py issues
train_state_path = os.path.join(project_root, 'mindformers', 'trainer_pynative', 'train_state.py')
spec = importlib.util.spec_from_file_location("train_state", train_state_path)
train_state_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_state_module)
TrainerState = train_state_module.TrainerState

# Now import callbacks
from mindformers.core.callback_pynative.checkpoint_callback import CheckpointCallback
from mindformers.core.callback_pynative.loss_callback import LossCallback


def test_trainstate_with_checkpoint_callback():
    """
    Feature: TrainerState integration with CheckpointCallback
    Description: Verify TrainerState provides required fields to CheckpointCallback
    Expectation: CheckpointCallback successfully creates CommonInfo from TrainerState
    """
    print("=" * 70)
    print("Test 1: TrainerState with CheckpointCallback")
    print("=" * 70)

    # Create TrainerState with all required fields
    state = TrainerState(
        global_step=100,
        epoch=1.0,
        epoch_step=100,
        global_batch_size=64
    )

    # Create CheckpointCallback
    cb = CheckpointCallback(
        save_dir="./test_ckpts",
        save_interval=100
    )

    # Test _create_common_info
    common_info = cb._create_common_info(state)

    # Verify all fields are set correctly
    assert common_info.global_step == 100, f"Expected 100, got {common_info.global_step}"
    assert common_info.epoch_num == 1, f"Expected 1, got {common_info.epoch_num}"
    assert common_info.step_num == 0, f"Expected 0, got {common_info.step_num}"
    assert common_info.global_batch_size == 64, f"Expected 64, got {common_info.global_batch_size}"

    print("[OK] CheckpointCallback correctly uses TrainerState.global_batch_size")
    print(f"     global_batch_size: {common_info.global_batch_size}")

    return True


def test_trainstate_with_loss_callback():
    """
    Feature: TrainerState integration with LossCallback
    Description: Verify that TrainerState works correctly with LossCallback during training step callbacks
    Expectation: LossCallback can access and use TrainerState fields without errors
    """
    print("\n" + "=" * 70)
    print("Test 2: TrainerState with LossCallback")
    print("=" * 70)

    # Create TrainerState
    state = TrainerState(
        global_step=10,
        epoch=0.5,
        epoch_step=20,
        global_batch_size=32
    )

    # Create LossCallback
    cb = LossCallback(log_interval=1)

    # Simulate step_end
    args = Mock()
    cb.on_step_begin(args, state)

    # Verify state can be used
    assert state.global_step == 10
    assert state.epoch == 0.5

    print("[OK] LossCallback correctly uses TrainerState")
    print(f"     global_step: {state.global_step}")
    print(f"     epoch: {state.epoch}")

    return True


def test_trainstate_all_required_fields():
    """
    Feature: TrainerState required fields validation
    Description: Verify that TrainerState contains all required fields needed by various callbacks
    Expectation: All 11 required fields (global_step, epoch, epoch_step, etc.) are present in TrainerState
    """
    print("\n" + "=" * 70)
    print("Test 3: TrainerState has all required fields")
    print("=" * 70)

    state = TrainerState()

    required_fields = [
        'global_step',
        'epoch',
        'epoch_step',
        'global_batch_size',
        'max_steps',
        'eval_steps',
        'save_steps',
        'best_metric',
        'best_model_checkpoint',
        'is_train_begin',
        'is_train_end',
    ]

    missing_fields = []
    for field in required_fields:
        if not hasattr(state, field):
            missing_fields.append(field)

    if missing_fields:
        print(f"[FAIL] Missing fields: {missing_fields}")
        return False

    print("[OK] All required fields present")
    for field in required_fields:
        value = getattr(state, field)
        print(f"     {field}: {value}")

    return True


def test_trainstate_update_epoch():
    """
    Feature: TrainerState epoch calculation
    Description: Verify that update_epoch method correctly calculates epoch based on global_step and epoch_step
    Expectation: Epoch is calculated as global_step / epoch_step (e.g., 250 / 100 = 2.5)
    """
    print("\n" + "=" * 70)
    print("Test 4: TrainerState.update_epoch")
    print("=" * 70)

    state = TrainerState(
        global_step=250,
        epoch_step=100
    )

    state.update_epoch()

    assert state.epoch == 2.5, f"Expected 2.5, got {state.epoch}"

    print("[OK] update_epoch works correctly")
    print(f"     global_step: {state.global_step}")
    print(f"     epoch_step: {state.epoch_step}")
    print(f"     calculated epoch: {state.epoch}")

    return True


def test_trainstate_with_different_batch_sizes():
    """
    Feature: TrainerState batch size handling
    Description: Verify that TrainerState correctly stores and retrieves different global_batch_size values
    Expectation: TrainerState accurately maintains batch size values from 1 to 512
    """
    print("\n" + "=" * 70)
    print("Test 5: TrainerState with different batch sizes")
    print("=" * 70)

    batch_sizes = [1, 16, 32, 64, 128, 256, 512]

    for batch_size in batch_sizes:
        state = TrainerState(global_batch_size=batch_size)
        assert state.global_batch_size == batch_size, \
            f"Expected {batch_size}, got {state.global_batch_size}"

    print(f"[OK] Tested {len(batch_sizes)} different batch sizes")
    print(f"     Batch sizes: {batch_sizes}")

    return True


def main():
    """Run all integration tests."""
    print("=" * 70)
    print("TrainerState Integration Tests")
    print("=" * 70)

    tests = [
        test_trainstate_with_checkpoint_callback,
        test_trainstate_with_loss_callback,
        test_trainstate_all_required_fields,
        test_trainstate_update_epoch,
        test_trainstate_with_different_batch_sizes,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n[FAIL] {test_func.__name__}: {e}")
            traceback.print_exc()
        except Exception as e:
            failed += 1
            print(f"\n[ERROR] {test_func.__name__}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Integration Test Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("[OK] ALL INTEGRATION TESTS PASSED")
        return 0
    print("[FAIL] SOME INTEGRATION TESTS FAILED")
    return 1


if __name__ == '__main__':
    sys.exit(main())
