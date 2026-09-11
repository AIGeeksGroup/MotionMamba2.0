import torch

from state_memory_operator.content_memory import PersistentContentMemory


def test_same_run_shadow_read_tensor_exact():
    memory = PersistentContentMemory(4, 8, slots=2).float()
    slots = torch.randn(2, 2, 8)
    shadow = memory.read(slots).detach().clone()
    torch.testing.assert_close(memory.read(slots), shadow, rtol=0, atol=0)


def test_read_ready_preserves_dtype_device_stride():
    memory = PersistentContentMemory(4, 8, slots=2).float()
    value = torch.randn(2, 2, 8).contiguous()
    checked = memory.validate_read_ready(value)
    assert checked.dtype == value.dtype
    assert checked.device == value.device
    assert checked.stride() == value.stride()
    assert checked.is_contiguous()


def test_read_ready_rejects_wrong_shape():
    memory = PersistentContentMemory(4, 8, slots=2).float()
    with __import__("pytest").raises(ValueError):
        memory.validate_read_ready(torch.zeros(2, 3, 8))
