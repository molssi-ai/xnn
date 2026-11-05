"""The optional C++ CPU kernels (xnn.common.models.cpu_extension): off by default, and a build that fails
leaves the PyTorch paths in place, tried once per process."""
import torch.utils.cpp_extension

from xnn.common.models import cpu_extension


def test_a_failed_build_is_tried_once(monkeypatch):
    """While the backend is off nothing is built; a build that fails gives ``None`` and is not tried again."""
    calls = []

    def failing(**kwargs):
        calls.append(kwargs["name"])
        raise RuntimeError("no compiler")

    monkeypatch.setattr(torch.utils.cpp_extension, "load_inline", failing)
    monkeypatch.setattr(cpu_extension, "_ENABLED", cpu_extension.enabled())
    monkeypatch.setattr(cpu_extension, "_FAILED", set())
    cpu_extension.enable(False)
    assert not cpu_extension.enabled() and cpu_extension.load("xnn_test_broken", "", ["f"]) is None
    assert calls == []
    cpu_extension.enable()
    assert cpu_extension.enabled()
    assert cpu_extension.load("xnn_test_broken", "", ["f"]) is None
    assert cpu_extension.load("xnn_test_broken", "", ["f"]) is None
    assert calls == ["xnn_test_broken"] and "xnn_test_broken" in cpu_extension._FAILED
