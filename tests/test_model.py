# tests/test_model.py
import torch
from src.model import LeNetTiny


def test_param_count():
    model = LeNetTiny()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 5994, f"Expected 5994 params, got {n_params}"


def test_output_shape():
    model = LeNetTiny()
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"


def test_no_batchnorm():
    """Model must not contain BatchNorm (breaks Hessian computation)."""
    model = LeNetTiny()
    for name, module in model.named_modules():
        assert not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)), (
            f"Found BatchNorm at {name}"
        )
