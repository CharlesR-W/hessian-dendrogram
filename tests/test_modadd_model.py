import torch
from src.modadd_model import ModAddTransformer


def test_default_param_count():
    """d_model=32, p=113 should give ~11,456 parameters."""
    model = ModAddTransformer(p=113, d_model=32)
    n_params = sum(p.numel() for p in model.parameters())
    # token_embed: 114*32=3648, pos_embed: 3*32=96
    # W_Q,W_K,W_V,W_O: 4*32*32=4096, unembed: 32*113=3616
    assert n_params == 11456, f"Expected 11456 params, got {n_params}"


def test_output_shape():
    model = ModAddTransformer(p=113, d_model=32)
    tokens = torch.randint(0, 113, (8, 3))  # batch=8, seq=3
    tokens[:, 2] = 113  # = token at position 2
    out = model(tokens)
    assert out.shape == (8, 113), f"Expected (8, 113), got {out.shape}"


def test_small_model():
    """Smaller model for quick Hessian tests."""
    model = ModAddTransformer(p=7, d_model=8)
    tokens = torch.randint(0, 7, (4, 3))
    tokens[:, 2] = 7
    out = model(tokens)
    assert out.shape == (4, 7)


def test_no_batchnorm():
    """Model must not contain BatchNorm (breaks Hessian computation)."""
    model = ModAddTransformer()
    for name, module in model.named_modules():
        assert not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)), (
            f"Found BatchNorm at {name}"
        )


def test_multi_head():
    """Multi-head attention should work when d_model divisible by n_heads."""
    model = ModAddTransformer(p=113, d_model=32, n_heads=4)
    tokens = torch.randint(0, 113, (4, 3))
    tokens[:, 2] = 113
    out = model(tokens)
    assert out.shape == (4, 113)
