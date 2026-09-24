"""Tiny test helpers used by every notebook.

    check("softmax", mine, torch.softmax(x, -1))
    check_grad("softmax", my_softmax, lambda x: torch.softmax(x, -1), x)
"""
import torch


def seed(s: int = 0) -> None:
    torch.manual_seed(s)


def check_shape(name, t, expected_shape):
    got = tuple(t.shape)
    expected_shape = tuple(expected_shape)
    assert got == expected_shape, f"❌ {name}: shape {got}, expected {expected_shape}"
    print(f"✅ {name}: shape {got}")


def check(name, got, expected, atol=1e-5, rtol=1e-4):
    """Assert two tensors (or python numbers) match, with a useful error message."""
    got = torch.as_tensor(got)
    expected = torch.as_tensor(expected)
    assert got.shape == expected.shape, (
        f"❌ {name}: shape mismatch, got {tuple(got.shape)} expected {tuple(expected.shape)}"
    )
    got_f, exp_f = got.double(), expected.double()
    if not torch.isfinite(got_f).all() and torch.isfinite(exp_f).all():
        raise AssertionError(f"❌ {name}: your output has nan/inf but the reference doesn't")
    if not torch.allclose(got_f, exp_f, atol=atol, rtol=rtol, equal_nan=True):
        diff = (got_f - exp_f).abs().max().item()
        raise AssertionError(f"❌ {name}: values differ (max abs diff {diff:.3e})")
    print(f"✅ {name}")


def check_grad(name, fn, ref_fn, *inputs, atol=1e-5, rtol=1e-4):
    """Check that gradients of fn match ref_fn w.r.t. every floating-point input."""
    def grads(f):
        xs = [x.detach().clone().requires_grad_(x.is_floating_point()) if torch.is_tensor(x) else x
              for x in inputs]
        out = f(*xs)
        g = torch.Generator().manual_seed(1234)
        w = torch.randn(out.shape, generator=g, dtype=out.dtype)
        (out * w).sum().backward()
        return [x.grad for x in xs if torch.is_tensor(x) and x.requires_grad]

    for i, (a, b) in enumerate(zip(grads(fn), grads(ref_fn))):
        check(f"{name} (grad of input {i})", a, b, atol=atol, rtol=rtol)
