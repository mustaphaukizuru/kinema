import pytest
import torch

from kinema import Unet3D, VideoDiffusion


def make(cond_drop_prob = 0.1, device = 'cpu'):
    unet = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8, cond_dim = 4)
    return VideoDiffusion(
        unet, image_size = 16, num_frames = 2, timesteps = 10,
        cond_drop_prob = cond_drop_prob
    ).to(device)


def null_grad_after_step(cond_drop_prob):
    """Train one step and report whether the null-conditioning embedding received gradient."""
    torch.manual_seed(0)
    diffusion = make(cond_drop_prob)

    loss = diffusion(torch.rand(4, 3, 2, 16, 16), cond = torch.randn(4, 4))
    loss.backward()

    grad = diffusion.denoise_fn.null_cond_emb.grad
    return grad is not None and grad.abs().sum().item() > 0


def test_null_embedding_is_trained_by_default():
    """
    Regression: classifier-free guidance needs its unconditional branch trained, and nothing
    ever passed null_cond_prob during training. The null embedding kept its random init, so
    cond_scale extrapolated away from noise instead of from a learned prior.
    """
    assert null_grad_after_step(1.0), 'the null embedding received no gradient'


def test_cond_drop_prob_zero_leaves_the_null_branch_untrained():
    """The old behaviour, still reachable — and still wrong, which is why it is not the default."""
    assert not null_grad_after_step(0.0)


def test_default_drop_probability_is_ten_percent():
    assert make().cond_drop_prob == pytest.approx(0.1)


def test_caller_can_override_the_drop_rate_per_call():
    torch.manual_seed(0)
    diffusion = make(cond_drop_prob = 0.0)

    # an explicit null_cond_prob must win over the configured rate
    loss = diffusion(torch.rand(2, 3, 2, 16, 16), cond = torch.randn(2, 4), null_cond_prob = 1.0)
    loss.backward()

    assert diffusion.denoise_fn.null_cond_emb.grad.abs().sum().item() > 0


def test_unconditional_models_are_unaffected(device):
    unet = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8)
    diffusion = VideoDiffusion(unet, image_size = 16, num_frames = 2, timesteps = 5).to(device)

    loss = diffusion(torch.rand(2, 3, 2, 16, 16, device = device))
    loss.backward()
    assert torch.isfinite(loss)


def test_guidance_still_samples(device):
    diffusion = make(device = device)
    cond = torch.randn(2, 4, device = device)
    out = diffusion.sample(cond = cond, cond_scale = 2., sampling_timesteps = 3, progress = False)
    assert out.shape == (2, 3, 2, 16, 16) and torch.isfinite(out).all()


def test_checkpoints_remain_compatible():
    """cond_drop_prob adds no parameters, so weights move between settings freely."""
    a, b = make(cond_drop_prob = 0.0), make(cond_drop_prob = 0.2)
    b.load_state_dict(a.state_dict())
    a.load_state_dict(b.state_dict())


# ------------------------------------------------------------------ interpolation


def make_plain(device = 'cpu', **kw):
    unet = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8)
    return VideoDiffusion(unet, image_size = 16, num_frames = 2, timesteps = 20, **kw).to(device)


def test_interpolate_takes_the_ddim_path(device):
    diffusion = make_plain(device)
    a = torch.rand(1, 3, 2, 16, 16, device = device)
    b = torch.rand(1, 3, 2, 16, 16, device = device)

    out = diffusion.interpolate(a, b, t = 15, sampling_timesteps = 4, progress = False)
    assert out.shape == a.shape and torch.isfinite(out).all()


def test_interpolate_ddim_is_deterministic():
    diffusion = make_plain()
    a = torch.rand(1, 3, 2, 16, 16)

    torch.manual_seed(3)
    first = diffusion.interpolate(a, a, t = 10, sampling_timesteps = 4, eta = 0., progress = False)
    torch.manual_seed(3)
    second = diffusion.interpolate(a, a, t = 10, sampling_timesteps = 4, eta = 0., progress = False)
    assert torch.equal(first, second)


def test_interpolate_full_chain_still_available():
    diffusion = make_plain(sampling_timesteps = 20)
    a = torch.rand(1, 3, 2, 16, 16)
    out = diffusion.interpolate(a, a, t = 6, sampling_timesteps = 6, progress = False)
    assert out.shape == a.shape


def test_interpolate_ddim_is_faster_in_steps_taken(monkeypatch):
    """Fewer steps must mean fewer network evaluations, not just a different code path."""
    diffusion = make_plain()
    a = torch.rand(1, 3, 2, 16, 16)

    calls = []
    original = diffusion.denoise_fn.forward_with_cond_scale

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(diffusion.denoise_fn, 'forward_with_cond_scale', counting)

    diffusion.interpolate(a, a, t = 16, sampling_timesteps = 4, progress = False)
    ddim_calls = len(calls)

    calls.clear()
    diffusion.interpolate(a, a, t = 16, sampling_timesteps = 16, progress = False)
    full_calls = len(calls)

    assert ddim_calls < full_calls, f'ddim {ddim_calls} vs full {full_calls}'
