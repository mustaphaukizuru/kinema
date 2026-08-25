import pytest
import torch

from kinema import Unet3D, VideoDiffusion


def make(objective = 'noise', device = 'cpu', timesteps = 20, **kw):
    unet = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8, **kw)
    return VideoDiffusion(
        unet, image_size = 16, num_frames = 2, timesteps = timesteps, objective = objective
    ).to(device)


def test_noise_is_the_default():
    assert make().objective == 'noise'


def test_unknown_objective_is_rejected():
    with pytest.raises(AssertionError, match = "objective must be"):
        make(objective = 'epsilon')


def test_v_and_start_conversions_are_inverses():
    diffusion = make()
    x_start = torch.rand(2, 3, 2, 16, 16)
    noise = torch.randn_like(x_start)
    t = torch.tensor([4, 11])

    x_t = diffusion.q_sample(x_start, t, noise = noise)
    v = diffusion.predict_v(x_start, t, noise)

    assert torch.allclose(diffusion.predict_start_from_v(x_t, t, v), x_start, atol = 1e-4)


def test_model_predictions_returns_a_consistent_pair():
    """x_start and pred_noise must describe the same x_t, or the samplers disagree with themselves."""
    diffusion = make()
    x = torch.randn(1, 3, 2, 16, 16)
    t = torch.tensor([7])

    pred_noise, x_start = diffusion.model_predictions(x, t, clip_denoised = False)
    rebuilt = diffusion.q_sample(x_start, t, noise = pred_noise)

    assert torch.allclose(rebuilt, x, atol = 1e-3)


@pytest.mark.parametrize('objective', ['noise', 'v'])
def test_training_runs_for_both_objectives(objective, device):
    diffusion = make(objective, device)
    loss = diffusion(torch.rand(2, 3, 2, 16, 16, device = device))
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize('objective', ['noise', 'v'])
def test_sampling_runs_for_both_objectives(objective, device):
    diffusion = make(objective, device)

    ddpm = diffusion.sample(batch_size = 1, sampling_timesteps = 20, progress = False)
    ddim = diffusion.sample(batch_size = 1, sampling_timesteps = 4, progress = False)

    assert ddpm.shape == ddim.shape == (1, 3, 2, 16, 16)
    assert torch.isfinite(ddpm).all() and torch.isfinite(ddim).all()


def test_the_two_objectives_train_toward_different_targets():
    torch.manual_seed(0)
    noise_model = make('noise')
    torch.manual_seed(0)
    v_model = make('v')

    videos = torch.rand(2, 3, 2, 16, 16)

    torch.manual_seed(5)
    noise_loss = noise_model(videos)
    torch.manual_seed(5)
    v_loss = v_model(videos)

    assert not torch.allclose(noise_loss, v_loss)


def test_the_noise_path_is_numerically_unchanged():
    """
    v-parameterisation routed both samplers through a shared conversion. The default objective
    must come out bit-identical to what it produced before that refactor, or this was a silent
    behaviour change rather than an addition.
    """
    torch.manual_seed(0)
    diffusion = make('noise')

    x = torch.randn(1, 3, 2, 16, 16)
    t = torch.tensor([9])

    # what the old code did, inline
    raw = diffusion.denoise_fn.forward_with_cond_scale(x, t, cond = None, cond_scale = 1.)
    expected_start = diffusion.clip_x_start(diffusion.predict_start_from_noise(x, t = t, noise = raw))

    _, x_start = diffusion.model_predictions(x, t)

    assert torch.equal(x_start, expected_start)


def test_v_objective_with_guidance(device):
    diffusion = make('v', device, cond_dim = 4)
    cond = torch.randn(2, 4, device = device)

    diffusion(torch.rand(2, 3, 2, 16, 16, device = device), cond = cond).backward()
    out = diffusion.sample(cond = cond, cond_scale = 2., sampling_timesteps = 3, progress = False)

    assert out.shape == (2, 3, 2, 16, 16) and torch.isfinite(out).all()


def test_objective_adds_no_parameters():
    """Checkpoints stay interchangeable in shape — though not in meaning."""
    a, b = make('noise'), make('v')
    b.load_state_dict(a.state_dict())
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_v_objective_interpolates(device):
    diffusion = make('v', device)
    clip = torch.rand(1, 3, 2, 16, 16, device = device)
    out = diffusion.interpolate(clip, clip, t = 10, sampling_timesteps = 3, progress = False)
    assert out.shape == clip.shape and torch.isfinite(out).all()
