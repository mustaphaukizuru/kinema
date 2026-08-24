import pytest
import torch

from kinema import Unet3D, VideoDiffusion


def make(device = 'cpu', timesteps = 20, **kw):
    model = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8, cond_dim = kw.pop('cond_dim', None))
    return VideoDiffusion(model, image_size = 16, num_frames = 2, timesteps = timesteps, **kw).to(device)


def test_ddim_shape_and_range(device):
    out = make(device).sample(batch_size = 2, sampling_timesteps = 5, progress = False)
    assert out.shape == (2, 3, 2, 16, 16)
    assert torch.isfinite(out).all()


def test_ddim_is_deterministic_at_eta_zero():
    diff = make()
    torch.manual_seed(1)
    a = diff.sample(batch_size = 1, sampling_timesteps = 5, progress = False)
    torch.manual_seed(1)
    b = diff.sample(batch_size = 1, sampling_timesteps = 5, progress = False)
    assert torch.equal(a, b)


def test_ddim_eta_one_is_stochastic():
    diff = make()
    torch.manual_seed(1)
    a = diff.sample(batch_size = 1, sampling_timesteps = 5, eta = 0., progress = False)
    torch.manual_seed(1)
    b = diff.sample(batch_size = 1, sampling_timesteps = 5, eta = 1., progress = False)
    assert not torch.equal(a, b)


def test_sampling_timesteps_on_constructor_routes_to_ddim():
    diff = make(sampling_timesteps = 5)
    assert diff.sampling_timesteps == 5
    assert diff.sample(batch_size = 1, progress = False).shape == (1, 3, 2, 16, 16)


def test_full_chain_used_when_steps_equal_timesteps():
    diff = make(timesteps = 10)
    assert diff.sample(batch_size = 1, sampling_timesteps = 10, progress = False).shape == (1, 3, 2, 16, 16)


@pytest.mark.parametrize('steps', [0, 21])
def test_invalid_sampling_timesteps_rejected(steps):
    with pytest.raises(AssertionError):
        make(sampling_timesteps = steps)


def test_ddim_with_conditioning_and_cfg():
    diff = make(cond_dim = 4)
    cond = torch.randn(2, 4)
    out = diff.sample(cond = cond, cond_scale = 2., sampling_timesteps = 5, progress = False)
    assert out.shape == (2, 3, 2, 16, 16) and torch.isfinite(out).all()


def test_ddim_respects_dynamic_thresholding():
    diff = make(use_dynamic_thres = True)
    assert diff.sample(batch_size = 1, sampling_timesteps = 5, progress = False).shape == (1, 3, 2, 16, 16)


def test_predict_noise_from_start_inverts_predict_start_from_noise():
    diff = make()
    x_t = torch.randn(2, 3, 2, 16, 16)
    noise = torch.randn_like(x_t)
    t = torch.tensor([3, 7])

    x_start = diff.predict_start_from_noise(x_t, t = t, noise = noise)
    assert torch.allclose(diff.predict_noise_from_start(x_t, t = t, x_start = x_start), noise, atol = 1e-4)


def test_progress_false_is_silent(capsys):
    make().sample(batch_size = 1, sampling_timesteps = 3, progress = False)
    assert capsys.readouterr().err == ''


def test_progress_true_writes_a_bar(capsys):
    make().sample(batch_size = 1, sampling_timesteps = 3, progress = True)
    assert 'ddim sampling time step' in capsys.readouterr().err


def test_interpolate_progress_can_be_silenced(capsys):
    diff = make()
    a = torch.rand(1, 3, 2, 16, 16)
    diff.interpolate(a, a, t = 3, progress = False)
    assert capsys.readouterr().err == ''
