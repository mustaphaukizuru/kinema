import pytest
import torch

from kinema import Unet3D, VideoDiffusion


def make(device, cond_dim = None, **kw):
    model = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8, cond_dim = cond_dim, **kw)
    return VideoDiffusion(model, image_size = 16, num_frames = 2, timesteps = 10).to(device)


def test_train_step_shapes(device):
    diff = make(device)
    loss = diff(torch.rand(2, 3, 2, 16, 16, device = device))
    loss.backward()
    assert loss.ndim == 0 and torch.isfinite(loss)


@pytest.mark.parametrize('loss_type', ['l1', 'l2'])
def test_loss_types(loss_type):
    model = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8)
    diff = VideoDiffusion(model, image_size = 16, num_frames = 2, timesteps = 10, loss_type = loss_type)
    assert torch.isfinite(diff(torch.rand(1, 3, 2, 16, 16)))


def test_sample_shape_and_range(device):
    out = make(device).sample(batch_size = 2)
    assert out.shape == (2, 3, 2, 16, 16)
    assert torch.isfinite(out).all()


def test_cond_and_cfg(device):
    diff = make(device, cond_dim = 4)
    cond = torch.randn(2, 4, device = device)
    diff(torch.rand(2, 3, 2, 16, 16, device = device), cond = cond).backward()
    assert diff.sample(cond = cond, cond_scale = 2.).shape == (2, 3, 2, 16, 16)


def test_cond_required():
    diff = make('cpu', cond_dim = 4)
    with pytest.raises(AssertionError):
        diff(torch.rand(1, 3, 2, 16, 16))


def test_dynamic_thresholding():
    model = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8)
    diff = VideoDiffusion(model, image_size = 16, num_frames = 2, timesteps = 5, use_dynamic_thres = True)
    assert diff.sample(batch_size = 1).shape == (1, 3, 2, 16, 16)


def test_focus_present(device):
    diff = make(device)
    x = torch.rand(2, 3, 2, 16, 16, device = device)
    assert torch.isfinite(diff(x, prob_focus_present = 1.))
    assert torch.isfinite(diff(x, focus_present_mask = torch.tensor([True, False], device = device)))


def test_interpolate_unconditional(device):
    diff = make(device)
    a, b = torch.rand(1, 3, 2, 16, 16, device = device), torch.rand(1, 3, 2, 16, 16, device = device)
    assert diff.interpolate(a, b, t = 5).shape == a.shape


def test_interpolate_conditional():
    diff = make('cpu', cond_dim = 4)
    a = torch.rand(1, 3, 2, 16, 16)
    out = diff.interpolate(a, a, t = 3, cond = torch.randn(1, 4), cond_scale = 2.)
    assert out.shape == a.shape and torch.isfinite(out).all()


def test_unet_no_sparse_attn():
    diff = make('cpu', use_sparse_linear_attn = False)
    assert torch.isfinite(diff(torch.rand(1, 3, 2, 16, 16)))
