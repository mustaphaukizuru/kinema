import pytest
import torch

from kinema import Unet3D, VideoDiffusion
from kinema.modules import ResnetBlock, shift, token_shift


def make(device = 'cpu', **kw):
    unet = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8, **kw)
    return unet.to(device)


# ---------------------------------------------------------------- token shift


def test_shift_moves_forward_and_pads_with_zeros():
    t = torch.arange(5.).reshape(1, 1, 5)
    out = shift(t, 1, dim = 2)
    assert out[0, 0].tolist() == [0., 0., 1., 2., 3.]


def test_shift_moves_backward():
    t = torch.arange(5.).reshape(1, 1, 5)
    assert shift(t, -1, dim = 2)[0, 0].tolist() == [1., 2., 3., 4., 0.]


def test_shift_of_zero_is_identity():
    t = torch.rand(1, 1, 5)
    assert torch.equal(shift(t, 0, dim = 2), t)


def test_shift_larger_than_the_axis_is_identity():
    t = torch.rand(1, 1, 3)
    assert torch.equal(shift(t, 9, dim = 2), t)


def test_token_shift_preserves_shape():
    x = torch.rand(2, 16, 4, 8, 8)
    assert token_shift(x).shape == x.shape
    assert token_shift(x, shift_space = True).shape == x.shape


def test_token_shift_leaves_upper_channels_untouched():
    x = torch.rand(1, 16, 4, 8, 8)
    out = token_shift(x)
    # half the channels are shiftable, rounded down to a multiple of the direction count
    assert torch.equal(out[:, 8:], x[:, 8:])
    assert not torch.equal(out[:, :8], x[:, :8])


def test_token_shift_is_a_noop_on_too_few_channels():
    x = torch.rand(1, 1, 4, 8, 8)
    assert torch.equal(token_shift(x), x)


def test_token_shift_adds_no_parameters():
    plain = make()
    shifted = make(token_shift = 'space-time')
    assert sum(p.numel() for p in plain.parameters()) == sum(p.numel() for p in shifted.parameters())


def test_token_shift_checkpoint_is_interchangeable():
    """No parameters means a shifted model loads a plain checkpoint, and the reverse."""
    plain, shifted = make(), make(token_shift = 'time')
    shifted.load_state_dict(plain.state_dict())
    plain.load_state_dict(shifted.state_dict())


@pytest.mark.parametrize('mode', ['time', 'space-time'])
def test_token_shift_changes_the_output(mode):
    torch.manual_seed(0)
    plain = make()
    torch.manual_seed(0)
    shifted = make(token_shift = mode)

    x, t = torch.rand(1, 3, 4, 16, 16), torch.tensor([1])
    assert not torch.allclose(plain(x, t), shifted(x, t))


def test_unknown_token_shift_is_rejected():
    with pytest.raises(AssertionError, match = 'unknown token_shift'):
        ResnetBlock(4, 4, token_shift = 'sideways')


def test_token_shift_trains(device):
    unet = make(device, token_shift = 'space-time')
    diffusion = VideoDiffusion(unet, image_size = 16, num_frames = 4, timesteps = 5).to(device)
    loss = diffusion(torch.rand(1, 3, 4, 16, 16, device = device))
    loss.backward()
    assert torch.isfinite(loss)


# ------------------------------------------------------------ text memory tokens


def test_cond_tokens_off_by_default():
    assert make(cond_dim = 4).num_cond_tokens == 0


def test_cond_tokens_require_conditioning():
    # no cond_dim means nothing to project, whatever was asked for
    assert make(num_cond_tokens = 4).num_cond_tokens == 0


def test_cond_tokens_add_parameters():
    without = make(cond_dim = 4)
    with_tokens = make(cond_dim = 4, num_cond_tokens = 4)
    assert sum(p.numel() for p in with_tokens.parameters()) > sum(p.numel() for p in without.parameters())


@pytest.mark.parametrize('num_tokens', [4, 8])
def test_forward_with_cond_tokens(num_tokens, device):
    unet = make(device, cond_dim = 4, num_cond_tokens = num_tokens)
    out = unet(torch.rand(2, 3, 4, 16, 16, device = device), torch.tensor([1, 2], device = device),
               cond = torch.randn(2, 4, device = device))
    assert out.shape == (2, 3, 4, 16, 16) and torch.isfinite(out).all()


def test_the_caption_actually_changes_the_output():
    torch.manual_seed(0)
    unet = make(cond_dim = 4, num_cond_tokens = 4)
    x, t = torch.rand(1, 3, 4, 16, 16), torch.tensor([1])

    a = unet(x, t, cond = torch.zeros(1, 4))
    b = unet(x, t, cond = torch.ones(1, 4))
    assert not torch.allclose(a, b), 'memory tokens are not influencing attention'


def test_cond_tokens_work_with_classifier_free_guidance(device):
    unet = make(device, cond_dim = 4, num_cond_tokens = 4)
    out = unet.forward_with_cond_scale(
        torch.rand(1, 3, 4, 16, 16, device = device), torch.tensor([1], device = device),
        cond = torch.randn(1, 4, device = device), cond_scale = 2.
    )
    assert torch.isfinite(out).all()


def test_cond_tokens_survive_focus_present():
    unet = make(cond_dim = 4, num_cond_tokens = 4)
    x, t, cond = torch.rand(2, 3, 4, 16, 16), torch.tensor([1, 2]), torch.randn(2, 4)

    # a mix of arrested and free samples exercises the mask-padding path
    mask = torch.tensor([True, False])
    assert torch.isfinite(unet(x, t, cond = cond, focus_present_mask = mask)).all()
    assert torch.isfinite(unet(x, t, cond = cond, prob_focus_present = 1.)).all()


def test_cond_tokens_sample_end_to_end(device):
    unet = make(device, cond_dim = 4, num_cond_tokens = 4)
    diffusion = VideoDiffusion(unet, image_size = 16, num_frames = 4, timesteps = 10).to(device)

    cond = torch.randn(1, 4, device = device)
    out = diffusion.sample(cond = cond, sampling_timesteps = 3, progress = False)
    assert out.shape == (1, 3, 4, 16, 16) and torch.isfinite(out).all()


def test_both_features_together(device):
    unet = make(device, cond_dim = 4, num_cond_tokens = 8, token_shift = 'space-time')
    diffusion = VideoDiffusion(unet, image_size = 16, num_frames = 4, timesteps = 5).to(device)

    cond = torch.randn(2, 4, device = device)
    loss = diffusion(torch.rand(2, 3, 4, 16, 16, device = device), cond = cond)
    loss.backward()
    assert torch.isfinite(loss)
