import pytest
import torch

from kinema import Trainer, Unet3D, VideoDiffusion, video_tensor_to_gif
from kinema.autoencoder import FrameAutoencoder
from kinema.latent import LatentDiffusion, fit_latent_scale


def make_parts(image_size = 32, levels = 2, latent_channels = 4, num_frames = 4):
    autoencoder = FrameAutoencoder(latent_channels = latent_channels, dim = 16, levels = levels)
    latent_size = autoencoder.latent_shape(image_size)

    unet = Unet3D(
        dim = 16, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8,
        channels = latent_channels, out_dim = latent_channels
    )
    inner = VideoDiffusion(
        unet, image_size = latent_size, num_frames = num_frames,
        timesteps = 20, channels = latent_channels
    )
    return inner, autoencoder


def make(image_size = 32, **kw):
    inner, autoencoder = make_parts(image_size, **kw)
    return LatentDiffusion(inner, autoencoder, image_size = image_size)


# ---------------------------------------------------------------- autoencoder


@pytest.mark.parametrize('levels', [1, 2, 3])
def test_round_trip_preserves_shape(levels):
    autoencoder = FrameAutoencoder(latent_channels = 4, dim = 8, levels = levels)
    video = torch.rand(2, 3, 4, 32, 32) * 2 - 1

    latents = autoencoder.encode(video)
    assert latents.shape == (2, 4, 4, 32 // 2 ** levels, 32 // 2 ** levels)
    assert autoencoder.decode(latents).shape == video.shape


def test_time_axis_is_untouched():
    """Compression is spatial only — temporal structure is the U-Net's job."""
    autoencoder = FrameAutoencoder(dim = 8, levels = 2)
    latents = autoencoder.encode(torch.rand(1, 3, 7, 32, 32))
    assert latents.shape[2] == 7


def test_latent_shape_rejects_indivisible_sizes():
    autoencoder = FrameAutoencoder(dim = 8, levels = 2)
    with pytest.raises(AssertionError, match = 'divisible'):
        autoencoder.latent_shape(30)


def test_autoencoder_learns_to_reconstruct():
    """A metric that never improves is measuring nothing; the same goes for a trainable module."""
    torch.manual_seed(0)
    autoencoder = FrameAutoencoder(latent_channels = 8, dim = 16, levels = 1)
    video = torch.rand(1, 3, 2, 16, 16) * 2 - 1

    before = autoencoder.reconstruction_loss(video).item()

    opt = torch.optim.Adam(autoencoder.parameters(), lr = 1e-3)
    for _ in range(60):
        loss = autoencoder.reconstruction_loss(video)
        loss.backward()
        opt.step()
        opt.zero_grad()

    assert autoencoder.reconstruction_loss(video).item() < before


def test_measure_scale_is_positive():
    autoencoder = FrameAutoencoder(dim = 8, levels = 2)
    assert autoencoder.measure_scale(torch.rand(2, 3, 4, 32, 32)) > 0


def test_levels_must_be_at_least_one():
    with pytest.raises(AssertionError, match = 'levels'):
        FrameAutoencoder(levels = 0)


# ------------------------------------------------------------ latent diffusion


def test_surface_is_described_in_pixels():
    """Trainer reads these; they must describe pixels or the dataset loads the wrong size."""
    model = make(image_size = 32)
    assert (model.image_size, model.channels, model.num_frames) == (32, 3, 4)


def test_mismatched_latent_size_is_rejected():
    inner, autoencoder = make_parts(image_size = 32)
    with pytest.raises(AssertionError, match = 'compresses to'):
        LatentDiffusion(inner, autoencoder, image_size = 64)


def test_mismatched_channels_are_rejected():
    autoencoder = FrameAutoencoder(latent_channels = 8, dim = 16, levels = 2)
    unet = Unet3D(dim = 16, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8, channels = 4, out_dim = 4)
    inner = VideoDiffusion(unet, image_size = 8, num_frames = 4, timesteps = 20, channels = 4)

    with pytest.raises(AssertionError, match = 'latent channels'):
        LatentDiffusion(inner, autoencoder, image_size = 32)


def test_training_leaves_the_autoencoder_frozen():
    model = make()
    model(torch.rand(2, 3, 4, 32, 32)).backward()

    assert all(p.grad is None for p in model.autoencoder.parameters())
    assert any(p.grad is not None for p in model.diffusion.parameters())


def test_autoencoder_can_be_left_trainable():
    inner, autoencoder = make_parts()
    model = LatentDiffusion(inner, autoencoder, image_size = 32, freeze_autoencoder = False)
    model(torch.rand(1, 3, 4, 32, 32)).backward()

    assert any(p.grad is not None for p in model.autoencoder.parameters())


def test_sampling_returns_pixels_in_range(device):
    model = make().to(device)
    out = model.sample(batch_size = 2, sampling_timesteps = 3, progress = False)

    assert out.shape == (2, 3, 4, 32, 32)
    assert out.min() >= 0. and out.max() <= 1.


def test_encode_decode_round_trips_through_the_scale():
    model = make()
    videos = torch.rand(1, 3, 4, 32, 32)

    latents = model.encode(videos)
    assert latents.shape == (1, 4, 4, 8, 8)
    assert model.decode(latents).shape == videos.shape


def test_latent_scale_must_be_positive():
    inner, autoencoder = make_parts()
    with pytest.raises(AssertionError, match = 'latent_scale'):
        LatentDiffusion(inner, autoencoder, image_size = 32, latent_scale = 0.)


def test_fit_latent_scale_measures_something_usable():
    _, autoencoder = make_parts()
    assert fit_latent_scale(autoencoder, torch.rand(2, 3, 4, 32, 32)) > 0


def test_interpolation_works_in_latent_space():
    model = make()
    a = torch.rand(1, 3, 4, 32, 32)
    out = model.interpolate(a, a, t = 10, sampling_timesteps = 3, progress = False)
    assert out.shape == a.shape and torch.isfinite(out).all()


def test_it_trains_through_the_ordinary_trainer(tmp_path):
    """The whole point of the pixel-space surface: Trainer needs no changes at all."""
    for i in range(2):
        video_tensor_to_gif(torch.rand(3, 4, 32, 32), str(tmp_path / f'{i}.gif'))

    model = make()
    trainer = Trainer(
        model, tmp_path, device = 'cpu', train_batch_size = 1, train_num_steps = 3,
        gradient_accumulate_every = 1, results_folder = tmp_path / 'r',
        save_and_sample_every = 2, num_sample_rows = 1, step_start_ema = 1
    )

    logs = []
    trainer.train(log_fn = logs.append)

    assert len(logs) == 3
    assert (tmp_path / 'r' / 'model-1.pt').exists()
    assert (tmp_path / 'r' / '1.gif').exists(), 'samples should be written at pixel resolution'

    trainer.load(-1)
    assert trainer.step == 2


def test_evaluate_can_score_a_latent_model(tmp_path):
    from kinema import Dataset
    from kinema.evaluate import deterministic_loss

    for i in range(2):
        video_tensor_to_gif(torch.rand(3, 4, 32, 32), str(tmp_path / f'{i}.gif'))

    model = make()
    dataset = Dataset(tmp_path, 32, num_frames = 4)

    first = deterministic_loss(model, dataset, num_problems = 2)
    second = deterministic_loss(model, dataset, num_problems = 2)
    assert first == second
