"""Bugs found by auditing the seams between modules. Each one is pinned here."""

import pytest
import torch

from kinema import Dataset, Trainer, Unet3D, VideoDiffusion, video_tensor_to_gif
from kinema.evaluate import deterministic_loss


@pytest.fixture
def captioned(tmp_path):
    """A folder where every clip has a caption."""
    for i in range(2):
        video_tensor_to_gif(torch.rand(3, 2, 16, 16), str(tmp_path / f'{i}.gif'))
        (tmp_path / f'{i}.txt').write_text(f'caption {i}', encoding = 'utf-8')
    return tmp_path


@pytest.fixture
def no_bert(monkeypatch):
    """Make any attempt to embed text fail loudly, as it would without the text extra."""
    import kinema.diffusion as diffusion_module

    def explode(*args, **kwargs):
        raise AssertionError('embed_text was called for a model that cannot use conditioning')

    monkeypatch.setattr(diffusion_module, 'embed_text', explode)


def unconditional():
    unet = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8)
    return VideoDiffusion(unet, image_size = 16, num_frames = 2, timesteps = 5)


# ------------------------------------------------------------------------------
# captions reaching a model that cannot use them
# ------------------------------------------------------------------------------


def test_has_cond_reports_the_denoiser(captioned):
    assert not unconditional().has_cond

    conditional = VideoDiffusion(
        Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8, cond_dim = 4),
        image_size = 16, num_frames = 2, timesteps = 5
    )
    assert conditional.has_cond


def test_resolve_cond_drops_captions_an_unconditional_model_cannot_use(no_bert):
    assert unconditional().resolve_cond(['a caption']) is None


def test_training_ignores_captions_when_the_model_is_unconditional(captioned, no_bert):
    """
    Regression: a captioned folder plus an unconditional model sent captions into embed_text,
    loading BERT to build an embedding nothing would read — and failing outright on an install
    without the text extra.
    """
    diffusion = unconditional()
    trainer = Trainer(
        diffusion, captioned, device = 'cpu', train_batch_size = 1, train_num_steps = 2,
        gradient_accumulate_every = 1, results_folder = captioned.parent / 'r',
        save_and_sample_every = 1000
    )
    trainer.train()
    assert trainer.step == 2


def test_evaluation_ignores_captions_when_the_model_is_unconditional(captioned, no_bert):
    dataset = Dataset(captioned, 16, num_frames = 2)
    assert dataset.has_captions

    assert deterministic_loss(unconditional(), dataset, num_problems = 2) > 0


def test_sampling_ignores_captions_when_the_model_is_unconditional(no_bert):
    out = unconditional().sample(
        cond = ['a caption'], batch_size = 1, sampling_timesteps = 2, progress = False
    )
    assert out.shape == (1, 3, 2, 16, 16)


def test_interpolation_ignores_captions_when_the_model_is_unconditional(no_bert):
    diffusion = unconditional()
    clip = torch.rand(1, 3, 2, 16, 16)
    out = diffusion.interpolate(clip, clip, t = 3, cond = ['a caption'], progress = False)
    assert out.shape == clip.shape


def test_a_conditional_model_still_embeds_its_captions():
    """The guard must not disable conditioning for models that genuinely use it."""
    import kinema.diffusion as diffusion_module

    seen = []
    original = diffusion_module.embed_text
    diffusion_module.embed_text = lambda texts, device, **kw: (
        seen.append(texts) or torch.randn(len(texts), 4)
    )
    try:
        conditional = VideoDiffusion(
            Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8, cond_dim = 4),
            image_size = 16, num_frames = 2, timesteps = 5
        )
        conditional(torch.rand(1, 3, 2, 16, 16), cond = ['a real caption'])
    finally:
        diffusion_module.embed_text = original

    assert seen == [['a real caption']]


# ------------------------------------------------------------------------------
# checkpoint retention
# ------------------------------------------------------------------------------


def test_keep_last_n_zero_is_rejected(tmp_path):
    """Regression: it deleted the checkpoint that had just been written."""
    video_tensor_to_gif(torch.rand(3, 2, 16, 16), str(tmp_path / 'a.gif'))

    with pytest.raises(AssertionError, match = 'keep_last_n must be at least 1'):
        Trainer(
            unconditional(), tmp_path, device = 'cpu', train_batch_size = 1,
            results_folder = tmp_path / 'r', keep_last_n = 0
        )


def test_keep_last_n_one_keeps_the_newest(tmp_path):
    video_tensor_to_gif(torch.rand(3, 2, 16, 16), str(tmp_path / 'a.gif'))

    trainer = Trainer(
        unconditional(), tmp_path, device = 'cpu', train_batch_size = 1,
        results_folder = tmp_path / 'r', keep_last_n = 1
    )
    trainer.save(0)
    trainer.save(1)

    assert trainer.milestones() == [1]


# ------------------------------------------------------------------------------
# results written into the training data
# ------------------------------------------------------------------------------


def test_results_inside_the_data_folder_warns(tmp_path, caplog):
    """
    Regression: Dataset globs recursively, so samples written under the data folder become
    training data on the next run and the model learns from its own output.
    """
    video_tensor_to_gif(torch.rand(3, 2, 16, 16), str(tmp_path / 'a.gif'))

    with caplog.at_level('WARNING'):
        Trainer(
            unconditional(), tmp_path, device = 'cpu', train_batch_size = 1,
            results_folder = tmp_path / 'results'
        )

    assert 'picked up as training data' in caplog.text


def test_results_outside_the_data_folder_is_quiet(tmp_path, caplog):
    data = tmp_path / 'data'
    data.mkdir()
    video_tensor_to_gif(torch.rand(3, 2, 16, 16), str(data / 'a.gif'))

    with caplog.at_level('WARNING'):
        Trainer(
            unconditional(), data, device = 'cpu', train_batch_size = 1,
            results_folder = tmp_path / 'results'
        )

    assert 'picked up as training data' not in caplog.text
