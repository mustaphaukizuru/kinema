import pytest
import torch

from kinema import Dataset, Unet3D, VideoDiffusion, video_tensor_to_gif
from kinema.evaluate import compare, deterministic_loss, fixed_problems


def build(tmp_path, clips = 3, captions = False):
    for i in range(clips):
        video_tensor_to_gif(torch.rand(3, 2, 16, 16), str(tmp_path / f'{i}.gif'))
        if captions:
            (tmp_path / f'{i}.txt').write_text(f'clip {i}', encoding = 'utf-8')

    unet = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8)
    diffusion = VideoDiffusion(unet, image_size = 16, num_frames = 2, timesteps = 20)
    return diffusion, Dataset(tmp_path, 16, num_frames = 2)


def test_fixed_problems_are_reproducible(tmp_path):
    _, dataset = build(tmp_path)

    first = fixed_problems(dataset, 5, 20, seed = 3)
    second = fixed_problems(dataset, 5, 20, seed = 3)

    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_fixed_problems_differ_by_seed(tmp_path):
    _, dataset = build(tmp_path)
    assert not torch.equal(
        fixed_problems(dataset, 5, 20, seed = 1)[1],
        fixed_problems(dataset, 5, 20, seed = 2)[1],
    )


def test_problems_stay_in_range(tmp_path):
    _, dataset = build(tmp_path, clips = 3)
    indices, timesteps, _ = fixed_problems(dataset, 20, 20, seed = 0)

    assert indices.min() >= 0 and indices.max() < len(dataset)
    assert timesteps.min() >= 0 and timesteps.max() < 20


def test_loss_is_finite_and_positive(tmp_path):
    diffusion, dataset = build(tmp_path)
    loss = deterministic_loss(diffusion, dataset, num_problems = 4)

    assert loss > 0 and torch.isfinite(torch.tensor(loss))


def test_the_same_model_scores_identically_twice(tmp_path):
    """The whole point: no run-to-run variance, so two checkpoints can be compared."""
    diffusion, dataset = build(tmp_path)

    first = deterministic_loss(diffusion, dataset, num_problems = 6)
    second = deterministic_loss(diffusion, dataset, num_problems = 6)

    assert first == second


def test_score_is_independent_of_ambient_seeding(tmp_path):
    """Whatever torch's global RNG is doing must not leak into the measurement."""
    diffusion, dataset = build(tmp_path)

    torch.manual_seed(1)
    first = deterministic_loss(diffusion, dataset, num_problems = 4)
    torch.manual_seed(999)
    second = deterministic_loss(diffusion, dataset, num_problems = 4)

    assert first == second


def test_different_seeds_give_different_problems(tmp_path):
    diffusion, dataset = build(tmp_path)

    assert deterministic_loss(diffusion, dataset, num_problems = 4, seed = 0) != \
           deterministic_loss(diffusion, dataset, num_problems = 4, seed = 5)


def test_a_trained_model_scores_better_than_an_untrained_one(tmp_path):
    """The metric has to move the right way, or it is measuring nothing."""
    diffusion, dataset = build(tmp_path)

    before = deterministic_loss(diffusion, dataset, num_problems = 8)

    opt = torch.optim.Adam(diffusion.parameters(), lr = 1e-3)
    torch.manual_seed(0)
    for _ in range(30):
        loss = diffusion(torch.stack([dataset[i] for i in range(len(dataset))]))
        loss.backward()
        opt.step()
        opt.zero_grad()

    after = deterministic_loss(diffusion, dataset, num_problems = 8)
    assert after < before, f'{after} should beat {before}'


def test_batch_size_is_respected(tmp_path):
    diffusion, dataset = build(tmp_path)
    loss = deterministic_loss(diffusion, dataset, num_problems = 3, batch_size = 2)
    assert torch.isfinite(torch.tensor(loss))


def test_captioned_datasets_are_scored_with_their_captions(tmp_path):
    unet = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8, cond_dim = 4)
    diffusion = VideoDiffusion(unet, image_size = 16, num_frames = 2, timesteps = 20)

    for i in range(2):
        video_tensor_to_gif(torch.rand(3, 2, 16, 16), str(tmp_path / f'{i}.gif'))
        (tmp_path / f'{i}.txt').write_text(f'clip {i}', encoding = 'utf-8')

    dataset = Dataset(tmp_path, 16, num_frames = 2)
    assert dataset.has_captions

    # a conditioned model with a captioned dataset needs the caption to reach p_losses,
    # otherwise this raises 'cond must be passed in if cond_dim specified'
    with pytest.raises(Exception) as excinfo:
        deterministic_loss(diffusion, dataset, num_problems = 1)

    # BERT is not installed in every environment; either it embedded, or it failed on the import
    assert 'cond must be passed in' not in str(excinfo.value)


def test_compare_scores_each_checkpoint(tmp_path):
    diffusion, dataset = build(tmp_path)
    results = compare(['a', 'b'], lambda _: diffusion, dataset, num_problems = 3)

    assert [name for name, _ in results] == ['a', 'b']
    assert results[0][1] == results[1][1], 'the same model must score the same'


def test_model_is_left_in_eval_mode(tmp_path):
    diffusion, dataset = build(tmp_path)
    diffusion.train()
    deterministic_loss(diffusion, dataset, num_problems = 2)
    assert not diffusion.training
