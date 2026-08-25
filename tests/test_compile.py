import torch

from kinema import Trainer, Unet3D, VideoDiffusion, video_tensor_to_gif
from kinema.utils import compile_supported


def make_trainer(tmp_path, **kw):
    video_tensor_to_gif(torch.rand(3, 4, 16, 16), str(tmp_path / 'a.gif'))
    unet = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8)
    diffusion = VideoDiffusion(unet, image_size = 16, num_frames = 2, timesteps = 5)
    return Trainer(
        diffusion, tmp_path, device = 'cpu', train_batch_size = 1, train_num_steps = 2,
        gradient_accumulate_every = 1, results_folder = tmp_path / 'r',
        save_and_sample_every = 1000, **kw
    )


def test_compile_supported_is_a_cached_bool():
    first = compile_supported()
    assert isinstance(first, bool)
    assert compile_supported() is first


def test_compile_off_by_default(tmp_path):
    assert make_trainer(tmp_path).compiled_model is None


def test_compile_request_matches_platform_support(tmp_path):
    trainer = make_trainer(tmp_path, compile = True)
    assert (trainer.compiled_model is not None) == compile_supported()


def test_unsupported_compile_warns_rather_than_raising(tmp_path, caplog):
    if compile_supported():
        return  # nothing to assert where compilation works

    with caplog.at_level('WARNING'):
        trainer = make_trainer(tmp_path, compile = True)

    assert trainer.compiled_model is None
    assert 'training eagerly' in caplog.text


def test_training_works_either_way(tmp_path):
    trainer = make_trainer(tmp_path, compile = True)
    logs = []
    trainer.train(log_fn = logs.append)

    assert len(logs) == 2
    assert all(torch.isfinite(torch.tensor(log['loss'])) for log in logs)


def test_checkpoints_have_no_orig_mod_prefix(tmp_path):
    """torch.compile wraps the module; a naive implementation leaks `_orig_mod.` into every key."""
    trainer = make_trainer(tmp_path, compile = True)
    trainer.train()
    trainer.save(0)

    checkpoint = torch.load(tmp_path / 'r' / 'model-0.pt', map_location = 'cpu', weights_only = True)
    for section in ('model', 'ema'):
        assert not any(key.startswith('_orig_mod.') for key in checkpoint[section]), section


def test_compiled_checkpoint_loads_into_an_uncompiled_trainer(tmp_path):
    trainer = make_trainer(tmp_path, compile = True)
    trainer.train()
    trainer.save(3)

    plain = make_trainer(tmp_path)
    plain.load(3)
    assert plain.step == trainer.step


def test_compiled_model_shares_parameters_with_the_original(tmp_path):
    trainer = make_trainer(tmp_path, compile = True)
    if trainer.compiled_model is None:
        return

    original = next(trainer.model.parameters())
    compiled = next(trainer.compiled_model.parameters())
    assert original is compiled
