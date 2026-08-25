import pytest
import torch

from kinema import Trainer, Unet3D, VideoDiffusion, video_tensor_to_gif
from kinema.cli import _accepted_keys, load_config, main, validate
from kinema.utils import seed_everything

pytest.importorskip('yaml', reason = 'PyYAML is needed for config files')


def make_trainer(tmp_path, **kw):
    video_tensor_to_gif(torch.rand(3, 2, 16, 16), str(tmp_path / 'a.gif'))
    unet = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8)
    diffusion = VideoDiffusion(unet, image_size = 16, num_frames = 2, timesteps = 5)
    kw.setdefault('save_and_sample_every', 1000)
    kw.setdefault('train_num_steps', 2)
    return Trainer(
        diffusion, tmp_path, device = 'cpu', train_batch_size = 1,
        gradient_accumulate_every = 1, results_folder = tmp_path / 'r', **kw
    )


# ------------------------------------------------------------------ seeding


def test_seed_everything_returns_the_seed():
    assert seed_everything(11) == 11


def test_seeding_makes_ddim_sampling_reproducible():
    unet = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8)
    diffusion = VideoDiffusion(unet, image_size = 16, num_frames = 2, timesteps = 10)

    seed_everything(4)
    first = diffusion.sample(batch_size = 1, sampling_timesteps = 3, progress = False)
    seed_everything(4)
    second = diffusion.sample(batch_size = 1, sampling_timesteps = 3, progress = False)

    assert torch.equal(first, second)


# ------------------------------------------------------- checkpoint management


def test_milestones_lists_numbered_checkpoints(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.save(2)
    trainer.save(5)
    (tmp_path / 'r' / 'model-best.pt').write_bytes(b'not ours')

    assert trainer.milestones() == [2, 5]


def test_save_current_never_overwrites(tmp_path):
    """An interrupted run must add to the record, not replace a milestone that may be better."""
    trainer = make_trainer(tmp_path, save_and_sample_every = 1)
    trainer.save(0)
    before = (tmp_path / 'r' / 'model-0.pt').read_bytes()

    trainer.train()               # reaches step 2
    milestone = trainer.save_current()

    assert milestone != 0, 'it reused an occupied milestone'
    assert (tmp_path / 'r' / 'model-0.pt').read_bytes() == before


def test_keep_last_n_prunes_older_checkpoints(tmp_path):
    trainer = make_trainer(tmp_path, keep_last_n = 2)
    for milestone in range(5):
        trainer.save(milestone)

    assert trainer.milestones() == [3, 4]


def test_keep_last_n_is_off_by_default(tmp_path):
    trainer = make_trainer(tmp_path)
    for milestone in range(4):
        trainer.save(milestone)

    assert trainer.milestones() == [0, 1, 2, 3]


def test_pruning_leaves_unnumbered_files_alone(tmp_path):
    trainer = make_trainer(tmp_path, keep_last_n = 1)
    trainer.save(0)
    stray = tmp_path / 'r' / 'model-best.pt'
    stray.write_bytes(b'not ours')

    trainer.save(1)
    assert stray.exists()
    assert trainer.milestones() == [1]


def test_resume_still_finds_the_newest_after_pruning(tmp_path):
    trainer = make_trainer(tmp_path, keep_last_n = 2)
    trainer.train()
    trainer.save(0)
    trainer.save(1)

    fresh = make_trainer(tmp_path, keep_last_n = 2)
    fresh.load(-1)
    assert fresh.step == trainer.step


# ------------------------------------------------------------ mixed precision


def test_bfloat16_is_accepted(tmp_path):
    trainer = make_trainer(tmp_path, amp_dtype = 'bfloat16')
    assert trainer.amp_dtype is torch.bfloat16


def test_bfloat16_disables_the_grad_scaler(tmp_path):
    """bf16 has float32's exponent range, so loss scaling is unnecessary."""
    trainer = make_trainer(tmp_path, amp = True, amp_dtype = 'bfloat16')
    assert not trainer.scaler.is_enabled()


def test_float16_keeps_the_grad_scaler(tmp_path):
    trainer = make_trainer(tmp_path, amp = True, amp_dtype = 'float16')
    assert trainer.scaler.is_enabled()


def test_unknown_amp_dtype_is_rejected(tmp_path):
    with pytest.raises(AssertionError, match = 'amp_dtype'):
        make_trainer(tmp_path, amp_dtype = 'float8')


def test_bfloat16_trains(tmp_path):
    trainer = make_trainer(tmp_path, amp = True, amp_dtype = 'bfloat16')
    logs = []
    trainer.train(log_fn = logs.append)
    assert len(logs) == 2 and all(torch.isfinite(torch.tensor(log['loss'])) for log in logs)


# -------------------------------------------------------- config validation


def test_accepted_keys_skips_positional_arguments():
    keys = _accepted_keys(Unet3D.__init__, skip = 1)
    assert 'dim' in keys and 'self' not in keys


def test_valid_config_passes():
    assert validate(load_config()) is not None


def test_unknown_key_is_rejected_with_a_suggestion():
    with pytest.raises(SystemExit, match = 'train_lr'):
        validate(load_config(overrides = ['trainer.train_lrr=3e-4']))


def test_unknown_section_is_rejected():
    with pytest.raises(SystemExit, match = 'unknown config section'):
        validate(load_config(overrides = ['trainr.foo=1']))


def test_non_mapping_section_is_rejected():
    with pytest.raises(SystemExit, match = 'should be a mapping'):
        validate({'model': 5})


def test_every_documented_config_key_is_real():
    """configs/moving-mnist.yaml ships with the package; it must not drift from the code."""
    from pathlib import Path

    import yaml

    shipped = yaml.safe_load(Path('configs/moving-mnist.yaml').read_text(encoding = 'utf-8'))
    validate(shipped)


def test_cli_rejects_a_typo_before_doing_any_work(tmp_path):
    with pytest.raises(SystemExit):
        main(['train', '--device', 'cpu', '--set', 'trainer.nonsense=1'])


# ------------------------------------------------------------ release metadata


def test_citation_version_matches_the_package():
    """Four files carry the version; CITATION.cff is the one that silently drifts."""
    from pathlib import Path

    from kinema import __version__

    citation = Path('CITATION.cff').read_text(encoding = 'utf-8')
    declared = next(
        line.split(':', 1)[1].strip()
        for line in citation.splitlines()
        if line.startswith('version:')
    )

    assert declared == __version__, f'CITATION.cff says {declared}, package says {__version__}'


def test_changelog_documents_the_current_version():
    from pathlib import Path

    from kinema import __version__

    changelog = Path('CHANGELOG.md').read_text(encoding = 'utf-8')
    assert f'## {__version__}' in changelog, f'CHANGELOG.md has no entry for {__version__}'
