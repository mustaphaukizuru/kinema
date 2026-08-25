import pytest
import torch

from kinema import Trainer, Unet3D, VideoDiffusion, video_tensor_to_gif
from kinema.cli import main

pytest.importorskip('yaml', reason = 'PyYAML is needed for config files')


def write_setup(tmp_path):
    clips = tmp_path / 'clips'
    clips.mkdir()
    video_tensor_to_gif(torch.rand(3, 2, 16, 16), str(clips / '0.gif'))

    config = tmp_path / 'config.yaml'
    config.write_text(
        'model:\n'
        '  dim: 8\n'
        '  dim_mults: [1, 2]\n'
        '  attn_heads: 2\n'
        '  attn_dim_head: 8\n'
        'diffusion:\n'
        '  image_size: 16\n'
        '  num_frames: 2\n'
        '  timesteps: 10\n'
        '  sampling_timesteps: 3\n',
        encoding = 'utf-8'
    )
    return clips, config


def make_trainer(tmp_path, **kw):
    video_tensor_to_gif(torch.rand(3, 2, 16, 16), str(tmp_path / 'a.gif'))
    unet = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8)
    diffusion = VideoDiffusion(unet, image_size = 16, num_frames = 2, timesteps = 5)
    kw.setdefault('save_and_sample_every', 1000)
    return Trainer(
        diffusion, tmp_path, device = 'cpu', train_batch_size = 1, train_num_steps = 5,
        gradient_accumulate_every = 1, results_folder = tmp_path / 'r', **kw
    )


def test_fortran_console_handler_is_disabled():
    """
    PyTorch's Windows wheels bundle an Intel Fortran runtime whose console handler aborts the
    process on Ctrl+C before Python can raise KeyboardInterrupt, losing everything since the
    last checkpoint. Importing the CLI must switch it off.
    """
    import os

    import kinema.cli  # noqa: F401  (imported for the side effect on os.environ)

    assert os.environ.get('FOR_DISABLE_CONSOLE_CTRL_HANDLER') == '1'


def test_save_current_writes_a_resumable_checkpoint(tmp_path):
    trainer = make_trainer(tmp_path, save_and_sample_every = 100)
    trainer.train()

    milestone = trainer.save_current()
    assert (tmp_path / 'r' / f'model-{milestone}.pt').exists()

    resumed = make_trainer(tmp_path, save_and_sample_every = 100)
    resumed.load(-1)
    assert resumed.step == trainer.step


def test_load_latest_ignores_unnumbered_checkpoints(tmp_path):
    """A stray .pt with a non-numeric name must not break --resume."""
    trainer = make_trainer(tmp_path, save_and_sample_every = 100)
    trainer.save(2)
    (tmp_path / 'r' / 'model-best.pt').write_bytes(b'not a checkpoint')

    trainer.load(-1)
    assert trainer.step == 0


def test_no_numbered_checkpoints_gives_a_clear_error(tmp_path):
    trainer = make_trainer(tmp_path)
    (tmp_path / 'r' / 'model-best.pt').write_bytes(b'not a checkpoint')

    with pytest.raises(AssertionError, match = 'no numbered checkpoints'):
        trainer.load(-1)


def test_interrupt_saves_and_reports_130(tmp_path, monkeypatch, capsys):
    """Ctrl+C mid-run must checkpoint rather than throw the work away."""
    clips, config = write_setup(tmp_path)
    results = tmp_path / 'results'

    real_train = Trainer.train

    def interrupted(self, *args, **kwargs):
        # let a few real steps happen, then behave as Ctrl+C does
        self.train_num_steps = 3
        real_train(self, *args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(Trainer, 'train', interrupted)

    code = main([
        'train', '-c', str(config), '--device', 'cpu',
        '--set', f'data.folder={clips}',
        '--set', f'trainer.results_folder={results}',
        '--set', 'trainer.train_batch_size=1',
        '--set', 'trainer.gradient_accumulate_every=1',
        '--set', 'trainer.save_and_sample_every=100000',   # never checkpoints on its own
    ])

    assert code == 130, 'SIGINT should exit 130'

    saved = list(results.glob('*.pt'))
    assert saved, 'the interrupt handler wrote no checkpoint'

    out = capsys.readouterr().out
    assert 'interrupted at step 3' in out
    assert 'resume this run with' in out


def test_interrupted_checkpoint_resumes(tmp_path, monkeypatch):
    clips, config = write_setup(tmp_path)
    results = tmp_path / 'results'

    real_train = Trainer.train

    def interrupted(self, *args, **kwargs):
        self.train_num_steps = 4
        real_train(self, *args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(Trainer, 'train', interrupted)

    common = [
        '--set', f'data.folder={clips}',
        '--set', f'trainer.results_folder={results}',
        '--set', 'trainer.train_batch_size=1',
        '--set', 'trainer.gradient_accumulate_every=1',
        '--set', 'trainer.save_and_sample_every=100000',
    ]
    main(['train', '-c', str(config), '--device', 'cpu', *common])

    monkeypatch.undo()

    # --resume must pick the interrupted checkpoint up and carry on
    code = main([
        'train', '-c', str(config), '--device', 'cpu', '--resume',
        *common, '--set', 'trainer.train_num_steps=6',
    ])
    assert code == 0
