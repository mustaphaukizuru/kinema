import pytest
import torch

from kinema import video_tensor_to_gif

pytest.importorskip('tensorboard', reason = 'TensorBoard is needed for monitoring')

from kinema.monitor import TensorBoardLogger  # noqa: E402  (after the skip guard)


def event_files(log_dir):
    return list(log_dir.glob('**/events.out.tfevents.*'))


def test_writes_an_event_file(tmp_path):
    with TensorBoardLogger(tmp_path) as tb:
        tb({'step': 0, 'loss': 0.5})

    assert event_files(tmp_path), 'expected a tfevents file'


def test_logs_every_numeric_field(tmp_path):
    with TensorBoardLogger(tmp_path, flush_every = 1) as tb:
        tb({'step': 1, 'loss': 0.5, 'grad_norm': 1.25})

    assert event_files(tmp_path)


def test_non_numeric_fields_are_skipped(tmp_path):
    # 'sample' is a path string; it must not reach add_scalar
    (tmp_path / 'clip.gif').write_bytes(b'')
    with TensorBoardLogger(tmp_path, flush_every = 1) as tb:
        tb({'step': 1, 'loss': 0.5, 'note': 'not a number'})

    assert event_files(tmp_path)


def logged_tags(log_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(event_files(log_dir)[0]))
    acc.Reload()
    return acc.Tags()


def test_sample_clip_is_logged(tmp_path):
    """
    Regression: torch's add_video *prints and returns* when moviepy is missing rather than
    raising, so a try/except fallback never fires and the sample vanishes silently. The clip
    must reach the event file by one route or the other.
    """
    clip = tmp_path / 'sample.gif'
    video_tensor_to_gif(torch.rand(3, 4, 16, 16), str(clip))

    log_dir = tmp_path / 'tb'
    with TensorBoardLogger(log_dir, flush_every = 1) as tb:
        video_route = tb._video_supported
        tb({'step': 100, 'loss': 0.1, 'sample': str(clip)})

    tags = logged_tags(log_dir)
    assert tags['scalars'] == ['train/loss']

    if video_route:
        assert tags['images'] == [] and 'samples/video' in str(tags)
    else:
        assert tags['images'], 'sample frames were dropped instead of logged'


def test_scalars_reach_the_event_file(tmp_path):
    with TensorBoardLogger(tmp_path, flush_every = 1) as tb:
        tb({'step': 0, 'loss': 0.9})
        tb({'step': 1, 'loss': 0.4})

    assert logged_tags(tmp_path)['scalars'] == ['train/loss']


def test_unreadable_sample_warns_without_raising(tmp_path, caplog):
    with caplog.at_level('WARNING'), TensorBoardLogger(tmp_path / 'tb') as tb:
        tb({'step': 1, 'loss': 0.1, 'sample': str(tmp_path / 'missing.gif')})

    assert 'could not read sample' in caplog.text


def test_step_defaults_to_the_call_count(tmp_path):
    with TensorBoardLogger(tmp_path, flush_every = 1) as tb:
        tb({'loss': 0.5})
        tb({'loss': 0.4})
        assert tb._calls == 2

    assert event_files(tmp_path)


def test_trainer_log_dict_carries_the_step(tmp_path):
    from kinema import Trainer, Unet3D, VideoDiffusion

    for i in range(2):
        video_tensor_to_gif(torch.rand(3, 4, 16, 16), str(tmp_path / f'{i}.gif'))

    unet = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8)
    diffusion = VideoDiffusion(unet, image_size = 16, num_frames = 2, timesteps = 5)
    trainer = Trainer(
        diffusion, tmp_path, device = 'cpu', train_batch_size = 1, train_num_steps = 3,
        gradient_accumulate_every = 1, results_folder = tmp_path / 'r', save_and_sample_every = 1000
    )

    logs = []
    trainer.train(log_fn = logs.append)

    assert [log['step'] for log in logs] == [0, 1, 2]


def test_end_to_end_through_the_trainer(tmp_path):
    from kinema import Trainer, Unet3D, VideoDiffusion

    video_tensor_to_gif(torch.rand(3, 4, 16, 16), str(tmp_path / 'a.gif'))

    unet = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8)
    diffusion = VideoDiffusion(unet, image_size = 16, num_frames = 2, timesteps = 5)
    trainer = Trainer(
        diffusion, tmp_path, device = 'cpu', train_batch_size = 1, train_num_steps = 3,
        gradient_accumulate_every = 1, results_folder = tmp_path / 'r',
        save_and_sample_every = 2, num_sample_rows = 1
    )

    log_dir = tmp_path / 'tb'
    with TensorBoardLogger(log_dir, flush_every = 1) as tb:
        trainer.train(log_fn = tb)

    assert event_files(log_dir)
    assert (tmp_path / 'r' / '1.gif').exists(), 'the run should have produced a sample to log'
