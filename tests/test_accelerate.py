import pytest
import torch

from kinema import Trainer, Unet3D, VideoDiffusion, video_tensor_to_gif

pytest.importorskip('accelerate', reason = 'Accelerate is needed for distributed training')


@pytest.fixture(autouse = True)
def _fresh_accelerator_state():
    """
    Accelerate keeps its state in a process-wide singleton, so the first Accelerator built in a
    process fixes the mixed-precision mode for every one after it. Tests would otherwise leak
    into each other, and the failure looks like a kinema bug rather than a global.
    """
    from accelerate.state import AcceleratorState

    AcceleratorState._reset_state(True)
    yield
    AcceleratorState._reset_state(True)


def make_trainer(tmp_path, clips = 2, **kw):
    for i in range(clips):
        video_tensor_to_gif(torch.rand(3, 4, 16, 16), str(tmp_path / f'{i}.gif'))

    unet = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8)
    diffusion = VideoDiffusion(unet, image_size = 16, num_frames = 2, timesteps = 5)

    kw.setdefault('train_num_steps', 2)
    return Trainer(
        diffusion, tmp_path, train_batch_size = 1, gradient_accumulate_every = 1,
        results_folder = tmp_path / 'r', save_and_sample_every = 1000, **kw
    )


def test_accelerator_absent_by_default(tmp_path):
    trainer = make_trainer(tmp_path, device = 'cpu')
    assert trainer.accelerator is None
    assert trainer.is_main is True
    assert trainer.unwrapped() is trainer.model


def test_accelerator_is_built_when_requested(tmp_path):
    trainer = make_trainer(tmp_path, accelerate = True)
    assert trainer.accelerator is not None
    assert trainer.device == trainer.accelerator.device


def test_explicit_device_is_ignored_under_accelerate(tmp_path, caplog):
    with caplog.at_level('INFO'):
        trainer = make_trainer(tmp_path, accelerate = True, device = 'cpu')

    assert trainer.device == trainer.accelerator.device
    assert 'is ignored' in caplog.text


def test_training_runs_through_accelerate(tmp_path):
    trainer = make_trainer(tmp_path, accelerate = True)
    logs = []
    trainer.train(log_fn = logs.append)

    assert len(logs) == 2
    assert all(torch.isfinite(torch.tensor(log['loss'])) for log in logs)


def test_parameters_actually_move(tmp_path):
    trainer = make_trainer(tmp_path, accelerate = True, train_num_steps = 3)
    before = next(trainer.unwrapped().parameters()).detach().clone()
    trainer.train()
    after = next(trainer.unwrapped().parameters()).detach()

    assert not torch.allclose(before, after), 'the optimizer step did not take effect'


def test_gradient_clipping_path(tmp_path):
    trainer = make_trainer(tmp_path, accelerate = True, max_grad_norm = 1.)
    trainer.train()
    assert trainer.step == 2


def test_checkpoint_round_trips_between_modes(tmp_path):
    """A checkpoint written under Accelerate must load into a plain trainer and vice versa."""
    accelerated = make_trainer(tmp_path, accelerate = True, train_num_steps = 2)
    accelerated.train()
    accelerated.save(7)

    checkpoint = torch.load(tmp_path / 'r' / 'model-7.pt', map_location = 'cpu', weights_only = True)
    assert not any(key.startswith('module.') for key in checkpoint['model']), 'DDP prefix leaked'

    plain = make_trainer(tmp_path, device = 'cpu')
    plain.load(7)
    assert plain.step == accelerated.step


def test_amp_selects_fp16_mixed_precision(tmp_path):
    trainer = make_trainer(tmp_path, accelerate = True, amp = True)
    assert trainer.accelerator.mixed_precision == 'fp16'


def test_no_amp_selects_no_mixed_precision(tmp_path):
    trainer = make_trainer(tmp_path, accelerate = True, amp = False)
    assert trainer.accelerator.mixed_precision == 'no'


def test_single_process_is_the_main_process(tmp_path):
    trainer = make_trainer(tmp_path, accelerate = True)
    assert trainer.is_main
    assert trainer.accelerator.num_processes == 1


def test_captions_survive_the_prepared_dataloader(tmp_path):
    from kinema.utils import is_list_str

    for i in range(2):
        (tmp_path / f'{i}.txt').write_text(f'caption {i}', encoding = 'utf-8')

    trainer = make_trainer(tmp_path, accelerate = True)
    assert trainer.ds.has_captions

    batch = next(trainer.dl)
    assert isinstance(batch, (list, tuple)) and len(batch) == 2
    assert is_list_str(batch[1])
