import warnings

import torch

from kinema import Dataset, Trainer, Unet3D, VideoDiffusion, gif_to_tensor, video_tensor_to_gif


def _write_gifs(folder, n = 3, frames = 4):
    for i in range(n):
        video_tensor_to_gif(torch.rand(3, frames, 16, 16), str(folder / f'{i}.gif'))


def test_gif_roundtrip_closes_file(tmp_path):
    path = tmp_path / 'v.gif'
    imgs = video_tensor_to_gif(torch.rand(3, 4, 16, 16), str(path))
    assert len(imgs) == 4
    with warnings.catch_warnings():
        warnings.simplefilter('error', ResourceWarning)
        t = gif_to_tensor(str(path))
    assert t.shape == (3, 4, 16, 16)


def test_dataset_casts_num_frames(tmp_path):
    _write_gifs(tmp_path, frames = 4)
    assert Dataset(tmp_path, 16, num_frames = 2)[0].shape == (3, 2, 16, 16)
    assert Dataset(tmp_path, 16, num_frames = 6)[0].shape == (3, 6, 16, 16)
    assert len(Dataset(tmp_path, 16)) == 3


def test_trainer_train_save_load(tmp_path, device):
    (tmp_path / 'data').mkdir()
    _write_gifs(tmp_path / 'data')
    model = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8)
    diff = VideoDiffusion(model, image_size = 16, num_frames = 4, timesteps = 5)
    trainer = Trainer(
        diff, tmp_path / 'data', device = device, train_batch_size = 2, train_num_steps = 3,
        save_and_sample_every = 2, results_folder = tmp_path / 'results', amp = device == 'cuda',
        num_sample_rows = 1, step_start_ema = 1, update_ema_every = 1, max_grad_norm = 1.
    )
    logs = []
    trainer.train(log_fn = logs.append)
    assert len(logs) == 3 and all('loss' in l for l in logs)
    assert (tmp_path / 'results' / 'model-1.pt').exists()
    assert (tmp_path / 'results' / '1.gif').exists()

    ckpt = torch.load(tmp_path / 'results' / 'model-1.pt', map_location = 'cpu', weights_only = True)
    assert {'step', 'model', 'ema', 'opt', 'scaler', 'version'} <= set(ckpt)

    trainer.load(-1)
    assert trainer.step == 2
    assert next(trainer.model.parameters()).device.type == device
