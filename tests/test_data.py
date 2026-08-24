import pytest
import torch
from PIL import Image

from kinema import Dataset, frames_to_tensor, read_clip, video_tensor_to_gif, video_to_tensor
from kinema.data import natural_key, video_tensor_to_mp4

av = pytest.importorskip('av', reason = 'PyAV is needed for container formats')


def clip(frames = 4, size = 16):
    return torch.rand(3, frames, size, size)


def test_natural_key_orders_frames_numerically():
    names = ['frame10.png', 'frame2.png', 'frame1.png']
    assert sorted(names, key = natural_key) == ['frame1.png', 'frame2.png', 'frame10.png']


def test_mp4_roundtrip(tmp_path):
    path = tmp_path / 'clip.mp4'
    video_tensor_to_mp4(clip(frames = 6), str(path))
    assert path.exists() and path.stat().st_size > 0

    out = video_to_tensor(path)
    assert out.shape[0] == 3 and out.shape[1] == 6 and out.shape[2:] == (16, 16)
    assert out.min() >= 0. and out.max() <= 1.


def test_mp4_crops_odd_dimensions(tmp_path):
    path = tmp_path / 'odd.mp4'
    video_tensor_to_mp4(torch.rand(3, 4, 15, 15), str(path))
    assert video_to_tensor(path).shape[2:] == (14, 14)


def test_frames_folder_reads_in_natural_order(tmp_path):
    for i in [1, 2, 10]:
        Image.new('RGB', (16, 16), color = (i, i, i)).save(tmp_path / f'frame{i}.png')

    out = frames_to_tensor(tmp_path)
    assert out.shape == (3, 3, 16, 16)
    # frame1 < frame2 < frame10, so the channel means must ascend
    means = [out[:, i].mean().item() for i in range(3)]
    assert means == sorted(means)


def test_frames_folder_without_images_raises(tmp_path):
    with pytest.raises(ValueError, match = 'no image frames'):
        frames_to_tensor(tmp_path)


def test_read_clip_dispatches_on_format(tmp_path):
    gif, mp4 = tmp_path / 'a.gif', tmp_path / 'b.mp4'
    video_tensor_to_gif(clip(), str(gif))
    video_tensor_to_mp4(clip(), str(mp4))

    frames = tmp_path / 'frames'
    frames.mkdir()
    Image.new('RGB', (16, 16)).save(frames / '1.png')

    assert read_clip(gif).shape[0] == 3
    assert read_clip(mp4).shape[0] == 3
    assert read_clip(frames).shape == (3, 1, 16, 16)


def test_read_clip_rejects_unknown_format(tmp_path):
    path = tmp_path / 'notes.txt'
    path.write_text('not a video')
    with pytest.raises(ValueError, match = 'unsupported clip format'):
        read_clip(path)


def test_dataset_mixes_gif_and_mp4(tmp_path):
    video_tensor_to_gif(clip(frames = 6), str(tmp_path / 'a.gif'))
    video_tensor_to_mp4(clip(frames = 6), str(tmp_path / 'b.mp4'))

    ds = Dataset(tmp_path, image_size = 8, num_frames = 4)
    assert len(ds) == 2
    assert all(item.shape == (3, 4, 8, 8) for item in ds)


def test_dataset_falls_back_to_frame_folders(tmp_path):
    for name in ['clip1', 'clip2']:
        folder = tmp_path / name
        folder.mkdir()
        for i in range(3):
            Image.new('RGB', (16, 16)).save(folder / f'{i}.png')

    ds = Dataset(tmp_path, image_size = 8, num_frames = 3)
    assert len(ds) == 2
    assert ds[0].shape == (3, 3, 8, 8)


def test_dataset_pads_short_clips(tmp_path):
    video_tensor_to_mp4(clip(frames = 2), str(tmp_path / 'short.mp4'))
    ds = Dataset(tmp_path, image_size = 8, num_frames = 5)
    assert ds[0].shape == (3, 5, 8, 8)
