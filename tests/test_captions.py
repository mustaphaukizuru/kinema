import pytest
import torch
from PIL import Image
from torch.utils import data

from kinema import Dataset, caption_for, video_tensor_to_gif
from kinema.utils import is_list_str


def write_clip(folder, name, caption = None, frames = 4):
    video_tensor_to_gif(torch.rand(3, frames, 16, 16), str(folder / f'{name}.gif'))
    if caption is not None:
        (folder / f'{name}.txt').write_text(caption, encoding = 'utf-8')


def test_caption_for_finds_the_sidecar(tmp_path):
    write_clip(tmp_path, 'fireworks', 'fireworks over a harbour')
    assert caption_for(tmp_path / 'fireworks.gif') == 'fireworks over a harbour'


def test_caption_for_strips_whitespace(tmp_path):
    write_clip(tmp_path, 'a', '  a dog running \n')
    assert caption_for(tmp_path / 'a.gif') == 'a dog running'


def test_caption_for_returns_none_without_a_sidecar(tmp_path):
    write_clip(tmp_path, 'a')
    assert caption_for(tmp_path / 'a.gif') is None


def test_caption_for_frame_folder_sibling(tmp_path):
    folder = tmp_path / 'clip'
    folder.mkdir()
    Image.new('RGB', (16, 16)).save(folder / '0.png')
    (tmp_path / 'clip.txt').write_text('a sibling caption', encoding = 'utf-8')
    assert caption_for(folder) == 'a sibling caption'


def test_caption_for_frame_folder_inner_file(tmp_path):
    folder = tmp_path / 'clip'
    folder.mkdir()
    Image.new('RGB', (16, 16)).save(folder / '0.png')
    (folder / 'caption.txt').write_text('an inner caption', encoding = 'utf-8')
    assert caption_for(folder) == 'an inner caption'


def test_dataset_yields_pairs_when_every_clip_is_captioned(tmp_path):
    write_clip(tmp_path, 'a', 'first caption')
    write_clip(tmp_path, 'b', 'second caption')

    ds = Dataset(tmp_path, 16, num_frames = 2)
    assert ds.has_captions

    video, caption = ds[0]
    assert video.shape == (3, 2, 16, 16)
    assert caption in {'first caption', 'second caption'}


def test_dataset_stays_unconditional_without_captions(tmp_path):
    write_clip(tmp_path, 'a')
    ds = Dataset(tmp_path, 16, num_frames = 2)
    assert not ds.has_captions
    assert isinstance(ds[0], torch.Tensor)


def test_partial_captions_are_ignored_with_a_warning(tmp_path, caplog):
    write_clip(tmp_path, 'a', 'only this one has a caption')
    write_clip(tmp_path, 'b')

    with caplog.at_level('WARNING'):
        ds = Dataset(tmp_path, 16, num_frames = 2)

    assert not ds.has_captions
    assert 'captions are ignored' in caplog.text


def test_captions_true_requires_every_caption(tmp_path):
    write_clip(tmp_path, 'a', 'captioned')
    write_clip(tmp_path, 'b')

    with pytest.raises(ValueError, match = 'no .txt sidecar'):
        Dataset(tmp_path, 16, num_frames = 2, captions = True)


def test_captions_false_ignores_present_sidecars(tmp_path):
    write_clip(tmp_path, 'a', 'ignored')
    ds = Dataset(tmp_path, 16, num_frames = 2, captions = False)
    assert not ds.has_captions
    assert isinstance(ds[0], torch.Tensor)


def test_dataloader_collates_captions_as_a_list_of_strings(tmp_path):
    write_clip(tmp_path, 'a', 'first')
    write_clip(tmp_path, 'b', 'second')

    loader = data.DataLoader(Dataset(tmp_path, 16, num_frames = 2), batch_size = 2)
    videos, captions = next(iter(loader))

    assert videos.shape == (2, 3, 2, 16, 16)
    # torch's default collate zips the batch, so the caption field arrives as a tuple of str.
    # is_list_str accepts either, which is what VideoDiffusion checks before embedding.
    assert isinstance(captions, (list, tuple)) and all(isinstance(c, str) for c in captions)
    assert is_list_str(captions)


class RecordingModel(torch.nn.Module):
    """Stands in for VideoDiffusion so the trainer can be tested without downloading BERT."""

    def __init__(self, image_size = 16, num_frames = 2, channels = 3):
        super().__init__()
        self.image_size, self.num_frames, self.channels = image_size, num_frames, channels
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.trained_on = []
        self.sampled_with = []

    def forward(self, videos, cond = None, **kwargs):
        self.trained_on.append(cond)
        return videos.mean() + self.weight

    def sample(self, batch_size = 1, cond = None, progress = True):
        self.sampled_with.append(cond)
        return torch.rand(batch_size, self.channels, self.num_frames, self.image_size, self.image_size)


def test_trainer_passes_captions_as_cond(tmp_path):
    from kinema import Trainer

    write_clip(tmp_path, 'a', 'first')
    write_clip(tmp_path, 'b', 'second')

    model = RecordingModel()
    trainer = Trainer(
        model, tmp_path, device = 'cpu', train_batch_size = 2, train_num_steps = 2,
        gradient_accumulate_every = 1, results_folder = tmp_path / 'r', save_and_sample_every = 1000
    )
    trainer.train()

    assert len(model.trained_on) == 2
    assert all(is_list_str(cond) for cond in model.trained_on)


def test_trainer_passes_none_when_unconditional(tmp_path):
    from kinema import Trainer

    write_clip(tmp_path, 'a')

    model = RecordingModel()
    trainer = Trainer(
        model, tmp_path, device = 'cpu', train_batch_size = 1, train_num_steps = 1,
        gradient_accumulate_every = 1, results_folder = tmp_path / 'r', save_and_sample_every = 1000
    )
    trainer.train()

    assert model.trained_on == [None]


def test_trainer_samples_with_captions(tmp_path):
    from kinema import Trainer

    write_clip(tmp_path, 'a', 'only caption')

    model = RecordingModel()
    trainer = Trainer(
        model, tmp_path, device = 'cpu', train_batch_size = 1, train_num_steps = 2,
        gradient_accumulate_every = 1, results_folder = tmp_path / 'r',
        save_and_sample_every = 1, num_sample_rows = 2
    )
    # one clip, four sample tiles: captions cycle so the grid is still full
    assert trainer.sample_cond == ['only caption'] * 4

    trainer.train()

    # periodic samples come from the EMA copy, not the model that was stepped
    sampled = trainer.ema_model.sampled_with
    assert sampled and all(is_list_str(cond) for cond in sampled)
    assert sum(len(cond) for cond in sampled) == 4
