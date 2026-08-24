"""Reading and writing video as animated GIFs, plus the training Dataset."""

from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

from kinema.utils import default, identity

CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# tensor of shape (channels, frames, height, width) -> gif

def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    images = list(map(T.ToPILImage(), tensor.unbind(dim = 1)))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# gif -> (channels, frame, height, width) tensor

def gif_to_tensor(path, channels = 3, transform = None):
    transform = default(transform, T.ToTensor)
    with Image.open(path) as img:
        tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)

def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))

class Dataset(data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels = 3,
        num_frames = 16,
        horizontal_flip = False,
        force_num_frames = True,
        exts = ('gif',)
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        tensor = gif_to_tensor(path, self.channels, transform = self.transform)
        return self.cast_num_frames_fn(tensor)
