"""Reading and writing video — GIF, MP4 and frame folders — plus the training Dataset."""

import re
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

# extensions understood by the Dataset. gif is read with pillow; the rest need PyAV.

GIF_EXTS = ('gif',)
VIDEO_EXTS = ('mp4', 'webm', 'avi', 'mov', 'mkv', 'm4v')
IMAGE_EXTS = ('png', 'jpg', 'jpeg', 'bmp', 'webp')

def _require_av(path):
    try:
        import av
    except ImportError as e:
        raise ImportError(
            f"reading or writing '{path}' needs PyAV. install it with: pip install 'kinema[video]'"
        ) from e
    return av

def natural_key(path):
    """Sort frame files the way a human reads them, so frame2 precedes frame10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', str(path))]

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

# mp4 / webm / mov -> (channels, frames, height, width) tensor

def video_to_tensor(path, channels = 3, transform = None):
    """Decode any PyAV-readable container (mp4, webm, mov, ...) into a video tensor in [0, 1]."""
    av = _require_av(path)
    transform = default(transform, T.ToTensor)
    mode = CHANNELS_TO_MODE[channels]

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f'no video stream in {path}')
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        tensors = [transform(frame.to_image().convert(mode)) for frame in container.decode(stream)]

    if len(tensors) == 0:
        raise ValueError(f'decoded zero frames from {path}')

    return torch.stack(tensors, dim = 1)

# tensor of shape (channels, frames, height, width) -> mp4

def video_tensor_to_mp4(tensor, path, fps = 8, codec = 'libx264', crf = 18):
    """Write a video tensor to an MP4. Unlike GIF this keeps full colour and stays small."""
    av = _require_av(path)
    images = [T.ToPILImage()(frame).convert('RGB') for frame in tensor.unbind(dim = 1)]

    width, height = images[0].size
    # h.264 requires even dimensions; crop a pixel rather than rescaling the content
    width, height = width - width % 2, height - height % 2
    if width == 0 or height == 0:
        raise ValueError(f'video too small to encode: {images[0].size}')

    with av.open(str(path), mode = 'w') as container:
        stream = container.add_stream(codec, rate = fps)
        stream.width, stream.height = width, height
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': str(crf)}

        for img in images:
            if img.size != (width, height):
                img = img.crop((0, 0, width, height))
            for packet in stream.encode(av.VideoFrame.from_image(img)):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

    return path

# folder of numbered image frames -> (channels, frames, height, width) tensor

def frames_to_tensor(folder, channels = 3, transform = None, exts = IMAGE_EXTS):
    """Read a directory of numbered image frames as one clip, ordered naturally."""
    transform = default(transform, T.ToTensor)
    mode = CHANNELS_TO_MODE[channels]

    paths = sorted((p for ext in exts for p in Path(folder).glob(f'*.{ext}')), key = natural_key)
    if len(paths) == 0:
        raise ValueError(f'no image frames ({", ".join(exts)}) in {folder}')

    tensors = []
    for path in paths:
        with Image.open(path) as img:
            tensors.append(transform(img.convert(mode)))

    return torch.stack(tensors, dim = 1)

def read_clip(path, channels = 3, transform = None):
    """Read one clip from a GIF, a video container or a folder of frames, dispatching on the path."""
    path = Path(path)

    if path.is_dir():
        return frames_to_tensor(path, channels = channels, transform = transform)

    ext = path.suffix.lower().lstrip('.')

    if ext in GIF_EXTS:
        return gif_to_tensor(path, channels = channels, transform = transform)

    if ext in VIDEO_EXTS:
        return video_to_tensor(path, channels = channels, transform = transform)

    raise ValueError(f'unsupported clip format: {path}')

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
        exts = GIF_EXTS + VIDEO_EXTS
    ):
        """
        A folder of clips. Each clip may be a GIF, a video container (mp4, webm, mov, ...) or a
        subdirectory of numbered image frames.

        Video files are searched first. If the folder holds none, every immediate subdirectory
        containing images is treated as one clip instead, so frame-per-file datasets work
        without reprocessing.
        """
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = sorted(
            (p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')),
            key = natural_key
        )

        if len(self.paths) == 0:
            self.paths = sorted(
                (d for d in Path(f'{folder}').iterdir()
                 if d.is_dir() and any(d.glob(f'*.{ext}') for ext in IMAGE_EXTS)),
                key = natural_key
            )

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
        tensor = read_clip(path, channels = self.channels, transform = self.transform)
        return self.cast_num_frames_fn(tensor)
