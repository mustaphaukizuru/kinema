"""Kinema — text-to-video diffusion in PyTorch.

A space-time factored 3D U-Net trained as a denoising diffusion probabilistic
model, after Ho et al., *Video Diffusion Models* (https://arxiv.org/abs/2204.03458).

    from kinema import Unet3D, VideoDiffusion

    model = Unet3D(dim = 64, dim_mults = (1, 2, 4, 8))
    diffusion = VideoDiffusion(model, image_size = 32, num_frames = 5)

    loss = diffusion(videos)          # videos: (batch, channels, frames, h, w)
    loss.backward()

    videos = diffusion.sample(batch_size = 4)
"""

from kinema.data import Dataset, gif_to_tensor, video_tensor_to_gif
from kinema.diffusion import GaussianDiffusion, VideoDiffusion
from kinema.trainer import Trainer
from kinema.unet import Unet3D
from kinema.version import __version__

__all__ = [
    'Unet3D',
    'VideoDiffusion',
    'GaussianDiffusion',
    'Trainer',
    'Dataset',
    'gif_to_tensor',
    'video_tensor_to_gif',
    '__version__',
]
