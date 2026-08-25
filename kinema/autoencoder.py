"""
A frame-wise convolutional autoencoder, for latent diffusion.

Diffusion at 256×256 is expensive because every reverse step runs the U-Net at full resolution.
Latent diffusion moves the process into a compressed space: an autoencoder shrinks each frame
first, the U-Net denoises there, and the result is decoded back. A 4× spatial compression makes
each step roughly 16× cheaper, which is what puts higher resolutions within reach of one GPU.

The autoencoder here works frame by frame, deliberately. Temporal structure is the U-Net's job —
it has attention across time and this does not — so compressing space alone keeps the two concerns
separate and lets a model trained on images be reused on video.
"""

import torch
from einops import rearrange
from torch import nn


class ResBlock(nn.Module):
    """A 2D residual block, applied per frame."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, padding = 1),
            nn.GroupNorm(min(8, dim_out), dim_out),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, 3, padding = 1),
            nn.GroupNorm(min(8, dim_out), dim_out),
        )
        self.residual = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.block(x) + self.residual(x))


class FrameAutoencoder(nn.Module):
    """
    Compresses each frame by ``2 ** levels`` in height and width, leaving the time axis alone.

        autoencoder = FrameAutoencoder(latent_channels = 4, levels = 2)   # 64x64 -> 16x16

        latents = autoencoder.encode(videos)     # (b, latent_channels, f, h/4, w/4)
        videos  = autoencoder.decode(latents)

    Train it on its own data first with :meth:`reconstruction_loss`; latent diffusion then treats
    it as fixed. Training both at once makes the diffusion target move underneath the model.
    """

    def __init__(self, channels = 3, latent_channels = 4, dim = 64, levels = 2):
        super().__init__()
        assert levels >= 1, 'levels must be at least 1'

        self.channels = channels
        self.latent_channels = latent_channels
        self.levels = levels
        self.downsample_factor = 2 ** levels

        dims = [dim * (2 ** i) for i in range(levels)]

        encoder = [nn.Conv2d(channels, dims[0], 3, padding = 1)]
        for i, width in enumerate(dims):
            encoder += [ResBlock(width, width), nn.Conv2d(width, dims[min(i + 1, levels - 1)], 4, 2, 1)]
        encoder += [ResBlock(dims[-1], dims[-1]), nn.Conv2d(dims[-1], latent_channels, 1)]
        self.encoder = nn.Sequential(*encoder)

        reversed_dims = list(reversed(dims))

        decoder = [nn.Conv2d(latent_channels, reversed_dims[0], 1), ResBlock(reversed_dims[0], reversed_dims[0])]
        for i, width in enumerate(reversed_dims):
            following = reversed_dims[min(i + 1, levels - 1)]
            decoder += [nn.ConvTranspose2d(width, following, 4, 2, 1), ResBlock(following, following)]

        decoder += [
            nn.GroupNorm(min(8, dims[0]), dims[0]),
            nn.SiLU(),
            nn.Conv2d(dims[0], channels, 3, padding = 1),
        ]
        self.decoder = nn.Sequential(*decoder)

    def encode(self, video):
        """``(b, c, f, h, w)`` in [-1, 1] to ``(b, latent_channels, f, h/s, w/s)``."""
        b = video.shape[0]
        frames = rearrange(video, 'b c f h w -> (b f) c h w')
        return rearrange(self.encoder(frames), '(b f) c h w -> b c f h w', b = b)

    def decode(self, latents):
        """The inverse of :meth:`encode`."""
        b = latents.shape[0]
        frames = rearrange(latents, 'b c f h w -> (b f) c h w')
        return rearrange(self.decoder(frames), '(b f) c h w -> b c f h w', b = b)

    def forward(self, video):
        return self.decode(self.encode(video))

    def reconstruction_loss(self, video):
        """L1 between a clip and its round trip. Train the autoencoder on this alone."""
        return nn.functional.l1_loss(self.forward(video), video)

    def latent_shape(self, image_size):
        """The spatial size of the latents for a given frame size."""
        assert image_size % self.downsample_factor == 0, (
            f'image_size {image_size} must be divisible by {self.downsample_factor}'
        )
        return image_size // self.downsample_factor

    @torch.inference_mode()
    def measure_scale(self, video):
        """
        The standard deviation of the latents for a batch.

        Diffusion assumes roughly unit-variance inputs. Latents rarely oblige, so this is
        measured once on real data and passed to :class:`~kinema.latent.LatentDiffusion` as
        ``latent_scale``, which normalises them.
        """
        return self.encode(video).std().item()
