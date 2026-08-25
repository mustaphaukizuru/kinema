"""
Latent diffusion: run the whole process in a compressed space.

:class:`LatentDiffusion` wraps a :class:`~kinema.diffusion.VideoDiffusion` and an autoencoder, and
presents the same surface as the pixel-space model — ``forward`` returns a loss, ``sample`` returns
video in ``[0, 1]``, and ``image_size`` / ``channels`` / ``num_frames`` describe *pixels*. That last
part is what lets it drop into the existing :class:`~kinema.trainer.Trainer` untouched: the dataset
still loads full-resolution clips, and the compression happens inside.

The autoencoder is frozen by default. Training it alongside the diffusion model moves the target
the diffusion model is chasing, which is why latent diffusion is normally done in two stages: train
the autoencoder, then freeze it and train diffusion on its latents.
"""

import logging

import torch
from torch import nn

from kinema.utils import normalize_img, unnormalize_img

logger = logging.getLogger(__name__)


class LatentDiffusion(nn.Module):
    """
        autoencoder = FrameAutoencoder(latent_channels = 4, levels = 2)
        # ... train the autoencoder, then ...

        unet = Unet3D(dim = 64, dim_mults = (1, 2, 4), channels = 4, out_dim = 4)
        inner = VideoDiffusion(unet, image_size = 16, num_frames = 10, channels = 4)

        model = LatentDiffusion(inner, autoencoder, image_size = 64)
        Trainer(model, './data').train()

    ``image_size`` is the pixel resolution; the inner diffusion's ``image_size`` must be that
    divided by the autoencoder's downsample factor, and its ``channels`` must match the latent
    channels. Both are checked on construction, because getting them wrong otherwise surfaces as
    a shape error several layers down.
    """

    def __init__(
        self,
        diffusion,
        autoencoder,
        image_size,
        latent_scale = 1.,
        freeze_autoencoder = True
    ):
        super().__init__()

        expected = autoencoder.latent_shape(image_size)
        assert diffusion.image_size == expected, (
            f'inner diffusion has image_size {diffusion.image_size}, but a {image_size}px frame '
            f'compresses to {expected}px through this autoencoder'
        )
        assert diffusion.channels == autoencoder.latent_channels, (
            f'inner diffusion has channels {diffusion.channels}, but the autoencoder produces '
            f'{autoencoder.latent_channels} latent channels'
        )
        assert latent_scale > 0, 'latent_scale must be positive'

        self.diffusion = diffusion
        self.autoencoder = autoencoder
        self.latent_scale = latent_scale

        # the surface Trainer and the samplers read, in pixels
        self.image_size = image_size
        self.channels = autoencoder.channels
        self.num_frames = diffusion.num_frames

        self.freeze_autoencoder = freeze_autoencoder
        if freeze_autoencoder:
            self.autoencoder.eval()
            for parameter in self.autoencoder.parameters():
                parameter.requires_grad_(False)
            logger.info('autoencoder frozen; only the diffusion model will train')

    def encode(self, videos):
        """Pixels in ``[0, 1]`` to scaled latents."""
        latents = self.autoencoder.encode(normalize_img(videos))
        return latents / self.latent_scale

    def decode(self, latents):
        """Scaled latents back to pixels in ``[0, 1]``."""
        videos = self.autoencoder.decode(latents * self.latent_scale)
        return unnormalize_img(videos).clamp(0., 1.)

    def forward(self, videos, *args, **kwargs):
        """The diffusion loss, measured on latents rather than pixels."""
        context = torch.no_grad() if self.freeze_autoencoder else torch.enable_grad()

        with context:
            latents = self.encode(videos)

        # the inner model normalises again internally, so hand it something in [0, 1]
        return self.diffusion(unnormalize_img(latents), *args, **kwargs)

    @torch.inference_mode()
    def sample(self, *args, **kwargs):
        """Generate video. Denoising happens in latent space; the result is decoded."""
        latents = self.diffusion.sample(*args, **kwargs)
        return self.decode(normalize_img(latents))

    @torch.inference_mode()
    def interpolate(self, x1, x2, *args, **kwargs):
        """Blend two clips through the latent space."""
        blended = self.diffusion.interpolate(
            unnormalize_img(self.encode(x1)),
            unnormalize_img(self.encode(x2)),
            *args, **kwargs
        )
        return self.decode(normalize_img(blended))

    def p_losses(self, x_start, t, *args, **kwargs):
        """
        The loss for already-normalised pixels, so :mod:`kinema.evaluate` can score a latent
        model like any other.

        It cannot simply delegate: callers hand this normalised *pixels*, and the inner model
        works in latents. Encoding here is what makes the two agree.
        """
        latents = self.autoencoder.encode(x_start) / self.latent_scale
        return self.diffusion.p_losses(latents, t, *args, **kwargs)

    @property
    def num_timesteps(self):
        return self.diffusion.num_timesteps


def fit_latent_scale(autoencoder, videos):
    """
    Choose ``latent_scale`` from real data.

    Diffusion assumes inputs of roughly unit variance. Latents rarely have it, and a mismatch
    quietly costs sample quality rather than raising anything, so measure it once on a batch and
    pass the result to :class:`LatentDiffusion`.
    """
    scale = autoencoder.measure_scale(normalize_img(videos))
    logger.info('measured latent scale %.4f', scale)
    return scale
