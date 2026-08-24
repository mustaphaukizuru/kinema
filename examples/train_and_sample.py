"""
End-to-end: build a model, train it on a real clip, then sample from it.

This is the smallest complete thing Kinema does. It runs in about a minute on a laptop
GPU and a few minutes on CPU, and it exercises every part of the public API.

    python examples/train_and_sample.py

Fifty steps on one clip is nowhere near enough to produce a recognisable video — the
point is that the loss falls and the pipeline runs end to end. Real training means
100k steps on thousands of clips, which is what ``kinema train`` is for.
"""

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from kinema import Unet3D, VideoDiffusion, gif_to_tensor, video_tensor_to_gif

ROOT = Path(__file__).resolve().parent.parent

IMAGE_SIZE = 32
NUM_FRAMES = 10
TRAIN_STEPS = 50
SAMPLING_STEPS = 50


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    unet = Unet3D(dim = 64, dim_mults = (1, 2, 4))
    diffusion = VideoDiffusion(
        unet,
        image_size = IMAGE_SIZE,
        num_frames = NUM_FRAMES,
        timesteps = 1000,
        sampling_timesteps = SAMPLING_STEPS   # sample with DDIM
    ).to(device)

    params = sum(p.numel() for p in unet.parameters())
    print(f'model: {params / 1e6:.1f}M parameters on {device}')

    # one real clip, resized to the model's resolution
    video = gif_to_tensor(ROOT / 'samples' / 'moving-mnist.gif', channels = 3)
    video = F.interpolate(video, size = (IMAGE_SIZE, IMAGE_SIZE), mode = 'bilinear', align_corners = False)
    video = video[:, :NUM_FRAMES].unsqueeze(0).to(device)
    print(f'data:  {tuple(video.shape)}  (batch, channels, frames, height, width)')

    opt = torch.optim.Adam(diffusion.parameters(), lr = 1e-4)
    start = time.time()

    for step in range(1, TRAIN_STEPS + 1):
        loss = diffusion(video)
        loss.backward()
        opt.step()
        opt.zero_grad()

        if step % 10 == 0:
            print(f'  step {step:3d}   loss {loss.item():.4f}   {(time.time() - start) / step * 1000:.0f} ms/step')

    print(f'\nsampling {SAMPLING_STEPS} DDIM steps...')
    start = time.time()
    videos = diffusion.sample(batch_size = 1, progress = False)
    print(f'sampled {tuple(videos.shape)} in {time.time() - start:.1f}s')

    out = ROOT / 'samples' / 'example-output.gif'
    video_tensor_to_gif(videos[0].cpu(), str(out))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
