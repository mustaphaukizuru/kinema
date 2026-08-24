"""
Measure what DDIM actually buys you.

Samples the same untrained model over the full DDPM chain and over progressively shorter
DDIM schedules, timing each one. The videos are meaningless (the model is untrained) —
the numbers are the point, and they are the numbers you will see at inference time.

    python examples/ddim_speedup.py
"""

import time

import torch

from kinema import Unet3D, VideoDiffusion

TIMESTEPS = 1000
SCHEDULES = [None, 250, 100, 50, 20, 10]   # None = the full DDPM chain


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    unet = Unet3D(dim = 64, dim_mults = (1, 2, 4))
    diffusion = VideoDiffusion(unet, image_size = 32, num_frames = 10, timesteps = TIMESTEPS).to(device)
    print(f'{sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M parameters on {device}, {TIMESTEPS} training timesteps\n')

    print(f'{"sampler":<16}{"steps":>7}{"seconds":>10}{"speedup":>10}')
    print('-' * 43)

    baseline = None

    for steps in SCHEDULES:
        # warm up so the first entry is not paying for CUDA kernel compilation
        diffusion.sample(batch_size = 1, sampling_timesteps = steps, progress = False)

        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        diffusion.sample(batch_size = 1, sampling_timesteps = steps, progress = False)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start

        baseline = baseline if baseline is not None else elapsed
        label = 'DDPM (full)' if steps is None else 'DDIM'
        print(f'{label:<16}{steps or TIMESTEPS:>7}{elapsed:>10.2f}{baseline / elapsed:>9.1f}x')

    print('\nDDIM at eta = 0 is deterministic: the same seed gives the same video every time.')


if __name__ == '__main__':
    main()
