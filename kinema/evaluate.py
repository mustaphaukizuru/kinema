"""
Comparing checkpoints.

Training loss is a poor guide to a diffusion model's progress, and not because it is noisy in the
usual sense — it is noisy *by construction*. Every step draws a fresh timestep and fresh noise, so
consecutive losses measure different problems. A model can improve while the number rises.

Fixing the timesteps, the noise and the clips removes that variance entirely. What is left is a
number that means the same thing for every checkpoint, so two of them can actually be compared.
"""

import logging

import torch

from kinema.utils import exists, normalize_img

logger = logging.getLogger(__name__)


def fixed_problems(dataset, num_problems, num_timesteps, seed, batch_size = 1):
    """
    Build the fixed evaluation set: which clips, at which timesteps, with which noise.

    Everything is drawn from a seeded CPU generator, so the same dataset and seed give the same
    problems on any machine and any device.
    """
    generator = torch.Generator().manual_seed(seed)

    indices = torch.randint(0, len(dataset), (num_problems, batch_size), generator = generator)
    timesteps = torch.randint(0, num_timesteps, (num_problems, batch_size), generator = generator)
    noise_seeds = torch.randint(0, 2 ** 31 - 1, (num_problems,), generator = generator)

    return indices, timesteps, noise_seeds


@torch.inference_mode()
def deterministic_loss(
    diffusion,
    dataset,
    num_problems = 16,
    batch_size = 1,
    seed = 0,
    device = None,
    progress = False
):
    """
    Average the diffusion loss over a fixed set of (clip, timestep, noise) problems.

    Unlike training loss this is comparable across checkpoints: lower is genuinely better, on the
    same problems every time. It needs no reference model and no extra download.
    """
    device = torch.device(device) if exists(device) else next(diffusion.parameters()).device
    diffusion.eval()

    indices, timesteps, noise_seeds = fixed_problems(
        dataset, num_problems, diffusion.num_timesteps, seed, batch_size
    )

    total = 0.
    steps = range(num_problems)

    if progress:
        from tqdm import tqdm
        steps = tqdm(steps, desc = 'evaluating')

    for problem in steps:
        clips, captions = [], []

        for index in indices[problem].tolist():
            item = dataset[index]
            clip, caption = item if isinstance(item, tuple) else (item, None)
            clips.append(clip)
            captions.append(caption)

        videos = normalize_img(torch.stack(clips).to(device))
        cond = captions if all(exists(c) for c in captions) else None

        # the same noise for this problem on every checkpoint, whatever the device
        noise = torch.randn(
            videos.shape,
            generator = torch.Generator().manual_seed(int(noise_seeds[problem])),
        ).to(device)

        loss = diffusion.p_losses(
            videos,
            timesteps[problem].to(device),
            cond = cond,
            noise = noise,
            null_cond_prob = 0.       # no dropout: evaluation must not be random
        )

        total += loss.item()

    return total / num_problems


def compare(checkpoints, build_model, dataset, **kwargs):
    """
    Score several checkpoints on identical problems.

    ``build_model`` is called once per checkpoint and must return a model with those weights
    loaded. Returns ``[(name, loss), ...]`` in the order given.
    """
    results = []

    for checkpoint in checkpoints:
        model = build_model(checkpoint)
        loss = deterministic_loss(model, dataset, **kwargs)
        results.append((str(checkpoint), loss))
        logger.info('%s: %.6f', checkpoint, loss)

    return results
