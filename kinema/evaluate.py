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

        # Seed inside a forked RNG so the noise is identical for this problem on every
        # checkpoint, while the caller's global RNG is left exactly as it was. The model
        # draws its own noise, which matters for a latent model: it diffuses at the latent
        # resolution, not the pixel one, so noise made out here would be the wrong shape.
        devices = [device] if device.type == 'cuda' else []

        with torch.random.fork_rng(devices = devices):
            torch.manual_seed(int(noise_seeds[problem]))
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(int(noise_seeds[problem]))

            loss = diffusion.p_losses(
                videos,
                timesteps[problem].to(device),
                cond = cond,
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


# ------------------------------------------------------------- distribution distance


def matrix_sqrt_trace(cov_a, cov_b):
    """
    ``Tr(sqrt(A @ B))`` for two symmetric positive semi-definite matrices.

    The usual route is ``scipy.linalg.sqrtm``, which pulls in SciPy and returns complex values
    that then have to be discarded. Going through the eigendecomposition of
    ``sqrt(A) B sqrt(A)`` — which is symmetric PSD, so its eigenvalues are real and non-negative —
    gives the same number with nothing but torch.
    """
    values, vectors = torch.linalg.eigh(cov_a)
    sqrt_a = vectors @ torch.diag(values.clamp(min = 0).sqrt()) @ vectors.T

    middle = sqrt_a @ cov_b @ sqrt_a
    return torch.linalg.eigvalsh(middle).clamp(min = 0).sqrt().sum()


def frechet_distance(features_a, features_b):
    """
    The Fréchet distance between two sets of feature vectors, each ``(n, d)``.

    Lower means the two distributions look more alike to the feature extractor. This is the
    quantity behind FID and FVD; which extractor produced the features decides what it measures.
    """
    features_a, features_b = features_a.double(), features_b.double()

    assert features_a.shape[0] > 1 and features_b.shape[0] > 1, (
        'a covariance needs at least two samples per set'
    )

    mean_a, mean_b = features_a.mean(0), features_b.mean(0)
    cov_a = torch.cov(features_a.T)
    cov_b = torch.cov(features_b.T)

    mean_term = (mean_a - mean_b).pow(2).sum()
    trace_term = cov_a.trace() + cov_b.trace() - 2 * matrix_sqrt_trace(cov_a, cov_b)

    return (mean_term + trace_term).clamp(min = 0).item()


def r3d_extractor(device = None, weights = 'DEFAULT'):
    """
    A Kinetics-pretrained R(2+1)D network from torchvision, as a feature extractor.

    Canonical FVD uses I3D, whose weights are not distributed through pip. This is the closest
    thing available without a manual download, so scores from it are **comparable to each other
    but not to published FVD numbers** — treat it as a relative measure between your own runs.

    The weights download on first use.
    """
    from torchvision.models.video import r3d_18

    model = r3d_18(weights = weights)
    model.fc = torch.nn.Identity()      # keep the 512-d pooled features
    model.eval()

    if exists(device):
        model = model.to(device)

    return model


@torch.inference_mode()
def video_features(videos, extractor, batch_size = 8):
    """
    Feature vectors for a batch of clips in ``[0, 1]``, shaped ``(b, c, f, h, w)``.

    Normalisation follows the Kinetics preprocessing the pretrained weights expect.
    """
    device = next(extractor.parameters()).device

    mean = torch.tensor([0.43216, 0.394666, 0.37645], device = device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989], device = device).view(1, 3, 1, 1, 1)

    features = []

    for start in range(0, videos.shape[0], batch_size):
        batch = videos[start:start + batch_size].to(device)
        features.append(extractor((batch - mean) / std).flatten(1).cpu())

    return torch.cat(features)


def frechet_video_distance(real, generated, extractor = None, batch_size = 8):
    """
    Compare two sets of clips, both in ``[0, 1]``.

    Needs at least two clips per side. With the default extractor the number is relative — use it
    to rank your own checkpoints, not to compare against published figures.
    """
    extractor = extractor if exists(extractor) else r3d_extractor()

    return frechet_distance(
        video_features(real, extractor, batch_size),
        video_features(generated, extractor, batch_size),
    )
