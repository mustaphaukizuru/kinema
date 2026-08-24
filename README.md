<div align="center">

![machine imagined fireworks](./fireworks.webp)

# Kinema

**Text-to-video diffusion in PyTorch — a space-time factored 3D U-Net you can actually train.**

*Complexity, simplified.*

[![CI](https://github.com/mustaphaukizuru/kinema/actions/workflows/ci.yml/badge.svg)](https://github.com/mustaphaukizuru/kinema/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%E2%80%93%203.13-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A5%202.0-ee4c2c)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

---

*Kinema* — from the Greek **κίνημα**, *movement*: the root that gave us *cinema*.

Kinema is a clean, tested implementation of [Video Diffusion Models](https://arxiv.org/abs/2204.03458)
(Ho et al., 2022), which extends denoising diffusion from 2D images to 3D video using a
**space-time factored U-Net**. Text conditioning, classifier-free guidance and a batteries-included
trainer are all here — and all covered by tests that run on CPU and GPU.

<div align="center">
<img src="./3d-unet.png" width="520px" alt="space-time factored 3D U-Net">
</div>

## Why this exists

Research code for video diffusion is excellent, but it tends to assume a CUDA box, a specific
PyTorch version, and that you will never resume a run. Kinema keeps the architecture faithful to the
paper and rebuilds everything around it: it runs on CPU, CUDA or MPS; it checkpoints optimizer state
so training resumes correctly; text conditioning works against current `transformers`; and every
claim here is enforced by CI on Python 3.9 through 3.13.

<div align="center">
<img src="./samples/moving-mnist.gif" width="260px" alt="moving mnist samples">
<br><em>Moving MNIST, ~14k steps</em>
</div>

## Install

```bash
pip install kinema            # core model + trainer
pip install "kinema[text]"    # + BERT text conditioning
```

Requires Python >= 3.9 and PyTorch >= 2.0. Runs on CPU, CUDA and Apple Silicon (MPS).

## Quickstart

Videos are float tensors shaped `(batch, channels, frames, height, width)` with values in `[0, 1]` —
Kinema handles normalisation internally, so you do not have to.

```python
import torch
from kinema import Unet3D, VideoDiffusion

model = Unet3D(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

diffusion = VideoDiffusion(
    model,
    image_size = 32,
    num_frames = 5,
    timesteps = 1000,
    loss_type = 'l1'     # 'l1' or 'l2'
)

videos = torch.rand(1, 3, 5, 32, 32)
loss = diffusion(videos)
loss.backward()

# after a lot of training
sampled = diffusion.sample(batch_size = 4)
sampled.shape  # (4, 3, 5, 32, 32)
```

## Conditioning on text

Pass your own embeddings by declaring their width with `cond_dim`:

```python
model = Unet3D(dim = 64, cond_dim = 64, dim_mults = (1, 2, 4, 8))
diffusion = VideoDiffusion(model, image_size = 32, num_frames = 5)

videos = torch.rand(2, 3, 5, 32, 32)
text   = torch.randn(2, 64)          # your embeddings

loss = diffusion(videos, cond = text)
loss.backward()

sampled = diffusion.sample(cond = text)
```

Or hand it raw strings and let Kinema embed them with BERT-base — requires the `[text]` extra.
`bert-base-cased` is fetched from the Hugging Face hub on first use:

```python
model = Unet3D(
    dim = 64,
    use_bert_text_cond = True,       # adopts the BERT dimensions automatically
    dim_mults = (1, 2, 4, 8)
)

diffusion = VideoDiffusion(model, image_size = 32, num_frames = 5)

videos = torch.rand(3, 3, 5, 32, 32)
text = [
    'a whale breaching from afar',
    'young girl blowing out candles on her birthday cake',
    'fireworks with blue and green sparkles'
]

loss = diffusion(videos, cond = text)
loss.backward()

# cond_scale > 1 strengthens classifier-free guidance
sampled = diffusion.sample(cond = text, cond_scale = 2.)
```

## Training

`Trainer` handles a folder of GIFs, EMA, gradient accumulation, mixed precision, periodic sampling
and checkpointing. It runs wherever your model lives — no CUDA assumption.

```python
import torch
from kinema import Unet3D, VideoDiffusion, Trainer

model = Unet3D(dim = 64, dim_mults = (1, 2, 4, 8))

diffusion = VideoDiffusion(
    model,
    image_size = 64,
    num_frames = 10,
    timesteps = 1000,
    loss_type = 'l1'
).cuda()

trainer = Trainer(
    diffusion,
    './data',                       # folder of .gif files
    train_batch_size = 32,
    train_lr = 1e-4,
    train_num_steps = 700000,
    gradient_accumulate_every = 2,
    ema_decay = 0.995,
    save_and_sample_every = 1000,
    amp = True,                     # mixed precision
    num_workers = 4
)

trainer.train()
trainer.load(-1)                    # resume from the newest checkpoint
```

Samples and weights land in `./results`. Checkpoints carry model, EMA, **optimizer** and scaler
state, so a resumed run picks up exactly where it stopped.

Kinema logs through the standard `logging` module rather than printing, so it composes with your own
setup:

```python
import logging
logging.basicConfig(level = logging.INFO)
```

### Choosing a device

```python
trainer = Trainer(diffusion, './data', device = 'cpu')   # or 'cuda', 'mps'
```

Left unset, the trainer follows whatever device the model is already on.

## Co-training images and video

The paper argues that factored space-time attention lets you train on images and video together by
forcing the network to attend to the present moment. Pass `prob_focus_present` to arrest attention
across time for a fraction of each batch:

```python
loss = diffusion(videos, cond = text, prob_focus_present = 0.5)
loss.backward()
```

## Interpolating between videos

Noise two clips to step `t`, mix them, and denoise the result — conditioning is respected:

```python
blend = diffusion.interpolate(video_a, video_b, t = 500, lam = 0.5)
blend = diffusion.interpolate(video_a, video_b, cond = text, cond_scale = 2.)
```

## Performance

Measured on a laptop RTX 3050 (6 GB) at `dim = 64`, `dim_mults = (1, 2, 4, 8)`, 32x32 across 5
frames — 35.7 M parameters:

| Operation | Cost |
|---|---|
| Training step, batch 4 | ~0.93 s |
| Peak VRAM, batch 4 | ~1.5 GB |
| Sampling, 4 videos at 1000 steps | ~3.5 min |

Sampling is the bottleneck, as it is for any ancestral DDPM sampler — a faster sampler is on the
roadmap.

## Development

```bash
git clone https://github.com/mustaphaukizuru/kinema && cd kinema
pip install -e ".[dev,text]"

ruff check .    # lint
pytest          # 22 tests — CUDA tests run automatically when a GPU is present
```

The suite promotes PyTorch deprecation warnings and `ResourceWarning` to errors, so breakage from a
new PyTorch release surfaces in CI rather than three hours into a training run.

## Roadmap

- [ ] DDIM / fewer-step sampler — the single biggest usability win
- [ ] MP4 and frame-folder datasets, not GIF only
- [ ] `scripts/train.py` CLI driven by a YAML config
- [ ] Conditional synthesis from `{video_filename}.txt` sidecar captions
- [ ] Project text into 4-8 tokens used as memory keys and values in attention
- [ ] Multi-GPU training via `accelerate`
- [ ] Token shifts along time and space

## Project lineage

Kinema builds on [lucidrains/video-diffusion-pytorch](https://github.com/lucidrains/video-diffusion-pytorch)
by [Phil Wang](https://github.com/lucidrains), released under MIT — the original PyTorch
implementation of the paper, and the source of this architecture. Kinema keeps that architecture and
rebuilds everything around it: packaging, device handling, text conditioning, checkpointing, tests
and CI. See [CHANGELOG.md](CHANGELOG.md) for the full record.

The moving-MNIST experiments in the original work were made possible by compute from
[Stability.ai](https://stability.ai/).

## Citations

```bibtex
@misc{ho2022video,
  title         = {Video Diffusion Models},
  author        = {Jonathan Ho and Tim Salimans and Alexey Gritsenko and William Chan and Mohammad Norouzi and David J. Fleet},
  year          = {2022},
  eprint        = {2204.03458},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

```bibtex
@misc{saharia2022imagen,
  title  = {Imagen: unprecedented photorealism, deep level of language understanding},
  author = {Chitwan Saharia and William Chan and Saurabh Saxena and Lala Li and Jay Whang and Emily Denton and Seyed Kamyar Seyed Ghasemipour and Burcu Karagol Ayan and S. Sara Mahdavi and Rapha Gontijo Lopes and Tim Salimans and Jonathan Ho and David Fleet and Mohammad Norouzi},
  year   = {2022}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Maintainer

**Mustapha Ukizuru** — IT Manager & Full-Stack Developer · CS Educator · Tech Consultant

[mustaphaukizuru.com](https://mustaphaukizuru.com) ·
[LinkedIn](https://linkedin.com/in/mustaphaukizuru) ·
[GitHub](https://github.com/mustaphaukizuru) ·
[hello@mustaphaukizuru.com](mailto:hello@mustaphaukizuru.com)

*Build it. Simplify it. Scale it.*
