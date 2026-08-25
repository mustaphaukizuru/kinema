# Changelog

All notable changes to Kinema are recorded here.
This project adheres to [Semantic Versioning](https://semver.org/).

## 0.12.0 — 2026-08-24

You can now watch a run instead of reading numbers off a terminal.

### Added
- **TensorBoard monitoring.** `kinema train --tensorboard`, or `TensorBoardLogger` as a `log_fn`:

  ```python
  from kinema.monitor import TensorBoardLogger

  with TensorBoardLogger('./results/tb') as tb:
      trainer.train(log_fn = tb)
  ```

  Scalars go in every step; the periodic **sample clips** go in too, because diffusion loss is a
  poor progress signal and the samples are not. From a recorded run, loss rose between steps 2000
  and 4000 while reconstruction error more than halved:

  | Steps | Training loss | Reconstruction error |
  |---:|---:|---:|
  | 2000 | 0.0421 | 0.1846 |
  | 3000 | 0.0507 | 0.1637 |
  | 4000 | 0.0613 | 0.0497 |

  Clips log as video when `moviepy` is present and as frame strips otherwise. Needs the new
  `[viz]` extra.

- `Trainer`'s `log_fn` dict now carries `step`, so a logger no longer needs a closure over the
  trainer to know where it is.

### Fixed
- **Sample clips were silently dropped when `moviepy` was missing.** `torch`'s `add_video` prints
  a message and returns instead of raising, so an `except ImportError` fallback never fired and
  the clip vanished with no error — the event file held scalars and nothing else. Availability is
  now detected up front, and a regression test asserts the clip reaches the event file by one
  route or the other.

### Verified
ruff clean, 80 tests passing on CPU and CUDA, and a live TensorBoard server confirmed serving both
`train/loss` and `samples/frames` over HTTP from a real training run.

## 0.11.0 — 2026-08-24

Text-conditioned training now needs a folder of videos and a folder of text files, and nothing else.

### Added
- **Sidecar captions.** `clip.txt` beside `clip.mp4` is picked up automatically; a frame folder
  takes either a sibling `clip.txt` or an inner `caption.txt`. Captioned datasets yield
  `(video, caption)` pairs, and `Trainer` passes them straight through as `cond`, so the whole
  text-to-video path is a directory layout rather than a pipeline you assemble yourself.

  ```
  data/
    fireworks.mp4
    fireworks.txt     ->  "fireworks over a harbour at night"
  ```

  ```python
  unet = Unet3D(dim = 64, dim_mults = (1, 2, 4, 8), use_bert_text_cond = True)
  trainer = Trainer(VideoDiffusion(unet, image_size = 64, num_frames = 10), './data')
  trainer.train()
  ```

- `Dataset(..., captions = ...)` and `Trainer(..., captions = ...)`:

  | Value | Behaviour |
  |---|---|
  | `'auto'` (default) | Use captions when every clip has one; otherwise train unconditionally and warn |
  | `True` | Require them — a missing sidecar raises |
  | `False` | Ignore sidecars entirely |

  A partially captioned folder trains unconditionally rather than silently conditioning on blanks.

- `caption_for(clip_path)` and `Dataset.has_captions`, both public.

### Changed
- The trainer's periodic samples are drawn from captions in the dataset when the model is
  conditioned. Previously `sample()` was called with no `cond`, which raises on a conditional
  model — periodic sampling simply could not work for text-conditioned training. Captions cycle
  when there are fewer clips than sample tiles, so the grid stays full.

### Verified
ruff clean, 71 tests passing on CPU and CUDA, and a conditional run trained end to end against real
`bert-base-cased` embeddings from sidecar captions, writing a conditioned sample.

## 0.10.0 — 2026-08-24

Sampling is roughly 19× faster, datasets are no longer GIF-only, and training no longer requires
writing a script.

### Added
- **DDIM sampling.** `VideoDiffusion` accepts `sampling_timesteps` and `ddim_sampling_eta`, and
  `sample()` takes both per call. Setting `sampling_timesteps` below `timesteps` denoises over a
  strided subsequence of the same chain — no retraining, the same weights.

  ```python
  diffusion = VideoDiffusion(model, image_size = 32, num_frames = 10, sampling_timesteps = 50)
  videos = diffusion.sample(batch_size = 4)
  ```

  Measured on a laptop RTX 3050, one 32×32 × 10-frame clip from a 10.6 M-parameter model:

  | Sampler | Steps | Time | Speedup |
  |---|---:|---:|---:|
  | DDPM (full chain) | 1000 | 163.3 s | 1.0× |
  | DDIM | 100 | 14.5 s | 11.3× |
  | DDIM | 50 | 8.4 s | 19.5× |
  | DDIM | 10 | 1.6 s | 103.2× |

  At the default `eta = 0` DDIM is deterministic: the same seed yields the same video.

- **MP4, WebM, MOV and frame-folder datasets.** `Dataset` reads GIFs, any PyAV-decodable container,
  or subdirectories of numbered image frames, dispatching on what it finds. New public functions:
  `read_clip`, `video_to_tensor`, `frames_to_tensor` and `video_tensor_to_mp4`. Container formats
  need the new `[video]` extra; frame folders need nothing.
- **A `kinema` command.** `kinema train -c config.yaml` and `kinema sample checkpoint.pt` replace
  hand-written training scripts. Configs are YAML, every key maps onto a constructor argument, and
  any value can be overridden with `--set section.key=value`. Needs the new `[cli]` extra.
  An example config lives in `configs/moving-mnist.yaml`.
- **`examples/`** — `train_and_sample.py` runs the whole pipeline end to end; `ddim_speedup.py`
  reproduces the benchmark table above.
- `predict_noise_from_start()` and `clip_x_start()` on `VideoDiffusion`, the pieces DDIM needs,
  exposed because they are useful on their own.

### Changed
- **`sample()` and `interpolate()` accept `progress = False`.** The tqdm bar was unconditional and
  wrote to stderr, so every script, notebook and CI job that sampled got hundreds of lines of bar.
  The trainer's periodic samples are now silent.
- Per-step training loss moved from `logging.info` to `logging.debug`. At INFO it drowned out
  anything a caller chose to report through `log_fn`.
- `Dataset`'s default `exts` now covers video containers as well as GIFs.

### Fixed
- **`--set trainer.train_lr=3e-4` crashed inside the optimizer.** YAML 1.1 only reads exponent-form
  floats that carry a decimal point, so `3e-4` arrived as a string. Command-line numbers are now
  coerced. Inside a YAML file standard YAML rules still apply — write `1.0e-4`.

### Verified
ruff clean, 57 tests passing on CPU and CUDA, wheel and sdist build, `kinema train` and
`kinema sample` driven end to end against a real checkpoint.

## 0.9.0 — 2026-08-24

The project is renamed **Kinema** and reorganised into a proper package.

### Changed
- **Renamed** from `video-diffusion-pytorch` to `kinema`; the import path is now `kinema`.
- **`GaussianDiffusion` is now `VideoDiffusion`.** The old name remains as an alias, so existing
  code keeps working:
  ```python
  from kinema import VideoDiffusion          # preferred
  from kinema import GaussianDiffusion       # alias, still supported
  ```
- **Split the 1,000-line monolith** into focused modules:

  | Module | Responsibility |
  |---|---|
  | `kinema.unet` | `Unet3D`, the space-time factored denoiser |
  | `kinema.diffusion` | `VideoDiffusion`, the forward and reverse processes |
  | `kinema.modules` | Attention, resnet blocks, norms, positional bias |
  | `kinema.trainer` | `Trainer` and EMA |
  | `kinema.data` | `Dataset` and GIF read/write |
  | `kinema.text` | BERT tokenisation and embedding |
  | `kinema.utils` | Shared helpers |

- Package metadata, documentation and project URLs now reflect the maintainer.
- Version is single-sourced from `kinema/version.py`.

## 0.8.0 — 2026-08-24

Correctness, portability and packaging pass. Everything below was verified on CPU and CUDA.

### Fixed
- **Text conditioning crashed on modern PyTorch.** `torch.hub.load('huggingface/pytorch-transformers', ...)`
  prompts interactively for repo trust, raising `EOFError` in any non-interactive run, and that hub
  repository is archived. Now loads `bert-base-cased` through `transformers.AutoModel`.
- **`interpolate()` ignored `cond`** and raised an assertion on conditioned models. It now accepts
  `cond` and `cond_scale`, and normalises inputs consistently with `forward()` and `sample()`.
- **`Trainer` hard-coded `.cuda()`**, so it could not train on CPU or MPS. It is now device-agnostic
  with an explicit `device=` argument.
- **`Trainer.save()` did not checkpoint the optimizer**, silently resetting Adam moments on resume.
- **`Trainer.load()`** now passes `map_location` and `weights_only=True`.
- **`gif_to_tensor()` leaked a file handle per item**, exhausting descriptors on large datasets.
- **`video_tensor_to_gif()` returned an already-exhausted iterator.**
- **EMA only tracked parameters**, never buffers.

### Changed
- Deprecated `torch.cuda.amp` replaced with `torch.amp`.
- `print()` replaced with the `logging` module.
- `transformers`, `sentencepiece` and `sacremoses` demoted to the optional `[text]` extra, cutting
  the default install substantially.
- Packaging moved from `setup.py` to `pyproject.toml`; `requires-python >= 3.9`, `torch >= 2.0`.
- Removed the unused `Unet3D(block_type=...)` argument and the dead `q_mean_variance` method.

### Added
- A pytest suite of 22 tests covering shapes, both loss types, classifier-free guidance, dynamic
  thresholding, focus-present masking, conditioned interpolation, GIF round-trips, dataset frame
  casting and the full train/save/load cycle. BERT is mocked, so the suite needs no network.
- CI running ruff and pytest across Python 3.9 and 3.12.
- PyPI trusted publishing in the release workflow.
- `__version__` and `__all__` on the package root.

### Notes
The test suite promotes PyTorch deprecation warnings and `ResourceWarning` to errors, so the classes
of bug fixed above surface in CI rather than mid-training.

## Earlier history

Versions up to 0.7.0 were released as `video-diffusion-pytorch` by Phil Wang. See the
[upstream repository](https://github.com/lucidrains/video-diffusion-pytorch) for that history.
