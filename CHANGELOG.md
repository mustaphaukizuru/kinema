# Changelog

All notable changes to Kinema are recorded here.
This project adheres to [Semantic Versioning](https://semver.org/).

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
