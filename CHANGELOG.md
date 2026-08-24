# Changelog

## 0.8.0 (2026-08-24)

### Fixed
- Text conditioning crashed on modern PyTorch: `torch.hub.load('huggingface/pytorch-transformers')` prompts for
  repo trust (EOFError in non-interactive runs) and the hub repo is archived. Now uses `transformers.AutoModel`.
- `GaussianDiffusion.interpolate()` ignored `cond` and asserted on conditioned models; also did not normalise /
  unnormalise inputs consistently with `forward()` / `sample()`.
- `Trainer` hard-coded `.cuda()`; now device-agnostic (`device=` argument, defaults to the model's device).
- `Trainer.save()` did not checkpoint the optimizer, so resumed runs reset Adam state.
- `Trainer.load()` now uses `map_location` and `weights_only=True`.
- `gif_to_tensor()` leaked file handles (`ResourceWarning: unclosed file`).
- `video_tensor_to_gif()` returned an exhausted iterator.
- EMA now also syncs buffers, not only parameters.

### Changed
- Deprecated `torch.cuda.amp` replaced with `torch.amp`.
- `print` replaced with the `logging` module (`video_diffusion_pytorch` logger).
- `transformers`, `sentencepiece`, `sacremoses` are now an optional extra: `pip install video-diffusion-pytorch[text]`.
- Packaging moved from `setup.py` to `pyproject.toml`; `requires-python >= 3.9`, `torch >= 2.0`.
- Removed unused `Unet3D(block_type=...)` argument and dead `q_mean_variance`.
- Added `__version__`, `__all__`, `Dataset` / gif helpers exported from the package root.

### Added
- `tests/` pytest suite (CPU + CUDA), CI workflow with ruff + pytest matrix, PyPI trusted publishing.
