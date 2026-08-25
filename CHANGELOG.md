# Changelog

All notable changes to Kinema are recorded here.
This project adheres to [Semantic Versioning](https://semver.org/).

## 0.18.0 — 2026-08-25

### Added
- **v-parameterisation.** `VideoDiffusion(..., objective = 'v')` trains the network to predict
  velocity rather than noise, after
  [Progressive Distillation](https://arxiv.org/abs/2202.00512). It is better conditioned at the
  noisy end of the chain and tends to hold up better at low step counts, which pairs naturally
  with DDIM.

  The architecture is unchanged — only the training target differs — so the two are
  interchangeable in shape but **not in meaning**: a checkpoint trained on one objective must be
  sampled with the same one.

- `VideoDiffusion.model_predictions()`, which runs the network and returns `(pred_noise, x_start)`
  whatever it was trained to predict. Both samplers go through it, so a new objective needs no
  changes to either. `predict_v()` and `predict_start_from_v()` are public alongside it.

### Changed
- `p_mean_variance()` and the DDIM loop each had their own copy of the "predict, clip, recompute
  the noise" sequence. They now share one.

### Verified
ruff clean, 203 tests passing on CPU and CUDA. Because this refactor touched the path every sampler
takes, a test asserts the default `noise` objective produces **bit-identical** output to the inline
sequence it replaced — this is an addition, not a silent change in behaviour.

## 0.17.0 — 2026-08-25

You can now tell which checkpoint is better. Until this release there was no way to, which is a
strange gap for a trainable model.

### Added
- **`kinema eval`** and the `kinema.evaluate` module.

  ```
  $ kinema eval results/model-*.pt -c configs/moving-mnist.yaml --ema

  checkpoint                        step          loss
  ----------------------------------------------------
  model-1.pt                         300      0.191026
  model-2.pt                         600      0.133042  <- best
  ```

  Training loss cannot answer this. Every step draws a fresh timestep and fresh noise, so
  consecutive losses measure *different problems* — the number wanders while the model improves,
  which is what it did in the runs behind the 0.12.0 benchmark. Fixing the clips, the timesteps
  and the noise removes that variance entirely, leaving a figure that means the same thing for
  every checkpoint.

  It needs no reference model, no I3D download and no held-out labels — only the dataset already
  on disk. The noise is drawn from seeded CPU generators, so a score is reproducible across
  machines and devices, and unaffected by whatever the ambient RNG is doing.

- `deterministic_loss()`, `fixed_problems()` and `compare()` in `kinema.evaluate`, usable directly.

### Verified
ruff clean, 184 tests passing on CPU and CUDA, and the command driven against two real checkpoints
from one run: it ranks step 600 above step 300 and returns byte-identical scores across invocations.
A test also trains a model briefly and asserts its score improves, so the metric is known to move
in the right direction rather than merely being stable.

## 0.16.0 — 2026-08-25

Everything here is about a long run surviving contact with reality: reproducing it, not filling the
disk, not losing a checkpoint, and finding out about a typo before the GPU warms up.

### Added
- **`--seed`** on both `kinema train` and `kinema sample`, and `seed_everything()` in
  `kinema.utils`. DDIM at `eta = 0` is deterministic, so a seed plus a checkpoint gives
  byte-identical video — which is what makes a result reproducible rather than merely repeatable.
- **`Trainer(keep_last_n = N)`** deletes all but the newest N checkpoints. At 170 MB each, written
  every N steps, a long run would otherwise fill a disk. Unnumbered `.pt` files are left alone.
- **`Trainer(amp_dtype = 'bfloat16')`.** bfloat16 carries float32's exponent range, so the
  gradient scaler is switched off automatically; Accelerate is told `bf16` to match.
- **Config validation.** Every key is checked against the constructor it feeds, before any model
  is built:

      $ kinema train -c config.yaml --set trainer.train_lrr=3e-4
      config error:
        unknown key 'trainer.train_lrr'; did you mean ['train_batch_size', 'train_lr', 'train_num_steps']?

  Previously a typo travelled into `Trainer.__init__` and surfaced as a bare `TypeError`.
- `Trainer.milestones()` and `Trainer.prune_checkpoints()`, both public.

### Fixed
- **`save_current()` could overwrite a checkpoint.** Interrupting at step 450 with
  `save_and_sample_every = 250` wrote over `model-1.pt` from step 250. It now advances to a free
  milestone, so an interrupted run adds to the record rather than replacing a model that may well
  have been the better one.
- Two `Trainer` arguments were documented in the `Unet3D` table in the README.

### Verified
ruff clean, 172 tests passing on CPU and CUDA. Two new guards keep the release metadata honest: one
asserts `CITATION.cff` matches `kinema.__version__`, the other that the CHANGELOG has an entry for
it. The second caught this very release mid-edit.

## 0.15.0 — 2026-08-25

A real bug in classifier-free guidance, inherited from the upstream implementation, plus DDIM for
interpolation.

### Fixed
- **Classifier-free guidance never trained its unconditional branch.** `p_losses` called the U-Net
  without `null_cond_prob`, so it defaulted to `0`, the dropout mask was always empty, and
  `null_cond_emb` kept the random initialisation it was born with. At sampling time
  `forward_with_cond_scale` asks that same embedding for the unconditional prediction — so
  `cond_scale` was extrapolating away from noise rather than from anything the model had learned.

  Training now drops the caption with probability `cond_drop_prob` (0.1 by default), which is what
  the README already claimed was happening. An explicit `null_cond_prob` from the caller still
  wins, and unconditional models are untouched.

  It adds no parameters, so existing checkpoints load either way — but a model trained before this
  release has an untrained null embedding and wants retraining before `cond_scale` means anything.

### Added
- **`interpolate()` takes `sampling_timesteps` and `eta`**, exactly as `sample()` does. It walked
  every step from `t` down to zero regardless of the DDIM settings on the model.
- `VideoDiffusion.ddim_loop()`, the strided reverse walk that `ddim_sample()` and `interpolate()`
  now share rather than each keeping their own copy.

### Verified
ruff clean, 150 tests passing on CPU and CUDA. The guidance fix is pinned by tests asserting the
null embedding receives gradient at the default drop rate and does not at zero — so the bug cannot
come back silently.

## 0.14.2 — 2026-08-25

Stopping a run is now as reliable as starting one.

### Fixed
- **Ctrl+C threw away every step since the last checkpoint.** Two separate problems, both fixed:

  1. Nothing caught `KeyboardInterrupt`, so an interrupted run died with a traceback and no save.
     It now checkpoints where it stands, prints the milestone and how to continue, and exits 130 —
     the conventional code for SIGINT.
  2. On Windows it never got that far. The Intel Fortran runtime inside PyTorch's wheels installs
     its own console handler and aborts the process before Python raises anything:

         forrtl: error (200): program aborting due to control-C event

     `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1` is now set at the top of `kinema.cli`, ahead of the torch
     import, which hands Ctrl+C back to Python.

- **`load(-1)` crashed on any `.pt` whose name did not end in a number.** One stray file — a
  `model-best.pt` — made `--resume` raise `ValueError` instead of finding the newest checkpoint.
  Unrecognised names are skipped, and an empty result gives a message naming the folder.

### Added
- `Trainer.save_current()`, which checkpoints at the current step and returns the milestone.
- A **Running it** section in the README: starting, stopping, resuming, the browser dashboard, and
  generating a video — with the exact commands, for the terminal and the GUI both.

### Verified
ruff clean, 136 tests passing on CPU and CUDA. The interrupt path is covered end to end: a run is
interrupted, the checkpoint it writes is reloaded, and training continues from it.

**Note:** the graceful stop is verified by raising `KeyboardInterrupt` inside a real run. Delivering
a true `CTRL_C_EVENT` to a child process cannot be scripted reliably on Windows without risking the
parent shell, so the handler is tested rather than the keystroke. Press Ctrl+C in a terminal once to
confirm it on your own machine — and use Ctrl+C, not Ctrl+Break, which kills the process outright.

## 0.14.1 — 2026-08-25

Two CI failures, both real. Found by the Ubuntu / CPU / Python 3.9 job, which is a different
machine from the one the previous releases were developed on.

### Fixed
- **`torch.compile` could still kill a run.** 0.13.0 probed the toolchain at startup so a missing
  Triton or C++ compiler would not surface mid-training — but the probe compiles a trivial module,
  which proves only that a backend can build *something*. On CI the toolchain is present and
  Inductor still fails on this U-Net's dynamic shapes:

      InductorError: TypeError: cannot determine truth value of Relational: 16*s0*s92 < 16

  The first compiled forward pass now catches that and drops to eager for the rest of the run.
  The fallback is narrow — only `TorchDynamoException` and `InductorError` — so genuine bugs in a
  model still raise rather than being silently swallowed, and there is a test for each direction.

- **Two CLI tests required PyAV.** They wrote MP4 fixtures without the `importorskip` guard that
  the data tests carry, so they failed on Python 3.9, where PyAV publishes no wheels and 0.13.0
  had correctly excluded it. The fixtures are GIFs now: those tests exercise the CLI, not the
  container format, and Python 3.9 keeps its CLI coverage.

### Verified
ruff clean, 130 tests passing on CPU and CUDA, and the whole suite re-run with PyAV blocked to
reproduce the Python 3.9 environment locally rather than guessing at it.

## 0.14.0 — 2026-08-24

The two remaining architecture items from the roadmap. Both are off by default, so nothing changes
for an existing model unless you ask for it.

### Added
- **Text as attention memory.** `Unet3D(..., num_cond_tokens = 4)` projects the conditioning
  vector into that many key/value tokens, prepended to the sequence in every attention layer.
  Previously the caption only ever modulated blocks through the timestep embedding — nothing
  attended to it.

  ```python
  unet = Unet3D(dim = 64, dim_mults = (1, 2, 4, 8), use_bert_text_cond = True, num_cond_tokens = 4)
  ```

  The memory tokens take no positional bias, they remain visible to queries arrested by
  `prob_focus_present`, and classifier-free guidance nulls the caption on both routes at once, so
  `cond_scale` keeps its meaning.

- **Token shift along time and space.** `Unet3D(..., token_shift = 'time' | 'space-time')` gives
  each resnet block cheap local mixing: half the channels are split between the shift directions
  and displaced one step, the rest pass through. After
  [CogVideo](https://arxiv.org/abs/2205.15868).

  **It adds no parameters.** A token-shifted model loads a checkpoint from a plain one and the
  reverse, so it can be switched on mid-project without retraining.

- `shift()` and `token_shift()` in `kinema.modules`, and `token_shift` on `ResnetBlock`.

### Verified
ruff clean, 128 tests passing on CPU and CUDA. A checkpoint written by 0.11.0 was loaded on this
code and sampled from, both into a plain model and into a token-shifted one.

### Roadmap
Every item published in the README roadmap is now done.

## 0.13.0 — 2026-08-24

Scale out and speed up: multi-GPU training, and `torch.compile` that degrades instead of exploding.

### Added
- **Multi-GPU via Accelerate.** `Trainer(..., accelerate = True)` or `kinema train --accelerate`:

  ```bash
  accelerate launch -m kinema.cli train -c config.yaml --accelerate
  ```

  Accelerate takes device placement, mixed precision and gradient synchronisation. Checkpointing
  and periodic sampling run on the main process only, and checkpoints are saved unwrapped, so a
  run trained across GPUs loads into a single-device trainer and back. New `[distributed]` extra.

- `Trainer.unwrapped()` and `Trainer.is_main`, for reaching past a DistributedDataParallel wrapper.

- **`torch.compile` support.** `Trainer(..., compile = True)` or `kinema train --compile`.
  Compilation is lazy — `torch.compile` returns happily and only fails on the first forward pass,
  deep inside a training loop — and it needs a toolchain the user may not have (Triton for CUDA, a
  C++ compiler for CPU). Rather than catch a wide family of backend errors mid-run, `compile_supported()`
  builds one trivial module up front and caches the answer; an unavailable toolchain logs a warning
  and trains eagerly. The compiled module shares parameters with the original, so EMA, saving and
  loading are untouched and checkpoints carry no `_orig_mod.` prefixes.

### Changed
- `backward()` moved outside the `autocast` region, which is what the `torch.amp` documentation
  asks for. It was previously called inside it.

### Verified
ruff clean, 99 tests passing on CPU and CUDA, and `accelerate launch -m kinema.cli train` driven
end to end.

**Not verified here:** multi-process training. The Windows PyTorch wheels are built without libuv,
so `torch.distributed`'s `TCPStore` cannot start and both `torchrun` and `accelerate launch` fail
before reaching kinema. The single-process Accelerate path is covered by tests and a real run; the
multi-process path is implemented but untested on this machine, and wants a Linux or WSL box with
more than one GPU to confirm.

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
