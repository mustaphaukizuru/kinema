# Contributing to Kinema

Contributions are welcome — bug reports, fixes, documentation and features alike.

## Getting set up

```bash
git clone https://github.com/mustaphaukizuru/kinema && cd kinema
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev,text]"
```

PyTorch is not pinned to a build here. If you want CUDA, install the matching wheel first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Before you open a pull request

```bash
ruff check .     # must pass
pytest           # must pass
```

CUDA tests are skipped automatically when no GPU is present, so a CPU-only machine is fine for
most work. CI runs the same two commands on Python 3.9 and 3.12.

## House rules

- **Tests come with behaviour changes.** Any fix should carry a test that fails without it.
- **No new warnings.** The suite treats PyTorch deprecation warnings and `ResourceWarning` as
  errors. If you need a new PyTorch API, use the current one.
- **Keep modules focused.** `unet` builds the network, `diffusion` runs the process, `trainer`
  runs the loop, `data` reads and writes video, `text` handles captions, `utils` holds shared
  helpers. New code belongs in whichever of those it actually serves.
- **Match the existing style.** Spaces around keyword-argument equals signs (`dim = 64`) are the
  convention throughout this codebase; `ruff` is configured to allow it.
- **Public API changes go through `kinema/__init__.py`** and get a `CHANGELOG.md` entry. If you
  rename something public, leave an alias behind.

## Reporting a bug

Please include your Python version, PyTorch version, operating system, whether you are on CPU or
GPU, and the shortest snippet that reproduces the problem.

## Licence

Contributions are accepted under the [MIT Licence](LICENSE) that covers this project.
