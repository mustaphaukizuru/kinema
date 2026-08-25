"""Command line interface: ``kinema train`` and ``kinema sample``.

Training a model should not require writing a script. A YAML config describes the model,
the diffusion process and the training loop; every value can be overridden from the command
line with ``--set key.path=value``.
"""

import os

# The Intel Fortran runtime that ships inside PyTorch's Windows wheels installs its own console
# handler, and it aborts the process on Ctrl+C before Python ever raises KeyboardInterrupt:
#
#     forrtl: error (200): program aborting due to control-C event
#
# That loses every step since the last checkpoint. Disabling the handler hands Ctrl+C back to
# Python, which is what makes the graceful save below possible. It must be set before torch is
# imported, so it comes first, ahead of every other import.
os.environ.setdefault('FOR_DISABLE_CONSOLE_CTRL_HANDLER', '1')

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from kinema.data import video_tensor_to_gif, video_tensor_to_mp4  # noqa: E402
from kinema.diffusion import VideoDiffusion  # noqa: E402
from kinema.trainer import Trainer  # noqa: E402
from kinema.unet import Unet3D  # noqa: E402
from kinema.version import __version__  # noqa: E402

logger = logging.getLogger('kinema')

DEFAULTS = {
    'data': {'folder': './data'},
    'model': {'dim': 64, 'dim_mults': [1, 2, 4, 8]},
    'diffusion': {'image_size': 64, 'num_frames': 10, 'timesteps': 1000},
    'trainer': {
        'train_batch_size': 4,
        'train_lr': 1e-4,
        'train_num_steps': 100000,
        'save_and_sample_every': 1000,
        'results_folder': './results',
    },
}


def _require_yaml():
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "reading a config file needs PyYAML. install it with: pip install 'kinema[cli]'"
        ) from e
    return yaml


def _merge(base, override):
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(value):
    """
    Recover numbers YAML declines to parse.

    YAML 1.1 only reads a float in exponent form when it carries a decimal point, so a
    perfectly reasonable ``--set trainer.train_lr=3e-4`` arrives as the string '3e-4' and
    blows up inside the optimizer. Numbers on the command line are meant as numbers.
    """
    if not isinstance(value, str):
        return value

    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue

    return value


def _apply_override(config, assignment):
    """Apply one ``section.key=value`` assignment, typed by the YAML scalar rules."""
    if '=' not in assignment:
        raise SystemExit(f"--set expects key.path=value, got '{assignment}'")

    path, raw = assignment.split('=', 1)
    keys = path.split('.')

    try:
        value = _coerce(_require_yaml().safe_load(raw))
    except ImportError:
        value = _coerce(raw)

    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def load_config(path = None, overrides = ()):
    """Build a config from the defaults, an optional YAML file and any --set overrides."""
    config = _merge(DEFAULTS, {})

    if path is not None:
        loaded = _require_yaml().safe_load(Path(path).read_text(encoding = 'utf-8')) or {}
        config = _merge(config, loaded)

    for assignment in overrides:
        _apply_override(config, assignment)

    return config


def build(config, device):
    """Construct the diffusion model described by a config, on the given device."""
    unet = Unet3D(**config['model'])
    diffusion = VideoDiffusion(unet, **config['diffusion']).to(device)

    params = sum(p.numel() for p in unet.parameters())
    logger.info('model: %.1fM parameters on %s', params / 1e6, device)

    return diffusion


def resolve_device(name = None):
    """Pick the best available device unless one was named explicitly."""
    if name is not None:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def cmd_train(args):
    config = load_config(args.config, args.set)
    device = resolve_device(args.device)
    diffusion = build(config, device)

    # explicit flags win over the config file
    for flag in ('compile', 'accelerate'):
        if getattr(args, flag):
            config['trainer'][flag] = True

    # accelerate picks its own device, so do not force one on it
    device_arg = None if config['trainer'].get('accelerate') else device

    trainer = Trainer(diffusion, config['data']['folder'], device = device_arg, **config['trainer'])

    if args.resume is not None:
        trainer.load(args.resume)
        logger.info('resumed from milestone %s at step %d', args.resume, trainer.step)

    def report(log):
        if trainer.step % args.log_every == 0:
            message = f"step {trainer.step}  loss {log['loss']:.4f}"
            if 'sample' in log:
                message += f"  sample {log['sample']}"
            print(message, flush = True)

    board = None
    if args.tensorboard:
        from kinema.monitor import TensorBoardLogger

        log_dir = Path(config['trainer']['results_folder']) / 'tb' if args.tensorboard is True else args.tensorboard
        board = TensorBoardLogger(log_dir)
        print(f'tensorboard --logdir {log_dir}', flush = True)

    def log_fn(log):
        report(log)
        if board is not None:
            board(log)

    try:
        trainer.train(log_fn = log_fn)
    except KeyboardInterrupt:
        # Ctrl+C during a long run should not throw away the work since the last checkpoint
        milestone = trainer.save_current()

        print(flush = True)
        print(f'interrupted at step {trainer.step}', flush = True)
        print(f'saved milestone {milestone} to {trainer.results_folder}', flush = True)
        print('resume this run with:  --resume', flush = True)

        return 130   # the conventional exit code for SIGINT
    finally:
        if board is not None:
            board.close()

    return 0


def cmd_sample(args):
    config = load_config(args.config, args.set)
    device = resolve_device(args.device)
    diffusion = build(config, device)

    checkpoint = torch.load(args.checkpoint, map_location = device, weights_only = True)
    diffusion.load_state_dict(checkpoint['ema' if args.ema else 'model'])
    diffusion.eval()
    logger.info('loaded %s (step %d)', args.checkpoint, checkpoint.get('step', -1))

    videos = diffusion.sample(
        cond = list(args.text) if args.text else None,
        cond_scale = args.cond_scale,
        batch_size = args.num,
        sampling_timesteps = args.steps,
        eta = args.eta,
        progress = not args.quiet
    )

    out = Path(args.out)
    if out.parent != Path(''):
        out.parent.mkdir(parents = True, exist_ok = True)

    writer = video_tensor_to_mp4 if out.suffix.lower() == '.mp4' else video_tensor_to_gif

    for i, video in enumerate(videos.cpu()):  # noqa: B007 - i is used below
        path = out if len(videos) == 1 else out.with_name(f'{out.stem}-{i}{out.suffix}')
        writer(video, str(path))
        print(f'wrote {path}', flush = True)

    return 0


def main(argv = None):
    parser = argparse.ArgumentParser(prog = 'kinema', description = 'Text-to-video diffusion in PyTorch.')
    parser.add_argument('--version', action = 'version', version = f'kinema {__version__}')
    sub = parser.add_subparsers(dest = 'command', required = True)

    def common(p):
        p.add_argument('-c', '--config', help = 'path to a YAML config file')
        p.add_argument('--set', action = 'append', default = [], metavar = 'KEY=VALUE',
                       help = 'override a config value, e.g. --set trainer.train_lr=3e-4')
        p.add_argument('--device', help = 'cuda, cpu or mps (autodetected by default)')
        p.add_argument('-v', '--verbose', action = 'store_true', help = 'log per-step detail')

    train = sub.add_parser('train', help = 'train a model from a config')
    common(train)
    train.add_argument('--resume', type = int, nargs = '?', const = -1, metavar = 'MILESTONE',
                       help = 'resume from a checkpoint milestone; -1 or a bare flag means the latest')
    train.add_argument('--log-every', type = int, default = 10, help = 'print a line every N steps')
    train.add_argument('--tensorboard', nargs = '?', const = True, metavar = 'LOGDIR',
                       help = 'log scalars and sample clips to TensorBoard (default: <results_folder>/tb)')
    train.add_argument('--compile', action = 'store_true',
                       help = 'compile the model with torch.compile when the toolchain allows it')
    train.add_argument('--accelerate', action = 'store_true',
                       help = 'train through Accelerate; required for multi-GPU via `accelerate launch`')
    train.set_defaults(func = cmd_train)

    sample = sub.add_parser('sample', help = 'generate videos from a checkpoint')
    common(sample)
    sample.add_argument('checkpoint', help = 'path to a .pt checkpoint written by the trainer')
    sample.add_argument('-o', '--out', default = 'sample.gif', help = 'output path; .mp4 or .gif')
    sample.add_argument('-n', '--num', type = int, default = 1, help = 'how many videos to generate')
    sample.add_argument('--steps', type = int, help = 'DDIM sampling steps; fewer is faster')
    sample.add_argument('--eta', type = float, help = 'DDIM noise level; 0 is deterministic')
    sample.add_argument('--text', nargs = '+', help = 'text prompts, for conditioned models')
    sample.add_argument('--cond-scale', type = float, default = 1., help = 'classifier-free guidance strength')
    sample.add_argument('--ema', action = 'store_true', help = 'sample from the EMA weights')
    sample.add_argument('-q', '--quiet', action = 'store_true', help = 'hide the sampling progress bar')
    sample.set_defaults(func = cmd_sample)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level = logging.DEBUG if args.verbose else logging.INFO,
        format = '%(message)s',
        stream = sys.stderr
    )

    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
