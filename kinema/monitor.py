"""
TensorBoard monitoring, wired through the trainer's ``log_fn`` hook.

Diffusion training loss is famously uninformative — it wanders while sample quality improves
steadily, so a dashboard that plots only the loss curve tells you very little. This logger sends
the periodic **sample clips** to TensorBoard alongside the scalars, because watching the samples
is how you actually judge a run.
"""

import importlib.util
import logging

logger = logging.getLogger(__name__)


def _require_writer():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as e:
        raise ImportError(
            "monitoring needs TensorBoard. install it with: pip install 'kinema[viz]'"
        ) from e
    return SummaryWriter


class TensorBoardLogger:
    """
    A ``log_fn`` for :meth:`kinema.Trainer.train` that writes to TensorBoard.

        from kinema.monitor import TensorBoardLogger

        with TensorBoardLogger('./results/tb') as tb:
            trainer.train(log_fn = tb)

    Then ``tensorboard --logdir ./results/tb`` and open http://localhost:6006.

    Scalars are written every step. Sample clips are written whenever the trainer produces one,
    as video if ``moviepy`` is installed and as a strip of frames otherwise — the frames carry
    the same information, so the fallback is not a degraded experience, only a different one.
    """

    def __init__(self, log_dir, fps = 8, flush_every = 50):
        self.writer = _require_writer()(str(log_dir))
        self.log_dir = str(log_dir)
        self.fps = fps
        self.flush_every = flush_every
        # add_video encodes through moviepy, which is not a kinema dependency. torch's
        # implementation prints and returns when it is missing rather than raising, so the
        # sample would be dropped without a word — detect it up front instead.
        self._video_supported = importlib.util.find_spec('moviepy') is not None
        self._calls = 0

        logger.info('logging to tensorboard at %s', self.log_dir)
        if not self._video_supported:
            logger.info('moviepy not installed, so sample clips log as frame strips rather than video')

    def __call__(self, log):
        step = log.get('step', self._calls)
        self._calls += 1

        for key, value in log.items():
            if key != 'step' and isinstance(value, (int, float)):
                self.writer.add_scalar(f'train/{key}', value, step)

        if 'sample' in log:
            self._add_sample(log['sample'], step)

        if self._calls % self.flush_every == 0:
            self.writer.flush()

    def _add_sample(self, path, step):
        """Send one sample clip, as video where possible and as frames otherwise."""
        from kinema.data import read_clip

        try:
            clip = read_clip(path)
        except (OSError, ValueError):
            logger.warning('could not read sample %s for tensorboard', path)
            return

        # read_clip gives (channels, frames, h, w); tensorboard wants (batch, frames, channels, h, w)
        video = clip.permute(1, 0, 2, 3).unsqueeze(0)

        if self._video_supported:
            self.writer.add_video('samples/video', video, step, fps = self.fps)
        else:
            self.writer.add_images('samples/frames', video[0], step)

    def close(self):
        self.writer.flush()
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
