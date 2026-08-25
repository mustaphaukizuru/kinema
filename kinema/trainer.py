"""Training loop: EMA, gradient accumulation, checkpointing, sampling and multi-GPU."""

import copy
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils import data

from kinema.data import Dataset, video_tensor_to_gif
from kinema.utils import compile_supported, cycle, default, exists, noop, num_to_groups
from kinema.version import __version__

logger = logging.getLogger(__name__)


def _compile_error_types():
    """
    The exceptions a compile backend raises, so a fallback can be narrow.

    Catching bare Exception around a training step would swallow genuine model errors. These are
    the two families a failing backend actually throws, looked up defensively because they live
    in private modules.
    """
    types = []

    try:
        from torch._dynamo.exc import TorchDynamoException
        types.append(TorchDynamoException)
    except ImportError:
        pass

    try:
        from torch._inductor.exc import InductorError
        types.append(InductorError)
    except ImportError:
        pass

    return tuple(types)


COMPILE_ERRORS = _compile_error_types()


def _build_accelerator(amp):
    try:
        from accelerate import Accelerator
    except ImportError as e:
        raise ImportError(
            "multi-GPU training needs Accelerate. install it with: pip install 'kinema[distributed]'"
        ) from e

    return Accelerator(mixed_precision = 'fp16' if amp else 'no')

class EMA:
    def __init__(self, beta):
        self.beta = beta

    @torch.no_grad()
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            ma_params.copy_(self.update_average(ma_params, current_params))

        for current_buf, ma_buf in zip(current_model.buffers(), ma_model.buffers()):
            ma_buf.copy_(current_buf)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer:
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995,
        num_frames = 16,
        train_batch_size = 32,
        train_lr = 1e-4,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        num_sample_rows = 4,
        max_grad_norm = None,
        num_workers = 0,
        device = None,
        captions = 'auto',
        compile = False,
        accelerate = False
    ):
        """
        ``accelerate = True`` hands device placement, mixed precision and gradient
        synchronisation to Accelerate, which is what makes multi-GPU work:

            accelerate launch -m kinema.cli train -c config.yaml --accelerate

        It also decides the device, so an explicit ``device`` is ignored in that mode. On a
        single GPU it behaves like the plain path, which is why the same code covers both.
        """
        super().__init__()

        self.accelerator = _build_accelerator(amp) if accelerate else None

        if exists(self.accelerator):
            if exists(device):
                logger.info('accelerate chooses the device, so device = %s is ignored', device)
            self.device = self.accelerator.device
        else:
            self.device = torch.device(default(device, lambda: next(diffusion_model.parameters()).device))

        self.model = diffusion_model.to(self.device)
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        self.ds = Dataset(folder, image_size, channels = channels, num_frames = num_frames, captions = captions)

        logger.info('found %d clips at %s', len(self.ds), folder)
        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        loader = data.DataLoader(
            self.ds,
            batch_size = train_batch_size,
            shuffle = True,
            pin_memory = self.device.type == 'cuda',
            num_workers = num_workers
        )
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr)

        if exists(self.accelerator):
            self.model, self.opt, loader = self.accelerator.prepare(self.model, self.opt, loader)
            logger.info(
                'accelerate: %d process(es), %s',
                self.accelerator.num_processes, self.accelerator.distributed_type
            )

        self.dl = cycle(loader)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(self.device.type, enabled = amp)
        self.max_grad_norm = max_grad_norm

        # captions for the periodic samples. a conditioned model cannot sample without cond,
        # and reusing captions from the data keeps successive samples comparable.
        self.sample_cond = None
        if self.ds.has_captions:
            captions = self.ds.captions
            self.sample_cond = [captions[i % len(captions)] for i in range(num_sample_rows ** 2)]

        # torch.compile wraps the model but shares its parameters, so EMA, saving and loading
        # all keep working against self.model and checkpoints stay free of _orig_mod prefixes.
        self.compiled_model = None
        if compile:
            if compile_supported():
                self.compiled_model = torch.compile(self.model)
                logger.info('torch.compile enabled')
            else:
                logger.warning('compile = True but torch.compile cannot build here; training eagerly')

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.reset_parameters()

    def unwrapped(self):
        """The bare model, with any DistributedDataParallel wrapper removed."""
        if exists(self.accelerator):
            return self.accelerator.unwrap_model(self.model)
        return self.model

    @property
    def is_main(self):
        return self.accelerator.is_main_process if exists(self.accelerator) else True

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.unwrapped().state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.unwrapped())

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.unwrapped().state_dict(),
            'ema': self.ema_model.state_dict(),
            'opt': self.opt.state_dict(),
            'scaler': self.scaler.state_dict(),
            'version': __version__
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def save_current(self):
        """
        Checkpoint wherever the run happens to be, and return the milestone written.

        Used when a run is interrupted. It reuses the milestone numbering rather than inventing
        a name, so ``load(-1)`` and ``--resume`` pick it up like any other checkpoint.
        """
        milestone = self.step // self.save_and_sample_every
        self.save(milestone)
        return milestone

    def load(self, milestone, **kwargs):
        if milestone == -1:
            all_milestones = []
            for path in Path(self.results_folder).glob('**/*.pt'):
                try:
                    all_milestones.append(int(path.stem.split('-')[-1]))
                except ValueError:
                    # a checkpoint named by something other than a number is not ours to rank
                    logger.debug('ignoring unrecognised checkpoint name %s', path.name)

            assert len(all_milestones) > 0, (
                f'no numbered checkpoints in {self.results_folder} to resume from'
            )
            milestone = max(all_milestones)

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location = self.device, weights_only = True)

        self.step = data['step']
        self.unwrapped().load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        if 'opt' in data:
            self.opt.load_state_dict(data['opt'])
        self.scaler.load_state_dict(data['scaler'])

    def train(
        self,
        prob_focus_present = 0.,
        focus_present_mask = None,
        log_fn = noop
    ):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            for _ in range(self.gradient_accumulate_every):
                batch = next(self.dl)

                # a captioned dataset yields (video, caption) pairs; an unconditional one yields tensors
                data, cond = batch if isinstance(batch, (list, tuple)) else (batch, None)
                data = data.to(self.device)

                # not default(): it calls a callable fallback, and an nn.Module is callable
                forward = self.compiled_model if exists(self.compiled_model) else self.model

                # accelerate owns mixed precision when it is driving
                amp_context = (
                    self.accelerator.autocast() if exists(self.accelerator)
                    else autocast(self.device.type, enabled = self.amp)
                )

                with amp_context:
                    kwargs = dict(
                        cond = cond,
                        prob_focus_present = prob_focus_present,
                        focus_present_mask = focus_present_mask
                    )

                    try:
                        loss = forward(data, **kwargs)
                    except COMPILE_ERRORS:
                        # the toolchain probe only proves a backend can build *something*.
                        # this model may still defeat it, and that surfaces here, on the first
                        # compiled forward. drop to eager rather than end the run.
                        if forward is self.model:
                            raise

                        logger.warning('torch.compile failed on this model; falling back to eager')
                        self.compiled_model = None
                        forward = self.model
                        loss = forward(data, **kwargs)

                # backward belongs outside the autocast region, per the torch.amp docs
                scaled = loss / self.gradient_accumulate_every

                if exists(self.accelerator):
                    self.accelerator.backward(scaled)
                else:
                    self.scaler.scale(scaled).backward()

                # per-step loss is debug detail; log_fn is the supported reporting hook
                logger.debug('%d: %.6f', self.step, loss.item())

            log = {'step': self.step, 'loss': loss.item()}

            if exists(self.max_grad_norm):
                if exists(self.accelerator):
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                else:
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            if exists(self.accelerator):
                self.opt.step()
            else:
                self.scaler.step(self.opt)
                self.scaler.update()

            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.is_main and self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                num_samples = self.num_sample_rows ** 2
                batches = num_to_groups(num_samples, self.batch_size)

                # progress bars would flood the training log, so they stay off here
                all_videos_list = []
                taken = 0
                for n in batches:
                    cond = self.sample_cond[taken:taken + n] if exists(self.sample_cond) else None
                    all_videos_list.append(self.ema_model.sample(batch_size = n, cond = cond, progress = False))
                    taken += n

                all_videos_list = torch.cat(all_videos_list, dim = 0)

                all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))

                one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i = self.num_sample_rows)
                video_path = str(self.results_folder / str(f'{milestone}.gif'))
                video_tensor_to_gif(one_gif, video_path)
                log = {**log, 'sample': video_path}
                self.save(milestone)

            log_fn(log)
            self.step += 1

        if exists(self.accelerator):
            self.accelerator.wait_for_everyone()

        logger.info('training completed')
