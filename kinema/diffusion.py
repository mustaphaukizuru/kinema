"""The Gaussian diffusion process wrapping the denoising U-Net."""

import torch
import torch.nn.functional as F
from einops import rearrange
from einops_exts import check_shape
from torch import nn
from tqdm import tqdm

from kinema.utils import (
    default,
    embed_text,
    exists,
    is_list_str,
    normalize_img,
    unnormalize_img,
)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

class VideoDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        text_use_bert_cls = False,
        channels = 3,
        timesteps = 1000,
        sampling_timesteps = None,
        ddim_sampling_eta = 0.,
        cond_drop_prob = 0.1,
        objective = 'noise',
        loss_type = 'l1',
        use_dynamic_thres = False, # from the Imagen paper
        dynamic_thres_percentile = 0.9
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        def register_buffer(name, val):
            self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # Classifier-free guidance only works if the *unconditional* branch is trained. During
        # training the caption is dropped this often, which is what teaches `null_cond_emb` to
        # mean "no caption". Left at 0, that embedding keeps its random initialisation and
        # `cond_scale` extrapolates away from noise rather than from a learned prior.
        self.cond_drop_prob = cond_drop_prob

        # What the network is asked to predict. 'noise' is the original DDPM target and the
        # default. 'v' is the velocity parameterisation from https://arxiv.org/abs/2202.00512,
        # which is better conditioned at the noisy end of the chain and behaves better with few
        # sampling steps. Both use the same architecture — only the training target differs.
        assert objective in ('noise', 'v'), f"objective must be 'noise' or 'v', got {objective!r}"
        self.objective = objective

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

        # sampling. when sampling_timesteps < timesteps, sample() takes the DDIM
        # path: the same trained model, denoised over a strided subsequence of
        # the chain. eta = 0 is deterministic DDIM, eta = 1 recovers DDPM.

        self.sampling_timesteps = default(sampling_timesteps, self.num_timesteps)
        assert 1 <= self.sampling_timesteps <= self.num_timesteps, (
            f'sampling_timesteps must be in [1, {self.num_timesteps}], got {self.sampling_timesteps}'
        )
        self.ddim_sampling_eta = ddim_sampling_eta

    @property
    def has_cond(self):
        """Whether the denoiser can use conditioning at all."""
        # default True for a denoiser that does not declare itself, preserving old behaviour
        return getattr(self.denoise_fn, 'has_cond', True)

    def resolve_cond(self, cond):
        """
        Normalise a conditioning argument, embedding text only when it can actually be used.

        A captioned dataset paired with an unconditional model used to send its captions all the
        way into ``embed_text``, loading BERT to produce an embedding nothing would read — and
        failing outright on an install without the ``text`` extra. Ignore it instead.
        """
        if not self.has_cond:
            return None

        if is_list_str(cond):
            return embed_text(cond, self.betas.device, return_cls_repr = self.text_use_bert_cls)

        return cond

    def clip_x_start(self, x_recon):
        """Clamp the predicted x_0 — statically to [-1, 1], or by the Imagen dynamic-thresholding rule."""
        s = 1.
        if self.use_dynamic_thres:
            s = torch.quantile(
                rearrange(x_recon, 'b ... -> b (...)').abs(),
                self.dynamic_thres_percentile,
                dim = -1
            )

            s.clamp_(min = 1.)
            s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

        # clip by threshold, depending on whether static or dynamic
        return x_recon.clamp(-s, s) / s

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_v(self, x_start, t, noise):
        """The velocity target: what the network learns to predict when ``objective = 'v'``."""
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        """Recover x_0 from a velocity prediction."""
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def model_predictions(self, x, t, cond = None, cond_scale = 1., clip_denoised = True):
        """
        Run the network and return ``(pred_noise, x_start)``, whatever it was trained to predict.

        Both samplers need the pair, and both objectives can supply it — keeping the conversion
        in one place is what lets 'noise' and 'v' share every sampler.
        """
        output = self.denoise_fn.forward_with_cond_scale(x, t, cond = cond, cond_scale = cond_scale)

        if self.objective == 'v':
            x_start = self.predict_start_from_v(x, t, output)
        else:
            x_start = self.predict_start_from_noise(x, t = t, noise = output)

        if clip_denoised:
            x_start = self.clip_x_start(x_start)

        # after clipping, the noise must be recomputed to stay consistent with x_start
        pred_noise = self.predict_noise_from_start(x, t = t, x_start = x_start)

        return pred_noise, x_start

    def predict_noise_from_start(self, x_t, t, x_start):
        """Inverse of ``predict_start_from_noise`` — recover the noise implied by a (possibly clipped) x_0."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_start
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond = None, cond_scale = 1.):
        _, x_recon = self.model_predictions(
            x, t, cond = cond, cond_scale = cond_scale, clip_denoised = clip_denoised
        )

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond = None, cond_scale = 1., clip_denoised = True):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(
            x = x, t = t, clip_denoised = clip_denoised, cond = cond, cond_scale = cond_scale
        )
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond = None, cond_scale = 1., progress = True):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        steps = tqdm(
            reversed(range(0, self.num_timesteps)),
            desc = 'sampling loop time step',
            total = self.num_timesteps,
            disable = not progress
        )

        for i in steps:
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond = cond, cond_scale = cond_scale)

        return unnormalize_img(img)

    @torch.inference_mode()
    def ddim_sample(
        self,
        shape,
        cond = None,
        cond_scale = 1.,
        sampling_timesteps = None,
        eta = None,
        clip_denoised = True,
        progress = True
    ):
        """
        Sample with DDIM (https://arxiv.org/abs/2010.02502).

        Denoises over a strided subsequence of the training chain, so a model trained with
        ``timesteps = 1000`` can be sampled in 50 network passes instead of 1000. No retraining
        is involved — the same epsilon-prediction is reused on a shorter schedule.

        ``eta = 0`` is deterministic: the same noise always yields the same video, which is what
        makes latent interpolation and reproducible outputs possible. ``eta = 1`` recovers the
        stochastic DDPM update.
        """
        device = self.betas.device

        steps = default(sampling_timesteps, self.sampling_timesteps)
        eta = default(eta, self.ddim_sampling_eta)
        assert 1 <= steps <= self.num_timesteps, (
            f'sampling_timesteps must be in [1, {self.num_timesteps}], got {steps}'
        )

        # a strided subsequence of [0, num_timesteps), walked in reverse, paired with its successor.
        # the final pair steps to -1, which denotes x_0 itself.

        img = torch.randn(shape, device = device)

        return self.ddim_loop(
            img, self.num_timesteps, steps,
            cond = cond, cond_scale = cond_scale, eta = eta,
            clip_denoised = clip_denoised, progress = progress
        )

    @torch.inference_mode()
    def ddim_loop(
        self,
        img,
        from_step,
        steps,
        cond = None,
        cond_scale = 1.,
        eta = 0.,
        clip_denoised = True,
        progress = True,
        desc = 'ddim sampling time step'
    ):
        """
        Walk a strided subsequence of ``[0, from_step)`` in reverse, denoising ``img`` as it goes.

        Shared by :meth:`ddim_sample` and :meth:`interpolate`, which differ only in where the
        starting image comes from and how far up the chain they begin.
        """
        device = img.device
        b = img.shape[0]

        times = torch.linspace(-1, from_step - 1, steps = steps + 1).flip(0).long().tolist()
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in tqdm(time_pairs, desc = desc, disable = not progress):
            t = torch.full((b,), time, device = device, dtype = torch.long)

            pred_noise, x_start = self.model_predictions(
                img, t, cond = cond, cond_scale = cond_scale, clip_denoised = clip_denoised
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).clamp(min = 0.).sqrt()

            img = x_start * alpha_next.sqrt() + c * pred_noise

            if eta > 0:
                img = img + sigma * torch.randn_like(img)

        return unnormalize_img(img)

    @torch.inference_mode()
    def sample(
        self,
        cond = None,
        cond_scale = 1.,
        batch_size = 16,
        sampling_timesteps = None,
        eta = None,
        progress = True
    ):
        """
        Generate videos, returned in [0, 1] with shape ``(batch, channels, frames, h, w)``.

        Uses the full DDPM chain by default. Pass ``sampling_timesteps`` (or set it on the model)
        below ``timesteps`` to take the much faster DDIM path instead. Set ``progress = False``
        to silence the progress bar, which is what you want inside scripts, notebooks and CI.
        """
        cond = self.resolve_cond(cond)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        shape = (batch_size, channels, num_frames, image_size, image_size)

        steps = default(sampling_timesteps, self.sampling_timesteps)

        if steps < self.num_timesteps:
            return self.ddim_sample(
                shape, cond = cond, cond_scale = cond_scale,
                sampling_timesteps = steps, eta = eta, progress = progress
            )

        return self.p_sample_loop(shape, cond = cond, cond_scale = cond_scale, progress = progress)

    @torch.inference_mode()
    def interpolate(
        self, x1, x2, t = None, lam = 0.5, cond = None, cond_scale = 1.,
        sampling_timesteps = None, eta = None, progress = True
    ):
        """
        Interpolate between two videos (values in [0, 1]) by noising both to step ``t`` and
        denoising the mix.

        Takes ``sampling_timesteps`` and ``eta`` exactly as :meth:`sample` does, so a blend can
        use the DDIM path instead of walking every step from ``t`` down to zero.
        """
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        cond = self.resolve_cond(cond)

        x1, x2 = normalize_img(x1), normalize_img(x2)

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        steps = default(sampling_timesteps, min(self.sampling_timesteps, t))

        if steps < t:
            return self.ddim_loop(
                img, t, steps,
                cond = cond, cond_scale = cond_scale,
                eta = default(eta, self.ddim_sampling_eta),
                progress = progress, desc = 'ddim interpolation time step'
            )

        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t, disable = not progress):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond = cond, cond_scale = cond_scale)

        return unnormalize_img(img)

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond = None, noise = None, **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        cond = self.resolve_cond(cond)

        # an explicit null_cond_prob from the caller wins; otherwise use the configured rate
        kwargs.setdefault('null_cond_prob', self.cond_drop_prob)

        x_recon = self.denoise_fn(x_noisy, t, cond = cond, **kwargs)

        target = noise if self.objective == 'noise' else self.predict_v(x_start, t, noise)

        if self.loss_type == 'l1':
            loss = F.l1_loss(target, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(target, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, device, img_size, = x.shape[0], x.device, self.image_size
        check_shape(x, 'b c f h w', c = self.channels, f = self.num_frames, h = img_size, w = img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x = normalize_img(x)
        return self.p_losses(x, t, *args, **kwargs)


# backwards-compatible alias for the upstream class name
GaussianDiffusion = VideoDiffusion
