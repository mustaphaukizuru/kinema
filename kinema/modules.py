"""Reusable neural-network building blocks for the space-time U-Net."""

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from einops_exts import rearrange_many
from torch import einsum, nn

from kinema.utils import exists


class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

def shift(t, amount, dim):
    """Slide a tensor along ``dim``, filling the vacated edge with zeros rather than wrapping."""
    if amount == 0 or abs(amount) >= t.shape[dim]:
        return t

    size = t.shape[dim] - abs(amount)
    pad = torch.zeros_like(t.narrow(dim, 0, abs(amount)))

    if amount > 0:
        return torch.cat((pad, t.narrow(dim, 0, size)), dim = dim)

    return torch.cat((t.narrow(dim, abs(amount), size), pad), dim = dim)

def token_shift(x, shift_space = False):
    """
    Shift part of the channels one step along time, and optionally along height and width.

    A cheap way to let a feature see its neighbours: half the channels are split evenly between
    the shift directions and displaced by one, the rest pass through untouched. It costs no
    parameters and no attention, which is the point — the network gets local mixing along axes
    that otherwise only communicate through attention layers.

    After CogVideo (https://arxiv.org/abs/2205.15868), which shifts along time.
    """
    channels = x.shape[1]

    directions = [(1, 2), (-1, 2)]
    if shift_space:
        directions += [(1, 3), (-1, 3), (1, 4), (-1, 4)]

    groups = len(directions)
    shiftable = (channels // 2) // groups * groups

    if shiftable == 0:
        return x

    to_shift, passthrough = x[:, :shiftable], x[:, shiftable:]
    chunks = to_shift.chunk(groups, dim = 1)

    shifted = [shift(chunk, amount, dim) for chunk, (amount, dim) in zip(chunks, directions)]
    return torch.cat((*shifted, passthrough), dim = 1)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.scale * self.gamma

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1))
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, token_shift = None):
        """
        ``token_shift`` is ``None``, ``'time'`` or ``'space-time'``. It adds no parameters, so a
        model with it enabled loads checkpoints from one without, and the reverse.
        """
        super().__init__()
        assert token_shift in (None, 'time', 'space-time'), f'unknown token_shift: {token_shift}'
        self.shift_mode = token_shift
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        if exists(self.shift_mode):
            x = token_shift(x, shift_space = self.shift_mode == 'space-time')

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h = self.heads)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b = b)

# attention along space and time

class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None,
        cond_dim = None,
        num_cond_tokens = 4
    ):
        """
        ``cond_dim`` turns on text memory. The conditioning vector is projected into
        ``num_cond_tokens`` key/value pairs that are prepended to the sequence, so every query
        can attend to the caption directly instead of receiving it only as a bias on the
        timestep embedding. Four to eight tokens is the useful range.
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

        self.num_cond_tokens = num_cond_tokens if exists(cond_dim) else 0
        self.to_cond_kv = (
            nn.Linear(cond_dim, num_cond_tokens * hidden_dim * 2, bias = False)
            if exists(cond_dim) else None
        )

    def _cond_kv(self, cond, like):
        """Project the conditioning vector into key/value tokens shaped to match the queries."""
        kv = self.to_cond_kv(cond)
        k, v = rearrange(kv, 'b (two t h d) -> two b h t d', two = 2, t = self.num_cond_tokens, h = self.heads)

        # queries carry leading dims beyond the batch — (b, h w) for temporal attention, say —
        # and the same caption applies across all of them
        while k.ndim < like.ndim:
            k, v = k.unsqueeze(1), v.unsqueeze(1)

        expand_to = (*like.shape[:-2], -1, -1)
        return k.expand(*expand_to), v.expand(*expand_to)

    def forward(
        self,
        x,
        pos_bias = None,
        focus_present_mask = None,
        cond = None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim = -1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # text memory: extra keys and values every query can attend to.
        # prepended, so the caption occupies columns 0 .. num_cond_tokens - 1.

        num_cond = 0
        if exists(self.to_cond_kv) and exists(cond):
            cond_k, cond_v = self._cond_kv(cond, like = q)
            k = torch.cat((cond_k, k), dim = -2)
            v = torch.cat((cond_v, v), dim = -2)
            num_cond = self.num_cond_tokens

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            # memory tokens have no position, so they take no bias
            if num_cond > 0:
                pos_bias = F.pad(pos_bias, (num_cond, 0), value = 0.)
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device = device, dtype = torch.bool)
            attend_self_mask = torch.eye(n, device = device, dtype = torch.bool)

            if num_cond > 0:
                # the caption stays visible even to a query arrested on the present
                always = torch.ones((n, num_cond), device = device, dtype = torch.bool)
                attend_all_mask = torch.cat((always, attend_all_mask), dim = -1)
                attend_self_mask = torch.cat((always, attend_self_mask), dim = -1)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)
