"""Small helpers shared across the package."""

import logging

import torch

logger = logging.getLogger(__name__)

_COMPILE_SUPPORTED = None


def compile_supported():
    """
    Whether ``torch.compile`` can actually build on this machine.

    Compilation is lazy — ``torch.compile`` returns happily and only fails on the first forward
    pass, deep inside a training loop. It also needs a toolchain the installing user may not
    have: Triton for the CUDA backend, a C++ compiler for the CPU one. Rather than catch a wide
    family of backend errors mid-training, build one trivial module up front and find out.

    The result is cached, so the probe runs at most once per process.
    """
    global _COMPILE_SUPPORTED

    if _COMPILE_SUPPORTED is None:
        try:
            torch.compile(torch.nn.Linear(4, 4))(torch.zeros(1, 4))
            _COMPILE_SUPPORTED = True
        except Exception as e:  # noqa: BLE001 - a probe; any failure means unavailable
            logger.info('torch.compile is unavailable here (%s)', type(e).__name__)
            _COMPILE_SUPPORTED = False

    return _COMPILE_SUPPORTED

def exists(x):
    return x is not None

def noop(*args, **kwargs):
    pass

def is_odd(n):
    return (n % 2) == 1

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        yield from dl

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all(isinstance(el, str) for el in x)

def embed_text(texts, device, return_cls_repr = False):
    from kinema.text import bert_embed, tokenize
    return bert_embed(tokenize(texts), return_cls_repr = return_cls_repr, device = device)

def identity(t, *args, **kwargs):
    return t

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5
