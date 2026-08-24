"""Text conditioning helpers: tokenize strings and embed them with BERT.

``transformers`` is imported lazily so the core model has no dependency on it
unless text conditioning is actually used (``pip install video-diffusion-pytorch[text]``).
"""

from __future__ import annotations

from typing import Sequence, Union

import torch
from einops import rearrange

BERT_MODEL_NAME = 'bert-base-cased'
BERT_MODEL_DIM = 768

# lazily initialised singletons

_MODEL = None
_TOKENIZER = None


def exists(val):
    return val is not None


def _require_transformers():
    try:
        import transformers  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            'text conditioning requires the `transformers` package. '
            'Install with `pip install video-diffusion-pytorch[text]`'
        ) from e
    return transformers


def get_tokenizer():
    global _TOKENIZER
    if not exists(_TOKENIZER):
        transformers = _require_transformers()
        _TOKENIZER = transformers.AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    return _TOKENIZER


def get_bert(device: Union[torch.device, str, None] = None):
    global _MODEL
    if not exists(_MODEL):
        transformers = _require_transformers()
        _MODEL = transformers.AutoModel.from_pretrained(BERT_MODEL_NAME).eval()
        for p in _MODEL.parameters():
            p.requires_grad_(False)

    if exists(device):
        _MODEL = _MODEL.to(device)

    return _MODEL


def tokenize(texts: Union[str, Sequence[str]], add_special_tokens: bool = True) -> torch.Tensor:
    """Tokenize one or more strings into a padded ``(batch, seq)`` LongTensor."""
    if not isinstance(texts, (list, tuple)):
        texts = [texts]

    tokenizer = get_tokenizer()

    encoding = tokenizer(
        list(texts),
        add_special_tokens = add_special_tokens,
        padding = True,
        return_tensors = 'pt'
    )
    return encoding.input_ids


@torch.no_grad()
def bert_embed(
    token_ids: torch.Tensor,
    return_cls_repr: bool = False,
    eps: float = 1e-8,
    pad_id: int = 0,
    device: Union[torch.device, str, None] = None
) -> torch.Tensor:
    """Embed token ids with BERT.

    Returns the ``[CLS]`` representation if ``return_cls_repr`` else the
    length-normalised mean over all non-pad, non-``[CLS]`` tokens.
    Output shape: ``(batch, BERT_MODEL_DIM)``.
    """
    device = default_device(device)
    model = get_bert(device)

    token_ids = token_ids.to(device)
    mask = token_ids != pad_id

    outputs = model(
        input_ids = token_ids,
        attention_mask = mask,
        output_hidden_states = True
    )

    hidden_state = outputs.hidden_states[-1]

    if return_cls_repr:
        return hidden_state[:, 0]  # [CLS] token as representation

    mask = rearrange(mask[:, 1:], 'b n -> b n 1')  # exclude [CLS]

    numer = (hidden_state[:, 1:] * mask).sum(dim = 1)
    denom = mask.sum(dim = 1)
    return numer / (denom + eps)


def default_device(device = None) -> torch.device:
    if exists(device):
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
