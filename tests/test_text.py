"""Text-conditioning path with BERT mocked out (no network / 400 MB download in CI)."""
import torch

from video_diffusion_pytorch import GaussianDiffusion, Unet3D
from video_diffusion_pytorch import text as text_mod
from video_diffusion_pytorch.text import BERT_MODEL_DIM


class FakeOutputs:
    def __init__(self, h):
        self.hidden_states = [h]


class FakeBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(100, BERT_MODEL_DIM)

    def forward(self, input_ids, attention_mask = None, output_hidden_states = False):
        return FakeOutputs(self.emb(input_ids))


class FakeTokenizer:
    def __call__(self, texts, add_special_tokens = True, padding = True, return_tensors = 'pt'):
        n = max(len(t.split()) for t in texts) + 1
        ids = torch.zeros(len(texts), n, dtype = torch.long)
        for i, t in enumerate(texts):
            ids[i, 0] = 99  # [CLS]
            for j, w in enumerate(t.split(), start = 1):
                ids[i, j] = (hash(w) % 97) + 1

        class Enc:
            input_ids = ids
        return Enc()


def _patch(monkeypatch):
    monkeypatch.setattr(text_mod, '_TOKENIZER', FakeTokenizer())
    monkeypatch.setattr(text_mod, '_MODEL', FakeBert())


def test_bert_embed_masked_mean_and_cls(monkeypatch):
    _patch(monkeypatch)
    ids = text_mod.tokenize(['a cat', 'fireworks with blue and green sparkles'])
    mean = text_mod.bert_embed(ids, device = 'cpu')
    cls = text_mod.bert_embed(ids, return_cls_repr = True, device = 'cpu')
    assert mean.shape == cls.shape == (2, BERT_MODEL_DIM)
    assert not torch.allclose(mean, cls)


def test_string_conditioning_end_to_end(monkeypatch):
    _patch(monkeypatch)
    model = Unet3D(dim = 8, dim_mults = (1, 2), attn_heads = 2, attn_dim_head = 8, use_bert_text_cond = True)
    diff = GaussianDiffusion(model, image_size = 16, num_frames = 2, timesteps = 5)
    texts = ['a cat', 'fireworks']
    diff(torch.rand(2, 3, 2, 16, 16), cond = texts).backward()
    assert diff.sample(cond = texts, cond_scale = 2.).shape == (2, 3, 2, 16, 16)
