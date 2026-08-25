import pytest
import torch
from torch import nn

from kinema.evaluate import (
    frechet_distance,
    frechet_video_distance,
    matrix_sqrt_trace,
    video_features,
)


class DummyExtractor(nn.Module):
    """A tiny stand-in for the pretrained network, so tests need no download."""

    def __init__(self, out_dim = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(3, out_dim)

    def forward(self, x):
        return self.fc(self.pool(x).flatten(1))


def test_identical_sets_score_zero():
    torch.manual_seed(0)
    features = torch.randn(64, 8)
    assert frechet_distance(features, features.clone()) == pytest.approx(0., abs = 1e-6)


def test_a_mean_shift_shows_up_as_its_square():
    """Shifting every dimension by d should cost about d^2 per dimension."""
    torch.manual_seed(0)
    a = torch.randn(512, 8)
    b = a + 2.

    assert frechet_distance(a, b) == pytest.approx(8 * 4., rel = 0.05)


def test_distance_is_symmetric():
    torch.manual_seed(0)
    a, b = torch.randn(128, 8), torch.randn(128, 8) + 1.
    assert frechet_distance(a, b) == pytest.approx(frechet_distance(b, a), rel = 1e-9)


def test_distance_is_never_negative():
    torch.manual_seed(0)
    for _ in range(5):
        a, b = torch.randn(32, 6), torch.randn(32, 6) * 3
        assert frechet_distance(a, b) >= 0.


def test_a_closer_distribution_scores_lower():
    torch.manual_seed(0)
    reference = torch.randn(256, 8)
    near = torch.randn(256, 8) + 0.2
    far = torch.randn(256, 8) + 2.0

    assert frechet_distance(reference, near) < frechet_distance(reference, far)


def test_one_sample_per_side_is_rejected():
    with pytest.raises(AssertionError, match = 'at least two samples'):
        frechet_distance(torch.randn(1, 4), torch.randn(1, 4))


def test_matrix_sqrt_trace_matches_a_known_case():
    """For A = B, Tr(sqrt(A B)) is just the trace of A."""
    torch.manual_seed(0)
    root = torch.randn(6, 6)
    cov = (root @ root.T).double()

    assert matrix_sqrt_trace(cov, cov).item() == pytest.approx(cov.trace().item(), rel = 1e-6)


def test_video_features_shape():
    extractor = DummyExtractor(out_dim = 8)
    features = video_features(torch.rand(5, 3, 4, 16, 16), extractor, batch_size = 2)
    assert features.shape == (5, 8)


def test_video_features_batching_does_not_change_the_result():
    torch.manual_seed(0)
    extractor = DummyExtractor()
    videos = torch.rand(6, 3, 4, 16, 16)

    assert torch.allclose(
        video_features(videos, extractor, batch_size = 6),
        video_features(videos, extractor, batch_size = 2),
        atol = 1e-6
    )


def test_frechet_video_distance_end_to_end():
    torch.manual_seed(0)
    extractor = DummyExtractor()

    real = torch.rand(8, 3, 4, 16, 16)
    similar = real + torch.randn_like(real) * 0.01
    different = torch.rand(8, 3, 4, 16, 16) * 0.2

    close = frechet_video_distance(real, similar, extractor = extractor)
    distant = frechet_video_distance(real, different, extractor = extractor)

    assert close < distant, f'{close} should beat {distant}'


def test_frechet_video_distance_of_a_set_with_itself_is_zero():
    extractor = DummyExtractor()
    videos = torch.rand(6, 3, 4, 16, 16)

    assert frechet_video_distance(videos, videos.clone(), extractor = extractor) == pytest.approx(0., abs = 1e-4)
