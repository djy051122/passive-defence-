import pytest
import torch

from data.wavelet import TwoLevelHaarDWT


def test_two_level_rgb_shapes() -> None:
    image = torch.rand(3, 224, 224)

    bands = TwoLevelHaarDWT()(image)

    assert bands.ll2.shape == (3, 56, 56)
    assert bands.h1.shape == (9, 112, 112)
    assert bands.h2.shape == (9, 56, 56)


def test_two_level_batch_shapes() -> None:
    images = torch.rand(2, 3, 64, 64)

    bands = TwoLevelHaarDWT()(images)

    assert bands.ll2.shape == (2, 3, 16, 16)
    assert bands.h1.shape == (2, 9, 32, 32)
    assert bands.h2.shape == (2, 9, 16, 16)


def test_rejects_size_not_divisible_by_four() -> None:
    image = torch.rand(3, 63, 64)

    with pytest.raises(ValueError, match="divisible by 4"):
        TwoLevelHaarDWT()(image)
