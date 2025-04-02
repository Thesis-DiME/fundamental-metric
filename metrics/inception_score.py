import torch
from torchmetrics.image.inception import InceptionScore
from typing import Dict


class InceptionScoreMetric(InceptionScore):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def update_images(self, images: torch.Tensor) -> None:
        assert torch.max(images) <= 1.0 and torch.min(images) >= 0, (
            "Images must have pixel values in the range [0, 1]."
        )
        preprocessed_images = (images * 255).to(torch.uint8)
        self.update(preprocessed_images)

    def compute(self) -> Dict[str, float]:
        inception_mean, inception_std = super().compute()
        return {"inception_mean": inception_mean, "inception_std": inception_std}
