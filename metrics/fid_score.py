from typing import Any, Dict

import torch
from torchmetrics.image.fid import FrechetInceptionDistance


class FIDMetric(FrechetInceptionDistance):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def update_real_images(
        self,
        reference_images: torch.Tensor,
    ) -> None:
        assert (
            torch.max(reference_images) <= 1.0 and torch.min(reference_images) >= 0
        ), "Images must have pixel values in the range [0, 1]."
        preprocessed_images = (reference_images * 255).to(torch.uint8)
        self.update(preprocessed_images, real=True)

    def update_generated_images(
        self,
        generated_images: torch.Tensor,
    ) -> None:
        assert (
            torch.max(generated_images) <= 1.0 and torch.min(generated_images) >= 0
        ), "Images must have pixel values in the range [0, 1]."
        preprocessed_images = (generated_images * 255).to(torch.uint8)
        self.update(preprocessed_images, real=False)

    def compute(self) -> Dict[str, float]:
        return {"fid_torchmetrics": super().compute()}
