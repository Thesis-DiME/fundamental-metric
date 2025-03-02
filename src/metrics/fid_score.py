import torch
from typing import List, Any

import torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import Compose, Resize
from PIL import Image
import numpy as np

from utils import create_dummy_image


class FIDScore(torchmetrics.Metric):
    """Frechet Inception Distance (FID) metric calculator for PIL images."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fid = FrechetInceptionDistance()
        self.transform = self._create_transform()

    def _create_transform(self) -> Compose:
        """Create transformation pipeline that outputs uint8 tensors."""
        return Compose(
            [
                Resize((299, 299)),
                lambda img: torch.from_numpy(np.array(img)).permute(
                    2, 0, 1
                ),  # HWC to CHW
            ]
        )

    def update(
        self, real_images: List[Image.Image], fake_images: List[Image.Image]
    ) -> None:
        """Update metric state with new batch of real and generated PIL images."""
        real_batch = torch.stack([self.transform(img) for img in real_images])
        fake_batch = torch.stack([self.transform(img) for img in fake_images])
        self.fid.update(real_batch, real=True)
        self.fid.update(fake_batch, real=False)

    def compute(self) -> torch.Tensor:
        return self.fid.compute()


if __name__ == "__main__":
    real_imgs = [create_dummy_image() for _ in range(10)]
    fake_imgs = [create_dummy_image() for _ in range(10)]

    fid_calculator = FIDScore()
    fid_calculator.update(real_imgs, fake_imgs)
    fid_value = fid_calculator.compute()

    print(f"FID Score: {fid_value.item():.2f}")
