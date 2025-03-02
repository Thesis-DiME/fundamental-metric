import torch
import numpy as np
from PIL import Image
import torchmetrics
from torchmetrics.image.inception import InceptionScore
from torchvision.transforms import Compose, Resize

from utils import create_dummy_image


class InceptionScoreMetric(torchmetrics.Metric):
    def __init__(self, splits: int = 10, **kwargs) -> None:
        super().__init__(**kwargs)
        self.inception_score = InceptionScore(splits=splits)
        self._transform = Compose(
            [
                Resize((299, 299)),
                lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1),
            ]
        )

    def update(self, images: list[Image.Image]) -> None:
        processed = torch.stack([self._transform(img) for img in images])
        self.inception_score.update(processed.to(torch.uint8))

    def compute(self) -> torch.Tensor:
        return self.inception_score.compute()


if __name__ == "__main__":
    sample_images = [create_dummy_image() for _ in range(10)]
    metric = InceptionScoreMetric()
    metric.update(sample_images)
    result = metric.compute()
    print(f"Inception Score: {result}")
