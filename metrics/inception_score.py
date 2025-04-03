from typing import Dict, List
from PIL import Image

import torch
from torchvision import transforms
from torchmetrics.image.inception import InceptionScore


class InceptionScoreMetric(InceptionScore):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _preprocess_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        """
        Preprocess a list of PIL images into a batch tensor with pixel values in [0, 255].
        """
        tensors = [self.preprocess(img) for img in pil_images]
        
        return (torch.stack(tensors) * 255).to(torch.uint8)

    def update_images(self, images: List[Image.Image]) -> None:
        """
        Update the metric with images (as a list of PIL images).
        """
        preprocessed_images = self._preprocess_images(images).to(self.device)
        self.update(preprocessed_images)

    def compute(self) -> Dict[str, float]:
        """
        Compute the final Inception Score.
        """
        inception_mean, inception_std = super().compute()
        return {"inception_mean": inception_mean, "inception_std": inception_std}