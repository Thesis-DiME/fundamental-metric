from typing import Any, Dict, List
from PIL import Image

import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance


class FIDMetric(FrechetInceptionDistance):
    def __init__(self, **kwargs: Any) -> None:
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

    def update_real_images(
        self,
        real_images: List[Image.Image],
    ) -> None:

        preprocessed_images = self._preprocess_images(real_images).to(self.device)
        self.update(preprocessed_images, real=True)

    def update_generated_images(
        self,
        generated_images: List[Image.Image],
    ) -> None:

        preprocessed_images = self._preprocess_images(generated_images).to(self.device)
        self.update(preprocessed_images, real=False)

    def compute(self) -> Dict[str, float]:
        return {"fid_torchmetrics": super().compute()}
    