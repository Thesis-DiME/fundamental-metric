from PIL import Image
import numpy as np
import torch


def create_dummy_image() -> Image.Image:
    """Generate a random RGB PIL image."""
    array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(array)


def create_dummy_tensor(batch_size: int) -> torch.Tensor:
    return torch.rand(batch_size, 3, 256, 256)
