from PIL import Image
import numpy as np


def create_dummy_image() -> Image.Image:
    """Generate a random RGB PIL image."""
    array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(array)
