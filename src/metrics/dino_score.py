import torch
from torchmetrics import Metric
from transformers import ViTImageProcessor, ViTModel
from PIL import Image


class DINOScore(Metric):
    def __init__(self, model_name: str = "facebook/dino-vits16", **kwargs):
        super().__init__(**kwargs)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.model.eval()

        self.add_state(
            "similarity_sum", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    def _extract_features(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        return self.model(**inputs).last_hidden_state[:, 0, :]

    def update(self, sources: list[Image.Image], generated: list[Image.Image]) -> None:
        src_features = self._extract_features(sources)
        gen_features = self._extract_features(generated)
        batch_similarity = torch.nn.functional.cosine_similarity(
            src_features, gen_features
        ).mean()
        self.similarity_sum += batch_similarity * len(sources)
        self.total += len(sources)

    def compute(self) -> torch.Tensor:
        return self.similarity_sum / self.total


if __name__ == "__main__":

    def create_image() -> Image.Image:
        return Image.fromarray(
            torch.randint(0, 255, (256, 256, 3), dtype=torch.uint8).numpy()
        )

    source_imgs = [create_image() for _ in range(8)]
    generated_imgs = [create_image() for _ in range(8)]

    metric = DINOScore()
    metric.update(source_imgs, generated_imgs)
    print(f"DINO Score: {metric.compute().item():.4f}")
