import torch
import torch.nn.functional as F
import torchmetrics
import clip
from torchvision import transforms
from PIL import Image
from typing import List, Union


class CLIPTextImageSimilarity(torchmetrics.Metric):
    def __init__(
        self,
        clip_model: str = "ViT-B/32",
        dist_sync_on_step: bool = False,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.model, self.clip_preprocess = clip.load(clip_model, device=self.device)
        self.add_state("similarities", default=[], dist_reduce_fx="cat")

        self.preprocess = transforms.Compose(
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
            + self.clip_preprocess.transforms[:2]
            + self.clip_preprocess.transforms[4:]
        )

    def get_image_features(
        self, images: Union[List[Image.Image], torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(images[0], Image.Image):
            images = torch.stack([self.clip_preprocess(image) for image in images]).to(
                self.device
            )
        else:
            images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text: str) -> torch.Tensor:
        tokens = clip.tokenize(text).to(self.device)
        return self.model.encode_text(tokens)

    def update(self, text: str, images: Union[List[Image.Image], torch.Tensor]) -> None:
        text_features = self.get_text_features(text)
        image_features = self.get_image_features(images)
        similarity = F.cosine_similarity(text_features, image_features).mean()
        self.similarities.append(similarity)

    def compute(self) -> torch.Tensor:
        return torch.stack(self.similarities).mean()


class CLIPImageImageSimilarity(torchmetrics.Metric):
    def __init__(
        self,
        clip_model: str = "ViT-B/32",
        dist_sync_on_step: bool = False,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.model, self.clip_preprocess = clip.load(clip_model, device=self.device)
        self.add_state("similarities", default=[], dist_reduce_fx="cat")

        self.preprocess = transforms.Compose(
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
            + self.clip_preprocess.transforms[:2]
            + self.clip_preprocess.transforms[4:]
        )

    def get_image_features(
        self, images: Union[List[Image.Image], torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(images[0], Image.Image):
            images = torch.stack([self.clip_preprocess(image) for image in images]).to(
                self.device
            )
        else:
            images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def update(
        self,
        src_images: Union[List[Image.Image], torch.Tensor],
        generated_images: Union[List[Image.Image], torch.Tensor],
    ) -> None:
        src_features = self.get_image_features(src_images)
        gen_features = self.get_image_features(generated_images)
        similarity = F.cosine_similarity(src_features, gen_features).mean()
        self.similarities.append(similarity)

    def compute(self) -> torch.Tensor:
        return torch.stack(self.similarities).mean()


if __name__ == "__main__":
    text_image_similarity = CLIPTextImageSimilarity()
    text = "A photo of a cat"
    images = [Image.open("../../data/example.png").convert("RGB")]
    text_image_similarity.update(text, images)
    print("Text-to-Image Similarity:", text_image_similarity.compute().item())

    image_image_similarity = CLIPImageImageSimilarity()
    src_images = [Image.open("../../data/example1.png").convert("RGB")]
    gen_images = [Image.open("../../data/example2.png").convert("RGB")]
    image_image_similarity.update(src_images, gen_images)
    print("Image-to-Image Similarity:", image_image_similarity.compute().item())
