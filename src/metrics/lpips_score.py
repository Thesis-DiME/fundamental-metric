from lpips import LPIPS
import torchmetrics
import torch


class LPIPSSimilarity(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.lpips = LPIPS(net="vgg").to(self.device)
        self.add_state("scores", default=[], dist_reduce_fx="cat")

    def update(self, src_images: torch.Tensor, generated_images: torch.Tensor) -> None:
        score = self.lpips(src_images, generated_images).mean()
        self.scores.append(score)

    def compute(self) -> torch.Tensor:
        return torch.stack(self.scores).mean()


if __name__ == "__main__":
    lpips_similarity = LPIPSSimilarity()
    src_tensor = torch.randn(1, 3, 224, 224)
    gen_tensor = torch.randn(1, 3, 224, 224)
    lpips_similarity.update(src_tensor, gen_tensor)
    print("LPIPS Similarity:", lpips_similarity.compute().item())
