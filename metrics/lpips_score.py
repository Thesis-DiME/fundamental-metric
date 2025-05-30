import torch
from torch import Tensor
from torchmetrics import Metric
import lpips
from typing import List
from PIL import Image
from torchvision import transforms


class LPIPSSimilarity(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, net_type: str = "alex", **kwargs):
        super().__init__(**kwargs)

        assert net_type in ['alex', 'vgg', 'squeeze'], "net_type must be one of: 'alex', 'vgg', 'squeeze'"
        self.loss_fn = lpips.LPIPS(net=net_type)
        self.loss_fn.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # LPIPS expects 256x256
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # scale to [-1, 1]
        ])

        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def _prepare_images(self, images: List[Image.Image]) -> Tensor:
        return torch.stack([self.transform(img) for img in images])

    def update(self, sources: List[Image.Image], targets: List[Image.Image]) -> None:
        assert len(sources) == len(targets), "Sources and targets must be the same length"

        source_tensor = self._prepare_images(sources).to(self.device)
        target_tensor = self._prepare_images(targets).to(self.device)

        for i in range(len(source_tensor)):
            dist = self.loss_fn(source_tensor[i].unsqueeze(0), target_tensor[i].unsqueeze(0))
            self.scores.append(dist.squeeze().detach())
            self.count += 1

    def compute(self) -> Tensor:
        if self.count == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(self.scores).mean()
