import torch
from torch import Tensor
import torch.nn.functional as F
from torchmetrics import Metric
import clip


class CLIPSimilarity(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    
    def __init__(
        self,
        mode: str = 'text_image',
        model_name: str = 'ViT-B/32',
        **kwargs
    ):
        super().__init__(**kwargs)
        
        assert mode in ['text_image', 'image_image'], "Mode must be 'text_image' or 'image_image'"
        self.mode = mode
        
        self.model, _ = clip.load(model_name)
        self.model.eval()
        
        self.add_state("similarities", default=[], dist_reduce_fx="cat")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def _prepare_images(self, images: Tensor) -> Tensor:
        if images.max() > 1.0:
            images = images / 255.0
            
        images = F.interpolate(images, size=(224, 224), mode='bilinear')
        
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device)
        images = (images - mean[None, :, None, None]) / std[None, :, None, None]
        
        return images

    def _prepare_text(self, text_tensors: Tensor) -> Tensor:
        return self.model.encode_text(text_tensors)
    
    def update(
        self,
        sources: Tensor,
        targets: Tensor
    ) -> None:

        if self.mode == 'text_image':
            source_features = self._prepare_text(sources)
        else:
            source_features = self.model.encode_image(self._prepare_images(sources))
            
        target_features = self.model.encode_image(self._prepare_images(targets))
        
        source_features = F.normalize(source_features, p=2, dim=-1)
        target_features = F.normalize(target_features, p=2, dim=-1)
        sim = (source_features * target_features).sum(dim=-1)
        
        # Update states
        self.similarities.append(sim.detach())
        self.count += len(sim)
    
    def compute(self) -> Tensor:
        if self.count == 0:
            return torch.tensor(0.0)
        return torch.cat(self.similarities).mean()