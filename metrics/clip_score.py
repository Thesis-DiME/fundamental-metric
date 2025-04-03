import torch
from torch import Tensor
import torch.nn.functional as F
from torchmetrics import Metric
import clip
from typing import List
from PIL import Image


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
        
        self.model, self.preprocess = clip.load(model_name)
        self.model.eval()
        
        self.add_state("similarities", default=[], dist_reduce_fx="cat")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def _prepare_images(self, images: List[Image.Image]) -> Tensor: 
        return torch.stack([self.preprocess(img) for img in images]).to(self.device).float()

    def _prepare_text(self, text_list: List[str]) -> Tensor:
        text_tokens = clip.tokenize(text_list).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        
        return text_features.float()
    
    def update(
        self,
        sources: List[str],
        targets: List[Image.Image]
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