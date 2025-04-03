import hydra
from omegaconf import DictConfig

import torch
import clip
import json
from torchvision import transforms
import os
from PIL import Image


class MetricsPipeline:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        self.metrics = {}
        for name, params in cfg.metrics.items():
            self.metrics[name] = hydra.utils.instantiate(params).to(self.device)


    def load_data(self):
        
        metadata_path = '/home/naumov/code/general-pipeline/data/generated_images/stable-diffusion-v1-5-stable-diffusion-v1-5/2/metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        real_tensors = []
        text_tokens = []
        
        for data in metadata:
            try:
                image = Image.open(data['img_path']).convert('RGB')
            except Exception as e:
                print(f"Error loading image at {data['img_path']}: {e}")
                continue
            
            real_tensors.append(image)
            text_tokens.append(data['prompt'])

        return {
            "real_images": real_tensors,
            "text_tokens": text_tokens,
            "generated_images": real_tensors
        }

    def compute_metrics(self):
        data = self.load_data()
        results = {}

        if 'clip' in self.metrics:
            self.metrics['clip'].update(data['text_tokens'], data['generated_images'])
            results['clip_score'] = self.metrics['clip'].compute().item()

        if 'fid' in self.metrics:
            self.metrics['fid'].update_real_images(data['real_images'])
            self.metrics['fid'].update_generated_images(data['generated_images'])
            results['fid_score'] = self.metrics['fid'].compute()['fid_torchmetrics'].item()

        if 'inception' in self.metrics:
            self.metrics['inception'].update_images(data['generated_images'])
            results['inception_score_mean'] = self.metrics['inception'].compute()['inception_mean'].item()
            results['inception_score_std'] = self.metrics['inception'].compute()['inception_std'].item()
        return results


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    pipeline = MetricsPipeline(cfg)
    results = pipeline.compute_metrics()
    
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric.upper()}: {value:.4f}")


if __name__ == "__main__":
    main()