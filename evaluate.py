import hydra
from omegaconf import DictConfig

import torch
import clip


class MetricsPipeline:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        self.metrics = {}
        for name, params in cfg.metrics.items():
            self.metrics[name] = hydra.utils.instantiate(params).to(self.device)


    def load_data(self):
        real_images = torch.rand(10, 3, 256, 256).to(self.device)
        generated_images = torch.rand(10, 3, 256, 256).to(self.device)
        text_tokens = clip.tokenize(["A cat in space"] * 10).to(self.device)
        
        return {
            "real_images": real_images,
            "generated_images": generated_images,
            "text_tokens": text_tokens
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