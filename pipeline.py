import hydra
from omegaconf import DictConfig

import torch
import json
from PIL import Image
import pandas as pd
import os
from pathlib import Path


class FundamentalMetricsPipeline:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.metrics = {}
        for name, params in cfg.metrics.items():
            self.metrics[name] = hydra.utils.instantiate(params).to(self.device)

    def load_data(self):
        with open(self.cfg.metadata_path, "r") as f:
            metadata = json.load(f)

        real_tensors = []
        text_tokens = []
        img_paths = []

        for data in metadata:
            try:
                metadata_path = Path(self.cfg.metadata_path)
                image = Image.open(metadata_path.parent / data["img_path"]).convert(
                    "RGB"
                )
            except Exception as e:
                print(f"Error loading image at {data['img_path']}: {e}")
                continue

            real_tensors.append(image)
            text_tokens.append(data["prompt"])
            img_paths.append(data["img_path"])

        return {
            "real_images": real_tensors,
            "text_tokens": text_tokens,
            "generated_images": real_tensors,
            "img_paths": img_paths,
        }

    def initialize_csv(self, data):
        """Create CSV if it doesn't exist."""
        if not os.path.exists(self.cfg.csv_path):
            df = pd.DataFrame(
                {"img_path": data["img_paths"], "prompt": data["text_tokens"]}
            )
            df.to_csv(self.cfg.csv_path, index=False)
            print(f"[✔] Initialized CSV at {self.cfg.csv_path}")

    def compute_individual_metrics(self):
        data = self.load_data()
        self.initialize_csv(data)  # Ensure CSV exists

        df = pd.read_csv(self.cfg.csv_path)
        clip_scores = []

        for real_image, generated_image, text_token, img_path in zip(
            data["real_images"],
            data["generated_images"],
            data["text_tokens"],
            data["img_paths"],
        ):
            if "clip" in self.metrics:
                self.metrics["clip"].update([text_token], [generated_image])
                clip_score = self.metrics["clip"].compute().item()
                clip_scores.append(clip_score)

        if clip_scores:
            df["clip_score"] = clip_scores

        df.to_csv(self.cfg.csv_path, index=False)
        return df

    def compute_group_metrics(self):
        data = self.load_data()
        results = {}

        self.initialize_csv(data)  # Ensure CSV exists
        df = pd.read_csv(self.cfg.csv_path)

        if "fid" in self.metrics:
            self.metrics["fid"].update_real_images(data["real_images"])
            self.metrics["fid"].update_generated_images(data["generated_images"])
            results["fid_score"] = (
                self.metrics["fid"].compute()["fid_torchmetrics"].item()
            )
            df["fid_score"] = results["fid_score"]

        if "inception" in self.metrics:
            self.metrics["inception"].update_images(data["generated_images"])
            inception = self.metrics["inception"].compute()
            results["inception_score_mean"] = inception["inception_mean"].item()
            results["inception_score_std"] = inception["inception_std"].item()
            df["inception_score_mean"] = results["inception_score_mean"]
            df["inception_score_std"] = results["inception_score_std"]

        df.to_csv(self.cfg.csv_path, index=False)
        return results
