import hydra
from omegaconf import DictConfig
import pandas as pd
import os
from hydra.utils import get_original_cwd

from pipeline import FundamentalMetricsPipeline


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    pipeline = FundamentalMetricsPipeline(cfg)

    # Compute and save metrics
    pipeline.compute_individual_metrics()
    grouped_results = pipeline.compute_group_metrics()

    # Save to final results folder
    orig_cwd = get_original_cwd()

    os.makedirs(os.path.join(orig_cwd, "results/individual"), exist_ok=True)
    os.makedirs(os.path.join(orig_cwd, "results/grouped"), exist_ok=True)

    individual_df = pd.read_csv(cfg.csv_path)
    individual_path = os.path.join(
        orig_cwd, "results/individual/fundamental_metrics.csv"
    )
    individual_df.to_csv(individual_path, index=False)

    grouped_df = pd.DataFrame([grouped_results])
    grouped_path = os.path.join(orig_cwd, "results/grouped/fundamental_metrics.csv")
    grouped_df.to_csv(grouped_path, index=False)

    print("Evaluation Results:")
    for metric, value in grouped_results.items():
        print(f"{metric.upper()}: {value:.4f}")


if __name__ == "__main__":
    main()
