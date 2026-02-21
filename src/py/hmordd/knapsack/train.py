"""CLI entrypoint to train the XGBoost node scorer for knapsack DDs."""

import hydra

from hmordd.knapsack.trainer import XGBTrainer


@hydra.main(config_path="./configs", config_name="train.yaml", version_base="1.2")
def main(cfg):
    trainer = XGBTrainer(cfg)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
