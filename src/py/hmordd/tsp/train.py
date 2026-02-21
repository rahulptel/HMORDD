"""CLI entrypoint to train the TSP Pareto node predictor."""

import hydra

from hmordd.tsp.trainer import TSPTrainer


@hydra.main(config_path="./configs", config_name="train.yaml", version_base="1.2")
def main(cfg):
    trainer = TSPTrainer(cfg)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
