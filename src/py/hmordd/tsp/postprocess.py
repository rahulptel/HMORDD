import hydra
from hmordd.common.postprocess_grid import (
    ExperimentSpec,
    nsga2_variant,
    restricted_variant,
    run_postprocess_grid,
)


TSP_RESTRICTED_VARIANTS = (
    restricted_variant("OrdMeanHigh", nosh="OrdMeanHigh"),
    restricted_variant("OrdMaxHigh", nosh="OrdMaxHigh"),
    restricted_variant("OrdMinHigh", nosh="OrdMinHigh"),
    restricted_variant("OrdMeanLow", nosh="OrdMeanLow"),
    restricted_variant("OrdMaxLow", nosh="OrdMaxLow"),
    restricted_variant("OrdMinLow", nosh="OrdMinLow"),
    restricted_variant("E2E", nosh="E2E"),
)

EXPERIMENT_SPECS = (
    ExperimentSpec(
        n_objs=3,
        n_vars=15,
        restricted_variants=TSP_RESTRICTED_VARIANTS,
        nsga2_variants=(
            nsga2_variant("pop100_time3", pop_size=100, run_time=3),
            nsga2_variant("pop500_time3", pop_size=500, run_time=3),
        ),
    ),
    ExperimentSpec(
        n_objs=4,
        n_vars=15,
        restricted_variants=TSP_RESTRICTED_VARIANTS,
        nsga2_variants=(
            nsga2_variant("pop100_time25", pop_size=100, run_time=25),
            nsga2_variant("pop500_time25", pop_size=500, run_time=25),
        ),
    ),
)


@hydra.main(config_path="./configs", config_name="postprocess.yaml", version_base="1.2")
def main(cfg):
    run_postprocess_grid(cfg, EXPERIMENT_SPECS)


if __name__ == "__main__":
    main()
