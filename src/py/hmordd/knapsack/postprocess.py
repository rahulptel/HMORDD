import hydra
from hmordd.common.postprocess_grid import (
    ExperimentSpec,
    nsga2_variant,
    restricted_variant,
    run_postprocess_grid,
)


EXPERIMENT_SPECS = (
    ExperimentSpec(
        n_objs=7,
        n_vars=40,
        restricted_variants=(
            restricted_variant("Scal+-2000", nosh="Scal+", width=2000),
            restricted_variant("Scal+-3000", nosh="Scal+", width=3000),
            restricted_variant("FE-2000", nosh="FE", width=2000),
            restricted_variant("FE-3000", nosh="FE", width=3000),
        ),
        nsga2_variants=(
            nsga2_variant("pop100_time60", pop_size=100, run_time=60),
            nsga2_variant("pop500_time60", pop_size=500, run_time=60),        
        ),
    ),
    ExperimentSpec(
        n_objs=4,
        n_vars=50,
        restricted_variants=(
            restricted_variant("Scal+-2500", nosh="Scal+", width=2500),
            restricted_variant("Scal+-3500", nosh="Scal+", width=3500),
            restricted_variant("FE-2500", nosh="FE", width=2500),
            restricted_variant("FE-3500", nosh="FE", width=3500),
        ),
        nsga2_variants=(
            nsga2_variant("pop100_time12", pop_size=100, run_time=12),
            nsga2_variant("pop500_time12", pop_size=500, run_time=12),            
        ),
    ),
    ExperimentSpec(
        n_objs=3,
        n_vars=80,
        restricted_variants=(
            restricted_variant("Scal+-4000", nosh="Scal+", width=4000),
            restricted_variant("Scal+-6000", nosh="Scal+", width=6000),
            restricted_variant("FE-4000", nosh="FE", width=4000),
            restricted_variant("FE-6000", nosh="FE", width=6000),
        ),
        nsga2_variants=(
            nsga2_variant("pop100_time58", pop_size=100, run_time=58),
            nsga2_variant("pop500_time58", pop_size=500, run_time=58),            
        ),
    ),
)


@hydra.main(config_path="./configs", config_name="postprocess.yaml", version_base="1.2")
def main(cfg):
    run_postprocess_grid(cfg, EXPERIMENT_SPECS)


if __name__ == "__main__":
    main()
