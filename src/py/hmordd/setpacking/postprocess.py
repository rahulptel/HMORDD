import hydra
from hmordd.common.postprocess_grid import (
    ExperimentSpec,
    nsga2_variant,
    restricted_variant,
    run_postprocess_grid,
)


EXPERIMENT_SPECS = (
    ExperimentSpec(
        n_objs=3,
        n_vars=100,
        restricted_variants=(restricted_variant("width-50-nosh-1", width=50, nosh_rule=1),),
        nsga2_variants=(
            nsga2_variant("pop100_time1", pop_size=100, run_time=1),
            nsga2_variant("pop500_time1", pop_size=500, run_time=1),
        ),
    ),
    ExperimentSpec(
        n_objs=4,
        n_vars=100,
        restricted_variants=(restricted_variant("width-50-nosh-1", width=50, nosh_rule=1),),
        nsga2_variants=(
            nsga2_variant("pop100_time1", pop_size=100, run_time=1),
            nsga2_variant("pop500_time1", pop_size=500, run_time=1),
        ),
    ),
    ExperimentSpec(
        n_objs=5,
        n_vars=100,
        restricted_variants=(restricted_variant("width-50-nosh-1", width=50, nosh_rule=1),),
        nsga2_variants=(
            nsga2_variant("pop100_time1", pop_size=100, run_time=1),
            nsga2_variant("pop500_time1", pop_size=500, run_time=1),
        ),
    ),
    ExperimentSpec(
        n_objs=6,
        n_vars=100,
        restricted_variants=(restricted_variant("width-50-nosh-1", width=50, nosh_rule=1),),
        nsga2_variants=(
            nsga2_variant("pop100_time2", pop_size=100, run_time=2),
            nsga2_variant("pop500_time2", pop_size=500, run_time=2),
        ),
    ),
    ExperimentSpec(
        n_objs=7,
        n_vars=100,
        restricted_variants=(restricted_variant("width-50-nosh-1", width=50, nosh_rule=1),),
        nsga2_variants=(
            nsga2_variant("pop100_time11", pop_size=100, run_time=11),
            nsga2_variant("pop500_time11", pop_size=500, run_time=11),
        ),
    ),
    ExperimentSpec(
        n_objs=3,
        n_vars=150,
        restricted_variants=(restricted_variant("width-5000-nosh-1", width=5000, nosh_rule=1),),
        nsga2_variants=(
            nsga2_variant("pop100_time1", pop_size=100, run_time=1),
            nsga2_variant("pop500_time1", pop_size=500, run_time=1),
        ),
    ),
    ExperimentSpec(
        n_objs=4,
        n_vars=150,
        restricted_variants=(restricted_variant("width-5000-nosh-1", width=5000, nosh_rule=1),),
        nsga2_variants=(
            nsga2_variant("pop100_time4", pop_size=100, run_time=4),
            nsga2_variant("pop500_time4", pop_size=500, run_time=4),
        ),
    ),
    ExperimentSpec(
        n_objs=5,
        n_vars=150,
        restricted_variants=(restricted_variant("width-5000-nosh-1", width=5000, nosh_rule=1),),
        nsga2_variants=(
            nsga2_variant("pop100_time47", pop_size=100, run_time=47),
            nsga2_variant("pop500_time47", pop_size=500, run_time=47),
        ),
    ),
    ExperimentSpec(
        n_objs=6,
        n_vars=150,
        restricted_variants=(restricted_variant("width-5000-nosh-1", width=5000, nosh_rule=1),),
        nsga2_variants=(
            nsga2_variant("pop100_time221", pop_size=100, run_time=221),
            nsga2_variant("pop500_time221", pop_size=500, run_time=221),
        ),
    ),
    ExperimentSpec(
        n_objs=7,
        n_vars=150,
        restricted_variants=(restricted_variant("width-5000-nosh-1", width=5000, nosh_rule=1),),
        nsga2_variants=(
            nsga2_variant("pop100_time395", pop_size=100, run_time=395),
            nsga2_variant("pop500_time395", pop_size=500, run_time=395),
        ),
    ),
)


@hydra.main(config_path="./configs", config_name="postprocess.yaml", version_base="1.2")
def main(cfg):
    run_postprocess_grid(cfg, EXPERIMENT_SPECS)


if __name__ == "__main__":
    main()
