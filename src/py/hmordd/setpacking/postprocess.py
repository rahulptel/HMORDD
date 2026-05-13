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
            nsga2_variant("pop250_time1", pop_size=250, run_time=1),
        ),
    ),
    ExperimentSpec(
        n_objs=4,
        n_vars=100,
        restricted_variants=(restricted_variant("width-50-nosh-1", width=50, nosh_rule=1),),
        nsga2_variants=(
            nsga2_variant("pop100_time1", pop_size=100, run_time=1),
            nsga2_variant("pop500_time1", pop_size=500, run_time=1),
            nsga2_variant("pop1100_time1", pop_size=1100, run_time=1),
        ),
    ),
    ExperimentSpec(
        n_objs=5,
        n_vars=100,
        restricted_variants=(restricted_variant("width-50-nosh-1", width=50, nosh_rule=1),),
        nsga2_variants=(
            nsga2_variant("pop100_time1", pop_size=100, run_time=1),
            nsga2_variant("pop500_time1", pop_size=500, run_time=1),
            nsga2_variant("pop4600_time1", pop_size=4600, run_time=1),
        ),
    ),
    ExperimentSpec(
        n_objs=6,
        n_vars=100,
        restricted_variants=(restricted_variant("width-50-nosh-1", width=50, nosh_rule=1),),
        nsga2_variants=(
            nsga2_variant("pop100_time2", pop_size=100, run_time=2),
            nsga2_variant("pop500_time2", pop_size=500, run_time=2),
            nsga2_variant("pop9000_time2", pop_size=9000, run_time=2),
        ),
    ),
    ExperimentSpec(
        n_objs=7,
        n_vars=100,
        restricted_variants=(restricted_variant("width-50-nosh-1", width=50, nosh_rule=1),),
        nsga2_variants=(
            nsga2_variant("pop100_time13", pop_size=100, run_time=13),
            nsga2_variant("pop500_time13", pop_size=500, run_time=13),
            nsga2_variant("pop25000_time13", pop_size=25000, run_time=13),
        ),
    ),
    ExperimentSpec(
        n_objs=3,
        n_vars=150,
        restricted_variants=(restricted_variant("width-5000-nosh-1", width=5000, nosh_rule=1),),
        nsga2_variants=(
            nsga2_variant("pop100_time1", pop_size=100, run_time=1),
            nsga2_variant("pop500_time1", pop_size=500, run_time=1),
            nsga2_variant("pop800_time1", pop_size=800, run_time=1),
        ),
    ),
    ExperimentSpec(
        n_objs=4,
        n_vars=150,
        restricted_variants=(restricted_variant("width-5000-nosh-1", width=5000, nosh_rule=1),),
        nsga2_variants=(
            nsga2_variant("pop100_time7", pop_size=100, run_time=7),
            nsga2_variant("pop500_time7", pop_size=500, run_time=7),
            nsga2_variant("pop6100_time7", pop_size=6100, run_time=7),
        ),
    ),
    ExperimentSpec(
        n_objs=5,
        n_vars=150,
        restricted_variants=(restricted_variant("width-5000-nosh-1", width=5000, nosh_rule=1),),
        nsga2_variants=(
            nsga2_variant("pop100_time77", pop_size=100, run_time=77),
            nsga2_variant("pop500_time77", pop_size=500, run_time=77),
            nsga2_variant("pop30000_time77", pop_size=30000, run_time=77),
        ),
    ),
    ExperimentSpec(
        n_objs=6,
        n_vars=150,
        restricted_variants=(restricted_variant("width-5000-nosh-1", width=5000, nosh_rule=1),),
        nsga2_variants=(
            nsga2_variant("pop100_time182", pop_size=100, run_time=182),
            nsga2_variant("pop500_time182", pop_size=500, run_time=182),
            nsga2_variant("pop60000_time182", pop_size=60000, run_time=182),
        ),
    ),
    ExperimentSpec(
        n_objs=7,
        n_vars=150,
        restricted_variants=(restricted_variant("width-5000-nosh-1", width=5000, nosh_rule=1),),
        nsga2_variants=(
            nsga2_variant("pop100_time313", pop_size=100, run_time=313),
            nsga2_variant("pop500_time313", pop_size=500, run_time=313),
            nsga2_variant("pop10400_time313", pop_size=10400, run_time=313),
        ),
    ),
)


@hydra.main(config_path="./configs", config_name="postprocess.yaml", version_base="1.2")
def main(cfg):
    run_postprocess_grid(cfg, EXPERIMENT_SPECS)


if __name__ == "__main__":
    main()
