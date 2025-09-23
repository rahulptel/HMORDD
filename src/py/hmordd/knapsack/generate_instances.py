from math import ceil
from pathlib import Path
import hydra

from hmordd.common.instances import generate_instances_for_problem
from hmordd.knapsack import PROB_NAME, PROB_PREFIX


def generate_instance_knapsack(rng, cfg):
    """Generates a multi-objective knapsack instance (uncorrelated)."""
    if getattr(cfg, "inst_type", "uncorr") != "uncorr":
        raise ValueError("Only 'uncorr' inst_type is supported in this generator.")

    weight = rng.randint(cfg.cons_lb, cfg.cons_ub + 1, cfg.n_vars)
    value = [list(rng.randint(cfg.obj_lb, cfg.obj_ub + 1, cfg.n_vars)) for _ in range(cfg.n_objs)]

    capacity = int(ceil(0.5 * float(weight.sum())))

    return {
        "n_vars": cfg.n_vars,
        "n_cons": 1,
        "n_objs": cfg.n_objs,
        "weight": weight.tolist(),
        "value": value,
        "capacity": capacity,
    }


def write_knapsack_instance(inst_path, data):
    text_lines = [
        str(data["n_vars"]),
        str(data["n_cons"]),
        str(data["n_objs"]),
    ]
    text_lines.extend(" ".join(map(str, coeffs)) for coeffs in data["value"])
    text_lines.append(" ".join(map(str, data["weight"])))
    text_lines.append(str(int(data["capacity"])))

    output_path = Path(str(inst_path) + ".dat")
    output_path.write_text("\n".join(text_lines))


@hydra.main(config_path="./configs", config_name="generate_instances.yaml", version_base="1.2")
def main(cfg):
    cfg.zip_output = getattr(cfg, "zip_output", False)
    generate_instances_for_problem(
        cfg,
        PROB_NAME,
        PROB_PREFIX,
        generate_instance_knapsack,
        write_knapsack_instance,
        size_delimiter="_",
        file_delimiter="_",
        zip_output=cfg.zip_output,
    )


if __name__ == "__main__":
    main()
