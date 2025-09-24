from pathlib import Path

import hydra
from hmordd.common.instances import generate_instances_for_problem
from hmordd.setpacking.utils import PROB_NAME, PROB_PREFIX


def generate_instance_stidsen(rng, cfg):
    """Generates a Stidsen-style set packing instance."""
    data = {
        'n_vars': cfg.n_vars,
        'n_objs': cfg.n_objs,
        'n_cons': int(cfg.n_vars / 5),
        'obj_coeffs': [],
        'cons_coeffs': [],
    }
    items = list(range(1, cfg.n_vars + 1))

    for _ in range(cfg.n_objs):
        data['obj_coeffs'].append(list(rng.randint(cfg.obj_lb, cfg.obj_ub + 1, cfg.n_vars)))

    for _ in range(data['n_cons']):
        vars_in_con = rng.randint(2, (2 * cfg.vars_per_con) + 1)
        data['cons_coeffs'].append(list(rng.choice(items, vars_in_con, replace=False)))

    var_count = []
    for con in data['cons_coeffs']:
        var_count.extend(con)
    missing_vars = list(set(range(1, cfg.n_vars + 1)).difference(set(var_count)))
    for v in missing_vars:
        cons_id = rng.randint(data['n_cons'])
        data['cons_coeffs'][cons_id].append(v)

    return data


def write_to_file_stidsen(inst_path, data):
    """Writes a Stidsen instance to a .dat file."""
    dat = f"{data['n_vars']} {data['n_cons']}\n"
    dat += f"{len(data['obj_coeffs'])}\n"
    for coeffs in data['obj_coeffs']:
        dat += " ".join(list(map(str, coeffs))) + "\n"

    for coeffs in data["cons_coeffs"]:
        dat += f"{len(coeffs)}\n"
        dat += " ".join(list(map(str, coeffs))) + "\n"

    output_path = Path(str(inst_path) + ".dat")
    output_path.write_text(dat)


@hydra.main(config_path="./configs", config_name="generate_instances.yaml", version_base="1.2")
def main(cfg):
    """Generates train/val/test splits for the set packing problem."""
    cfg.zip_output = getattr(cfg, "zip_output", False)
    generate_instances_for_problem(
        cfg,
        PROB_NAME,
        PROB_PREFIX,
        generate_instance_stidsen,
        write_to_file_stidsen,
        size_delimiter="_",
        file_delimiter="_",
        zip_output=cfg.zip_output,
    )


if __name__ == '__main__':
    main()
