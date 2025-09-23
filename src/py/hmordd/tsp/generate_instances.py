import hydra
import numpy as np

from hmordd.common.instances import generate_instances_for_problem
from hmordd.tsp import PROB_NAME, PROB_PREFIX


def generate_instance_tsp(rng, cfg):
    coords = []
    dists = []
    for _ in range(cfg.n_objs):
        points = rng.randint(0, cfg.grid_size, size=(cfg.n_vars, 2))
        diff = points[:, None, :] - points[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)
        distances = dist_matrix.astype(int)
        coords.append(points)
        dists.append(distances)

    return {
        "coords": np.stack(coords),
        "dists": np.stack(dists),
    }


def write_tsp_instance(inst_path, data):
    output_path = inst_path.with_suffix(".npz")
    np.savez(output_path, coords=data["coords"], dists=data["dists"])


@hydra.main(config_path="./configs", config_name="generate_instances.yaml", version_base="1.2")
def main(cfg):
    cfg.zip_output = getattr(cfg, "zip_output", False)
    generate_instances_for_problem(
        cfg,
        PROB_NAME,
        PROB_PREFIX,
        generate_instance_tsp,
        write_tsp_instance,
        size_delimiter="_",
        file_delimiter="_",
        zip_output=cfg.zip_output,
    )


if __name__ == "__main__":
    main()
