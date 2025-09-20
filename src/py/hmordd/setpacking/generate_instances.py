from pathlib import Path
import hydra
import numpy as np

# Import the new path management class
from hmordd import Paths
from hmordd.setpacking.utils import PROB_NAME, PROB_PREFIX


def generate_instance_stidsen(rng, cfg):
    """Generates a Stidsen-style set packing instance."""
    data = {'n_vars': cfg.n_vars, 'n_objs': cfg.n_objs, 'n_cons': int(cfg.n_vars / 5), 'obj_coeffs': [],
            'cons_coeffs': []}
    items = list(range(1, cfg.n_vars + 1))

    # Generate objective function coefficients
    for _ in range(cfg.n_objs):
        data['obj_coeffs'].append(list(rng.randint(cfg.obj_lb, cfg.obj_ub + 1, cfg.n_vars)))

    # Generate constraints
    for _ in range(data['n_cons']):
        vars_in_con = rng.randint(2, (2 * cfg.vars_per_con) + 1)
        data['cons_coeffs'].append(list(rng.choice(items, vars_in_con, replace=False)))

    # Ensure no variable is left out of all constraints
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

    inst_path = Path(str(inst_path) + ".dat")
    inst_path.write_text(dat)


@hydra.main(config_path="./configs", config_name="generate_instances.yaml", version_base="1.2")
def main(cfg):
    """Main function to generate train, validation, and test instances."""
    rng = np.random.RandomState(cfg.seed)
    
    # Define instance size string for directory naming
    cfg.size = f"{cfg.n_objs}-{cfg.n_vars}"

    # Use the new Paths class to define output directories
    base_path = Paths.resources / "instances" / PROB_NAME / cfg.size
    
    # Create train instances
    train_path = base_path / 'train'
    train_path.mkdir(parents=True, exist_ok=True)
    print(f"Creating training instances in: {train_path}")
    for i in range(cfg.n_train):
        instance_data = generate_instance_stidsen(rng, cfg)
        instance_path = train_path / f'{PROB_PREFIX}_{cfg.seed}_{cfg.size}_{i}'
        write_to_file_stidsen(instance_path, instance_data)

    # Create validation instances
    val_path = base_path / 'val'
    val_path.mkdir(parents=True, exist_ok=True)
    print(f"Creating validation instances in: {val_path}")
    start = cfg.n_train
    end = start + cfg.n_val
    for i in range(start, end):
        instance_data = generate_instance_stidsen(rng, cfg)
        instance_path = val_path / f'{PROB_PREFIX}_{cfg.seed}_{cfg.size}_{i}'
        write_to_file_stidsen(instance_path, instance_data)

    # Create test instances
    test_path = base_path / 'test'
    test_path.mkdir(parents=True, exist_ok=True)
    print(f"Creating test instances in: {test_path}")
    start = cfg.n_train + cfg.n_val
    end = start + cfg.n_test
    for i in range(start, end):
        instance_data = generate_instance_stidsen(rng, cfg)
        instance_path = test_path / f'{PROB_PREFIX}_{cfg.seed}_{cfg.size}_{i}'
        write_to_file_stidsen(instance_path, instance_data)


if __name__ == '__main__':
    main()

