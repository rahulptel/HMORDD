import ctypes
import os

from hmordd import Paths

PROB_NAME = "setpacking"
PROB_PREFIX = "sp"

import sys


def get_env(n_objs: int):
    try:
        lib = __import__("libsetpackingenvo" + str(n_objs))
        return lib.SetpackingEnv()
    except:
        raise ImportError(f"Could not import library for {n_objs} objectives.")

def get_instance_data(prob_name, size, split, seed, pid):
    instance_path = Paths.instances / prob_name / size / split / f"{PROB_PREFIX}_{seed}_{size}_{pid}.dat"
    if not instance_path.exists():
        raise FileNotFoundError(f"Instance {pid} not found.")
    
    with open(instance_path, "r") as f:
        lines = f.readlines()

    n_vars, n_cons = map(int, lines[0].strip().split())
    n_objs = int(lines[1].strip())

    obj_coeffs = []
    for i in range(n_objs):
        obj_coeffs.append(list(map(int, lines[2+i].strip().split())))

    cons_coeffs = []
    line_idx = 2 + n_objs
    for i in range(n_cons):
        n_vars_in_con = int(lines[line_idx].strip())
        line_idx += 1
        cons_coeffs.append([c - 1 for c in map(int, lines[line_idx].strip().split())])
        line_idx += 1

    return {
        "n_vars": n_vars,
        "n_cons": n_cons,
        "n_objs": n_objs,
        "obj_coeffs": obj_coeffs,
        "cons_coeffs": cons_coeffs,
    }