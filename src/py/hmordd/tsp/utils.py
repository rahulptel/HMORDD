
"""Utility helpers for TSP instances."""

import io
import zipfile

import numpy as np
import torch
from hmordd import Paths
from hmordd.tsp import PROB_NAME, PROB_PREFIX


def get_env(n_objs):
    try:
        lib = __import__("libtspenvo" + str(n_objs))
        return lib.TSPEnv()
    except:
        raise ImportError(f"Could not import library for {n_objs} objectives.")


def get_instance_path(size, split, seed, pid):
    filename = f"{PROB_PREFIX}_{seed}_{size}_{pid}.npz"
    return Paths.instances / PROB_NAME / size / split / filename


def get_instance_data(size, split, seed, pid):
    file_path = get_instance_path(size, split, seed, pid)
    if file_path.exists():
        return _load_npz(file_path)

    archive_path = Paths.instances / PROB_NAME / f"{size}.zip"
    if archive_path.exists():
        member = f"{size}/{split}/{file_path.name}"
        with zipfile.ZipFile(archive_path, "r") as zf:
            with zf.open(member) as fh:
                buffer = io.BytesIO(fh.read())
        return _load_npz(buffer)

    raise FileNotFoundError(f"Instance {pid} for size {size} and split {split} not found.")


def _load_npz(source):
    with np.load(source) as data:
        coords = data["coords"]
        dists = data["dists"]
    return {
        "coords": coords,
        "dists": dists,
        "n_objs": int(coords.shape[0]),
        "n_vars": int(coords.shape[1]),
    }


def compute_stat_features(dists):
    """Compute per-node distance statistics."""

    return torch.cat(
        (
            dists.max(dim=-1, keepdim=True)[0],
            dists.min(dim=-1, keepdim=True)[0],
            dists.std(dim=-1, keepdim=True),
            dists.median(dim=-1, keepdim=True)[0],
            dists.quantile(0.75, dim=-1, keepdim=True)
            - dists.quantile(0.25, dim=-1, keepdim=True),
        ),
        dim=-1,
    )


def get_model_str(cfg):
    model_str = f"{cfg.type}-v{cfg.version}"
    if cfg.d_emb != 32:
        model_str += f"-emb-{cfg.d_emb}"
    if cfg.n_layers != 2:
        model_str += f"-l-{cfg.n_layers}"
    if cfg.n_heads != 8:
        model_str += f"-h-{cfg.n_heads}"
    if cfg.act != "relu":
        model_str += f"-act-{cfg.act}"
    if cfg.concat_emb:
        model_str += "-cemb"
    if cfg.dropout_token != 0.0:
        model_str += f"-dptk-{cfg.dropout_token}"
    if cfg.dropout_attn != 0.0:
        model_str += f"-dpa-{cfg.dropout_attn}"
    if cfg.dropout_proj != 0.0:
        model_str += f"-dpp-{cfg.dropout_proj}"
    if cfg.dropout_mlp != 0.0:
        model_str += f"-dpm-{cfg.dropout_mlp}"
    if cfg.bias_mha:
        model_str += f"-ba-{cfg.bias_mha}"
    if cfg.bias_mlp:
        model_str += f"-bm-{cfg.bias_mlp}"
    if cfg.h2i_ratio != 2:
        model_str += f"-h2i-{cfg.h2i_ratio}"
    return model_str


def get_optimizer_str(cfg):
    opt_str = f"opt-{cfg.type}-lr-{cfg.lr}"
    if cfg.warmup > 0:
        opt_str += f"-wrm-{cfg.warmup}"
    if cfg.decay_lr:
        opt_str += "-dlr"
    if cfg.wd > 0:
        opt_str += f"-wd-{cfg.wd}"
    if cfg.beta1 != 0.9:
        opt_str += f"-b1-{cfg.beta1}"
    if cfg.beta2 != 0.999:
        opt_str += f"-b2-{cfg.beta2}"
    return opt_str


def get_exp_str(cfg):
    exp_str = f"bs-{cfg.batch_size}"
    if cfg.weighted_loss:
        exp_str += "-wl"
    exp_str += f"-gcl-{cfg.grad_clip}"
    exp_str += f"-nitr-{cfg.n_insts.train}"
    exp_str += f"-nivl-{cfg.n_insts.val}"
    exp_str += f"-pr-{cfg.pos_ratio}"
    exp_str += f"-npr-{cfg.neg_to_pos_ratio}"
    exp_str += f"-rs-{str(cfg.resample)}"
    exp_str += f"-sst-{str(cfg.subsample.train)}"
    exp_str += f"-ssv-{str(cfg.subsample.val)}"
    return exp_str


__all__ = [
    "get_env",
    "get_instance_data",
    "get_instance_path",
    "compute_stat_features",
    "get_model_str",
    "get_optimizer_str",
    "get_exp_str",
]
