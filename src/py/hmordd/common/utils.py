import json
import multiprocessing as mp
import zipfile
from abc import ABC

import numpy as np


class CONST:
    TIME_COMPILE = 1
    TIME_PARETO = 2
    ONE_ARC = 1
    ZERO_ARC = -1
    RESTRICT = 1
    RELAX = 2


class MetricCalculator:
    def __init__(self, n_objs):
        self.n_objs = n_objs

    @staticmethod
    def _igd_chunked(
        ref,
        approx,
        chunk_ref=256,
        chunk_approx=None,
        max_mem_mb=128,
    ):
        if ref.size == 0 or approx.size == 0:
            return np.nan

        total = 0.0
        n_ref = ref.shape[0]
        n_obj = ref.shape[1]

        if chunk_approx is None:
            bytes_per_entry = 8 * n_obj
            chunk_approx = max(
                1, int((max_mem_mb * 1024 * 1024) / (chunk_ref * bytes_per_entry))
            )
            chunk_approx = min(chunk_approx, approx.shape[0])

        for i in range(0, n_ref, chunk_ref):
            ref_block = ref[i : i + chunk_ref]
            min_d2 = np.full(ref_block.shape[0], np.inf, dtype=np.float64)
            for j in range(0, approx.shape[0], chunk_approx):
                approx_block = approx[j : j + chunk_approx]
                diff = ref_block[:, None, :] - approx_block[None, :, :]
                d2 = np.sum(diff * diff, axis=2)
                min_d2 = np.minimum(min_d2, d2.min(axis=1))
            total += np.sqrt(min_d2).sum()

        return float(total / n_ref)

    @staticmethod
    def _compute_igd_value(ref_pf, approx_pf):
        if ref_pf.size == 0 or approx_pf.size == 0:
            return np.nan

        ref = ref_pf.astype(np.float64, copy=False)
        approx = approx_pf.astype(np.float64, copy=False)

        # cKDTree is usually the most memory-efficient path.
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(approx)
            try:
                dists, _ = tree.query(ref, k=1, workers=-1)
            except TypeError:
                dists, _ = tree.query(ref, k=1)
            return float(np.mean(dists))
        except Exception:
            pass

        try:
            return MetricCalculator._igd_chunked(ref, approx)
        except Exception:
            pass

        try:
            from pymoo.indicators.igd import IGD

            return float(IGD(ref)(approx))
        except Exception:
            return np.nan

    @staticmethod
    def _count_points(frontier):
        if frontier is None:
            return -1
        arr = np.asarray(frontier)
        if arr.size == 0:
            return 0
        if arr.ndim >= 1:
            return int(arr.shape[0])
        return -1

    @staticmethod
    def compute_cardinality(true_pf=None, approx_pf=None):
        result = {'cardinality': -1, 'cardinality_raw': -1, 'precision': -1,
                  'n_exact_pf': -1, 'n_approx_pf': -1}
        
        if true_pf is not None and true_pf.size > 0:
            result['n_exact_pf'] = true_pf.shape[0]
            
        if approx_pf is not None and approx_pf.size > 0:
            result['n_approx_pf'] = approx_pf.shape[0]
        
        if result['n_exact_pf'] <= 0:
            print("True PF not available!")
            return result
        if result['n_approx_pf'] <= 0:
            print("Predicted PF not available!")
            return result
            
        true_pf, approx_pf = np.array(true_pf).astype(np.int64), \
            np.array(approx_pf).astype(np.int64)

        assert true_pf.ndim == approx_pf.ndim == 2
        assert true_pf.shape[1] == approx_pf.shape[1]

        # Defining a data type
        n_objs = true_pf.shape[1]
        dtype = {'names': [f'f{i}' for i in range(n_objs)],
                 'formats': [true_pf.dtype] * n_objs}                
        # Finding intersection
        found_ndps = np.intersect1d(true_pf.view(dtype), approx_pf.view(dtype))
        
        result['cardinality'] = found_ndps.shape[0] / true_pf.shape[0]
        result['cardinality_raw'] = found_ndps.shape[0]
        result['precision'] = found_ndps.shape[0] / approx_pf.shape[0]
        
        return result

    @staticmethod
    def compute_igd(true_pf=None, approx_pf=None):
        result = {
            "igd": None,
            "igd_raw": None,
            "n_exact_pf": -1,
            "n_approx_pf": -1,
        }

        result["n_exact_pf"] = MetricCalculator._count_points(true_pf)
        result["n_approx_pf"] = MetricCalculator._count_points(approx_pf)

        if true_pf is None or approx_pf is None:
            return result

        true_arr = np.asarray(true_pf)
        approx_arr = np.asarray(approx_pf)

        if true_arr.size == 0 or approx_arr.size == 0:
            return result
        if true_arr.ndim != 2 or approx_arr.ndim != 2:
            return result
        if true_arr.shape[1] != approx_arr.shape[1]:
            return result

        ref = true_arr.astype(np.float64, copy=False)
        approx = approx_arr.astype(np.float64, copy=False)

        igd_raw = MetricCalculator._compute_igd_value(ref, approx)

        min_ref = np.min(ref, axis=0)
        max_ref = np.max(ref, axis=0)
        range_ref = max_ref - min_ref
        range_ref = np.where(range_ref == 0.0, 1.0, range_ref)

        ref_norm = (ref - min_ref) / range_ref
        approx_norm = (approx - min_ref) / range_ref
        igd_norm = MetricCalculator._compute_igd_value(ref_norm, approx_norm)

        result["igd_raw"] = float(igd_raw) if np.isfinite(igd_raw) else None
        result["igd"] = float(igd_norm) if np.isfinite(igd_norm) else None

        return result

    @staticmethod
    def compute(true_pf=None, approx_pf=None):
        cardinality_result = MetricCalculator.compute_cardinality(
            true_pf=true_pf,
            approx_pf=approx_pf,
        )
        igd_result = MetricCalculator.compute_igd(
            true_pf=true_pf,
            approx_pf=approx_pf,
        )

        result = dict(cardinality_result)
        for key, value in igd_result.items():
            if key not in result:
                result[key] = value
        return result


class Baseline(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.hv_approx = None
        self.cardinality = None
        self.cardinality_raw = None
        self.precision = None
        self.times = None
        self.problem = None
        self.inst_data = None
        self.reset()
        self.metric_calculator = MetricCalculator(self.cfg.prob.n_objs)

    def reset(self):
        self.hv_approx = []
        self.cardinality = []
        self.precision = []
        self.times = []
        self.problem = None
        self.inst_data = None

    def save_final_result(self):
        raise NotImplementedError

    def worker(self, pid):
        raise NotImplementedError

    def run(self):
        if self.cfg.n_processes == 1:
            self.worker(0)
        else:
            pool = mp.Pool(processes=self.cfg.n_processes)
            results = []

            for rank in range(self.cfg.n_processes):
                results.append(pool.apply_async(self.worker, args=(rank,)))

            for r in results:
                r.get()

                
def compute_dd_size(dd):
    # Placeholder for actual DD size computation
    return 0


def compute_dd_width(dd):
    # Placeholder for actual DD width computation
    return 0


def get_static_order(prob_name, order_type, data):
    # Placeholder for getting static order
    return []


def handle_timeout(signum, frame):
    raise Exception("Timeout!")


def append_pf_dom_path(
    path,
    cfg,
    include_dominance=True,
    include_track_x=False,
    include_order_type=False,
):
    prob_cfg = getattr(cfg, "prob", None)
    pf_enum_method = getattr(prob_cfg, "pf_enum_method", None)
    dominance = None
    track_x = None
    order_type = None
    if include_dominance:
        dominance = getattr(prob_cfg, "dominance", None)
    if include_track_x:
        track_x = getattr(prob_cfg, "track_x", None)
    if include_order_type:
        order_type = getattr(prob_cfg, "order_type", None)
    if (
        pf_enum_method is None
        and dominance is None
        and track_x is None
        and order_type is None
    ):
        return path
    suffix_parts = []
    if pf_enum_method is not None:
        suffix_parts.append(f"pf-{pf_enum_method}")
    if dominance is not None:
        dom_value = int(dominance) if isinstance(dominance, bool) else dominance
        suffix_parts.append(f"dom-{dom_value}")
    if track_x is not None:
        track_x_value = int(track_x) if isinstance(track_x, bool) else track_x
        suffix_parts.append(f"trackx-{track_x_value}")
    if order_type is not None:
        suffix_parts.append(f"order-{order_type}")
    return path / "-".join(suffix_parts)


def read_from_zip(archive_path, file_path_in_zip, format="json"):
    try:
        with zipfile.ZipFile(archive_path, 'r') as zf:
            with zf.open(file_path_in_zip) as f:
                if format == "json":
                    return json.load(f)
                else:
                    return f.read()
    except Exception as e:
        print(f"Error reading from zip: {e}")
        return None
