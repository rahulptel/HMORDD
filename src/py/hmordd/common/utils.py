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

