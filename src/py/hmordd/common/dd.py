import signal

from hmordd.common.utils import CONST


class DDManager:
    def __init__(self, cfg):        
        self.cfg = cfg
        self.env = None
        self.dd = None
        self.layer = None
        # DD stats
        self.orig_size = None
        self.orig_width = None
        self.reduced_size = None
        self.reduced_width = None        
        # Result
        self.frontier = None
        self.nnds = None
        # Metrics
        ## Frontier
        self.hv_approx = None
        self.cardinality = None
        self.cardinality_raw = None
        self.precision = None
        ## Time
        self.time_build = None
        self.time_reduce = None
        self.time_frontier = None
        
    def reset(self, *args, **kwargs):
        raise NotImplementedError("Reset method must be implemented in subclasses.")
    
    def build_dd(self, *args, **kwargs):
        raise NotImplementedError("Build method must be implemented in subclasses.")
    
    def reduce_dd(self, *args, **kwargs):
        raise NotImplementedError("Reduce method must be implemented in subclasses.")

    def compute_frontier(self, time_limit=1800):
        try:
            signal.alarm(time_limit)
            self.env.compute_pareto_frontier()
            self.frontier = self.env.get_frontier()["z"]
            self.time_frontier = self.env.get_time(CONST.TIME_PARETO)
        except:
            self.frontier = None
            self.time_frontier = time_limit
        signal.alarm(0)

    def get_decision_diagram(self):
        return None
