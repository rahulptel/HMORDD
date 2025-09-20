import signal

import numpy as np
from hmordd.common.dd import DDManager
from hmordd.common.utils import handle_timeout
from hmordd.setpacking.utils import get_env


class SetPackingDDManager(DDManager):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def reset(self, inst, order=None):
        signal.signal(signal.SIGALRM, handle_timeout)

        self.env = get_env(self.cfg.prob.n_objs)
        self.env.reset()
        print(inst["n_vars"], inst["n_cons"], inst["n_objs"])
        self.env.set_inst(inst["n_vars"], 
                          inst["n_cons"], 
                          inst["n_objs"], 
                          inst["obj_coeffs"],
                          inst["cons_coeffs"])
        self.env.initialize_dd_constructor()
            
    def build_dd(self, *args, **kwargs):
        raise NotImplementedError("Build method must be implemented in subclasses.")
    
    def reduce_dd(self):
        pass
    
    def compute_frontier(self, pf_enum_method, time_limit=1800):
        try:
            signal.alarm(time_limit)
            self.env.compute_pareto_frontier(pf_enum_method)
            self.frontier = self.env.get_frontier()
            self.frontier = np.array(self.frontier).reshape(-1, self.cfg.prob.n_objs)
            self.time_frontier = self.env.get_pareto_time()
        except:
            self.frontier = None
            self.time_frontier = time_limit
        signal.alarm(0)
    
    
class SetPackingExactDDManager(SetPackingDDManager):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def build_dd(self):
        self.env.generate_exact_dd()
        self.time_build = self.env.get_compilation_time()

class SetPackingRestrictedDDManager(SetPackingDDManager):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def build_dd(self):
        self.env.generate_restricted_dd(self.cfg.dd.width, self.cfg.dd.nosh.rule)
        self.time_build = self.env.get_compilation_time()

class DDManagerFactory:
    _managers = {
        "exact": SetPackingExactDDManager,
        "restricted": SetPackingRestrictedDDManager,
    }

    @classmethod
    def create_dd_manager(cls, cfg):
        dd_type = cfg.dd.type
        manager_class = cls._managers.get(dd_type)
        
        if not manager_class:
            raise ValueError(f"Unknown dd_type: {dd_type}")
        
        return manager_class(cfg)
