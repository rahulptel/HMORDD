import random as rng

import numpy as np


class NOSH:
    """Base class for Node Selection Heuristics (NOSH) """
    def __init__(self, width):
        self.width = width
        
    def __call__(self, layer):
        """
        - Returns the indices of the nodes to be removed
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
        
class NOSHRuleRandom(NOSH):
    def __init__(self, width, seed=47):
        super().__init__(width)
        self.rng = np.random.RandomState(seed)
        
    def __call__(self, n_layer):
        """
        - Assume the size of the layer is passed as an input.
        """
        idx_to_remove = []
        if self.width < n_layer:
            idx_to_remove = np.arange(n_layer)
            rng.shuffle(idx_to_remove)
            idx_to_remove = idx_to_remove[self.width:]
            
        return idx_to_remove
        
class NOSHRuleScaler(NOSH):
    def __init__(self, width, rank="max"):
        super().__init__(width)
        assert rank in ["max", "min"], "Rank must be 'max' or 'min'."
        self.reverse = rank == "max"

    def __call__(self, layer):
        """Assume layer is a list of floating point numbers."""
        idx_score = [(i, n) for i, n in enumerate(layer)]
        idx_score = sorted(idx_score, key=lambda x: x[1], reverse=self.reverse)
        return [i[0] for i in idx_score[self.width:]]


class NOSHRuleCardinality(NOSH):
    def __init__(self, width, rank="max"):
        super().__init__(width)
        assert rank in ["max", "min"], "Rank must be 'max' or 'min'."
        self.reverse = rank == "max"
        
    def __call__(self, layer):
        """ Assume layer is a list of lists."""
        idx_score = [(i, np.sum(n)) for i, n in enumerate(layer)]
        idx_score = sorted(idx_score, key=lambda x: x[1], reverse=self.reverse)
        return [i[0] for i in idx_score[self.width:]]


