import numpy as np

class JointNmfClass:
    def __init__(self, x: dict, k: int, niter: int, super_niter: int):
        self.x = x
        self.k = k
        self.niter = niter
        self.super_niter = super_niter
        self.initialize_variables()

    def initialize_variables(self):
        self.w = np.random.rand(self.x.shape[0], self.k)
        self.h = {}
        for key in self.x:
            self.h[key] = np.random.rand(self.k, self.x[key].shape[1])
    