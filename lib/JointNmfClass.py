import numpy as np


class JointNmfClass:
    def __init__(self, x: dict, k: int, niter: int, super_niter: int):
        self.x = x
        self.k = k
        self.niter = niter
        self.super_niter = super_niter
        self.initialize_variables()
        self.eps = np.finfo(self.w.dtype).eps
        self.calc_error()

    def initialize_variables(self):
        number_of_samples = list(self.x.values())[0].shape[0]
        self.w = np.random.rand(number_of_samples, self.k)
        self.h = {}
        self.z_score = {}
        for key in self.x:
            self.h[key] = np.random.rand(self.k, self.x[key].shape[1])
            self.z_score[key] = np.zeros((self.k, self.x[key].shape[1]))

    def update_weights(self):
        w = self.w
        numerator = np.zeros(w.shape)
        denominator = np.zeros((w.shape[1], w.shape[1]))

        for key, value in self.x.items():
            numerator = numerator + np.dot(self.x[key], self.h[key].T)
            denominator = denominator + np.dot(self.h[key], self.h[key].T)
            self.h[key] = self.h[key] * np.dot(w.T, self.x[key]) / np.dot(np.dot(w.T, w), self.h[key])

        self.w = self.w * numerator / np.dot(w, denominator)
        self.calc_error()

    def wrapper_update(self, verbose=0):
        for i in range(1, self.niter):
            self.update_weights()
            if verbose == 1 and i % 39 == 0:
                print("\t\titer: %i | error: %f" % (i, self.error))

    def super_wrapper(self, verbose=0):
        for i in range(0, self.super_niter):
            self.initialize_variables()

            if verbose == 1 and i % self.super_niter == 0:
                self.wrapper_update(verbose=1)
                print("\tSuper iteration: %i Error: %f " % (i, self.error))
            else:
                self.wrapper_update(verbose=0)

    def calc_z_score(self):
        for key in self.h:
            self.z_score[key] = (self.x - np.mean(self.x, axis=1)) / np.std(self.x, axis=1)

    def calc_error(self):
        self.error = 0
        for key in self.x:
            self.error += np.mean(self.x[key] - np.dot(self.w, self.h[key]))
