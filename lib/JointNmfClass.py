from lib.functions import *


# Abstract Class - Do not instantiate this class

class JointNmfClass:
    def __init__(self, x: dict, k: int, niter: int, super_niter: int):
        self.x = x
        self.k = k
        self.niter = niter
        self.super_niter = super_niter
        self.cmz = None
        self.cmh = None
        self.cmw = None
        self.w = None
        self.h = None
        self.z_score = None
        self.error = float('inf')
        self.eps = np.finfo(list(x.values())[0].dtype).eps

    def initialize_variables(self):
        number_of_samples = list(self.x.values())[0].shape[0]
        self.cmw = np.zeros((number_of_samples, number_of_samples))

        self.cmh = {}
        self.z_score = {}
        self.cmz = {}
        for key in self.x:
            self.cmh[key] = np.zeros((self.x[key].shape[1], self.x[key].shape[1]))
            self.z_score[key] = np.zeros((self.k, self.x[key].shape[1]))
            self.cmz = self.cmh

    def wrapper_update(self, verbose=0):
        for i in range(1, self.niter):
            self.update_weights()
            if verbose == 1 and i % 39 == 0:
                print("\t\titer: %i | error: %f" % (i, self.error))

    def super_wrapper(self, verbose=0):
        self.initialize_variables()
        for i in range(0, self.super_niter):
            self.initialize_wh()
            if verbose == 1 and i % self.super_niter % 5 == 0:
                self.wrapper_update(verbose=1)
                print("\tSuper iteration: %i Error: %f " % (i, self.error))
            else:
                self.wrapper_update(verbose=0)

            self.cmw += self.connectivity_matrix_w()
            for key in self.h:
                self.cmh[key] += self.connectivity_matrix_h(key)

        self.cmw = reorderConsensusMatrix(self.cmw / self.super_niter)
        for key in self.h:
            self.cmh[key] = reorderConsensusMatrix(self.cmh[key] / self.super_niter)

        self.calc_z_score()

    # TODO - invalid value ocurred
    def calc_z_score(self):
        for key in self.h:
            self.z_score[key] = (self.x[key] - np.mean(self.x[key], axis=1).reshape((-1, 1))) / self.eps + np.std(
                self.x[key],
                axis=1).reshape(
                (-1, 1))
            self.cmz[key] = self.connectivity_matrix_h(key=key)

    def connectivity_matrix_h(self, key):
        max_tiled = np.tile(self.h[key].max(0), (self.h[key].shape[0], 1))
        max_index = np.zeros(self.h[key].shape)
        max_index[self.h[key] == max_tiled] = 1
        return np.dot(max_index.T, max_index)

    def connectivity_matrix_w(self):
        max_tiled = np.tile(self.w.max(1).reshape((-1, 1)), (1, self.w.shape[1]))
        max_index = np.zeros(self.w.shape)
        max_index[self.w == max_tiled] = 1
        return np.dot(max_index, max_index.T)

    def calc_error(self):
        self.error = 0
        for key in self.x:
            self.error += np.mean(np.abs(self.x[key] - np.dot(self.w, self.h[key])))

    def update_weights(self):
        raise NotImplementedError("Must override update_weights")

    def initialize_wh(self):
        raise NotImplementedError("Must override initialize_wh")
