import numpy as np
from lib.functions import *
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


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
        self.cmw = np.zeros((number_of_samples, number_of_samples))

        self.cmh = {}
        self.z_score = {}
        self.cmz = {}
        for key in self.x:
            self.cmh[key] = np.zeros((self.x[key].shape[1], self.x[key].shape[1]))
            self.z_score[key] = np.zeros((self.k, self.x[key].shape[1]))
            self.cmz = self.cmh
        self.initialize_wh()

    def initialize_wh(self):
        number_of_samples = list(self.x.values())[0].shape[0]
        self.w = np.random.rand(number_of_samples, self.k)
        self.h = {}
        for key in self.x:
            self.h[key] = np.random.rand(self.k, self.x[key].shape[1])

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

        self.cmw = self.reorderConsensusMatrix(self.cmw / self.super_niter)
        for key in self.h:
            self.cmh[key] = self.reorderConsensusMatrix(self.cmh[key] / self.super_niter)

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

    # TODO - Understand this func
    def reorderConsensusMatrix(self, M):
        M = pd.DataFrame(M)
        Y = 1 - M
        Z = linkage(squareform(Y), method='average')
        ivl = leaves_list(Z)
        ivl = ivl[::-1]
        reorderM = pd.DataFrame(M.as_matrix()[:, ivl][ivl, :], index=M.columns[ivl], columns=M.columns[ivl])
        return reorderM.as_matrix()
