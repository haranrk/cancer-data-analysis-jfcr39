"""
Created on 2016/10/19

@author: taiho
"""
import numpy as np
import pandas as pd


class JointNMF_mask(object):
    '''
    Joint NMF
    '''

    def __init__(self, X1, X2, X3, maskX1, maskX2, maskX3, rank, maxiter):
        '''
        Constructor
        '''
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.maskX1 = maskX1
        self.maskX2 = maskX2
        self.maskX3 = maskX3
        self.rank = rank
        self.maxiter = maxiter

    def check_nonnegativity(self):
        if self.X1.min().min() < 0 or self.X2.min().min() < 0 or self.X3.min().min() < 0:
            raise Exception('non negativity')

    def check_samplesize(self):
        if self.X1.shape[0] != self.X2.shape[0] or self.X1.shape[0] != self.X3.shape[0]:
            raise Exception('sample size')

    def initialize_W_H(self):
        self.W = pd.DataFrame(np.random.rand(self.X1.shape[0], self.rank), index=self.X1.index,
                              columns=map(str, range(1, self.rank + 1)))
        self.H1 = pd.DataFrame(np.random.rand(self.rank, self.X1.shape[1]), index=map(str, range(1, self.rank + 1)),
                               columns=self.X1.columns)
        self.H2 = pd.DataFrame(np.random.rand(self.rank, self.X2.shape[1]), index=map(str, range(1, self.rank + 1)),
                               columns=self.X2.columns)
        self.H3 = pd.DataFrame(np.random.rand(self.rank, self.X3.shape[1]), index=map(str, range(1, self.rank + 1)),
                               columns=self.X3.columns)
        self.X1r_pre = np.dot(self.W, self.H1)
        self.X2r_pre = np.dot(self.W, self.H2)
        self.X3r_pre = np.dot(self.W, self.H3)
        self.eps = np.finfo(self.W.as_matrix().dtype).eps

    def calc_euclidean_multiplicative_update(self):
        self.H1 = np.multiply(self.H1, np.divide(np.dot(self.W.T, np.multiply(self.maskX1, self.X1)), np.dot(self.W.T,
                                                                                                             np.multiply(
                                                                                                                 self.maskX1,
                                                                                                                 np.dot(
                                                                                                                     self.W,
                                                                                                                     self.H1) + self.eps))))
        self.H2 = np.multiply(self.H2, np.divide(np.dot(self.W.T, np.multiply(self.maskX2, self.X2)), np.dot(self.W.T,
                                                                                                             np.multiply(
                                                                                                                 self.maskX2,
                                                                                                                 np.dot(
                                                                                                                     self.W,
                                                                                                                     self.H2) + self.eps))))
        self.H3 = np.multiply(self.H3, np.divide(np.dot(self.W.T, np.multiply(self.maskX3, self.X3)), np.dot(self.W.T,
                                                                                                             np.multiply(
                                                                                                                 self.maskX3,
                                                                                                                 np.dot(
                                                                                                                     self.W,
                                                                                                                     self.H3) + self.eps))))
        self.W = np.multiply(self.W, np.divide(
            np.dot(np.multiply(np.c_[self.maskX1, self.maskX2, self.maskX3], np.c_[self.X1, self.X2, self.X3]),
                   np.transpose(np.c_[self.H1, self.H2, self.H3])), (np.dot(
                np.multiply(np.c_[self.maskX1, self.maskX2, self.maskX3],
                            np.dot(self.W, np.c_[self.H1, self.H2, self.H3])),
                np.transpose(np.c_[self.H1, self.H2, self.H3])) + self.eps)))

    def wrapper_calc_euclidean_multiplicative_update(self):
        for run in range(self.maxiter):
            self.calc_euclidean_multiplicative_update()
            self.calc_distance_of_HW_to_X()
            # self.print_distance_of_HW_to_X(run)

    def calc_distance_of_HW_to_X(self):
        self.X1r = np.dot(self.W, self.H1)
        self.X2r = np.dot(self.W, self.H2)
        self.X3r = np.dot(self.W, self.H3)
        self.diff = np.sum(np.sum(np.abs(self.X1r_pre - self.X1r))) + np.sum(
            np.sum(np.abs(self.X2r_pre - self.X2r))) + np.sum(np.sum(np.abs(self.X2r_pre - self.X2r)))
        self.X1r_pre = self.X1r
        self.X2r_pre = self.X2r
        self.X3r_pre = self.X3r
        self.eucl_dist1 = self.calc_euclidean_dist(self.X1, self.X1r)
        self.eucl_dist2 = self.calc_euclidean_dist(self.X2, self.X2r)
        self.eucl_dist3 = self.calc_euclidean_dist(self.X3, self.X3r)
        self.eucl_dist = self.eucl_dist1 + self.eucl_dist2 + self.eucl_dist3
        self.error1 = np.mean(np.mean(np.abs(self.X1 - self.X1r))) / np.mean(np.mean(self.X1))
        self.error2 = np.mean(np.mean(np.abs(self.X2 - self.X2r))) / np.mean(np.mean(self.X2))
        self.error3 = np.mean(np.mean(np.abs(self.X3 - self.X3r))) / np.mean(np.mean(self.X3))
        self.error = self.error1 + self.error2 + self.error3

    def print_distance_of_HW_to_X(self, text):
        print("[%s] diff = %f, eucl_dist = %f, error = %f" % (text, self.diff, self.eucl_dist, self.error))

    def calc_euclidean_dist(self, X, Y):
        dist = np.sum(np.sum(np.power(X - Y, 2)))
        return dist

    def set_PanH(self):
        self.PanH = pd.concat([self.H1, self.H2, self.H3], axis=1)
        columnsPanH = ["X1_" + x for x in self.H1.columns] + ["X2_" + x for x in self.H2.columns] + ["X3_" + x for x in
                                                                                                     self.H3.columns]
        self.PanH.columns = columnsPanH

    #    def run(self):
