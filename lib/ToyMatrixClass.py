import numpy as np


class ToyMatrix:
    def __init__(self, method='a'):
        self.W = 0
        self.H = 0
        self.X = 0

        if method == 'a':
            self.toy_matrix_a()
        elif method == 'b':
            self.toy_matrix_b()
        elif method == 'c':
            self.toy_matrix_c()

    def toy_matrix_a(self, m=40, n=30, k=2, seed=0):
        np.random.seed(seed)
        self.W = np.random.random((n, k))
        self.H = np.ones((k, m))
        self.H[0, :10] = 0.5
        self.H[1, 10:] = 0.2
        self.X = np.dot(self.W, self.H)

    def toy_matrix_b(self, m=40, n=30, k=3, seed=0):
        np.random.seed(seed)
        self.W = np.random.random((n, k))
        self.H = np.zeros((k, m))
        self.H[0, :10] = 0.5
        self.H[1, 10:20] = 0.1
        self.H[2, 20:] = 0.3
        self.X = np.dot(self.W, self.H)

    def toy_matrix_c(self, m=100, n=40, k=7, seed=0):
        np.random.seed(seed)
        self.W = np.random.random((n, k))
        self.H = np.zeros((k, m))
        self.H[0, :10] = 0.5
        self.H[1, 10:20] = 0.1
        self.H[2, 20:25] = 0.3
        self.H[3, 25:45] = 0.452
        self.H[4, 45:60] = 0.8
        self.H[5, 60:67] = 0.59
        self.H[6, 67:] = 0.702
        np.random.shuffle(self.H)
        self.X = np.dot(self.W, self.H)

    def cluster(self):
        return np.argmax(self.H, 0)
