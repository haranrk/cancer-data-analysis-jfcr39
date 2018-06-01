import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def clean_df(x: pd.DataFrame):
    y = x[x.notna().all(axis=1)]
    # y = x[x.notna().any(axis=1)]
    y = y.fillna(0)

    return y.as_matrix()


def rms(x: np.array, y: np.array):
    return np.mean(np.abs(x - y))


def heatmap(x, title, show_flag=1):
    plt.suptitle(title)
    plt.imshow(x, cmap="magma", interpolation="nearest")
    if show_flag == 1:
        plt.show()


def heatmap_dict(x: dict, title, show_flag=1):
    plt.suptitle(title)
    length = x.__len__()
    i = 1
    for key in x:
        plt.subplot(10 + length * 100 + i)
        plt.ylabel(key)
        plt.imshow(x[key], cmap="magma", interpolation="nearest")
        i += 1
    if show_flag == 1:
        plt.show()


def reorderConsensusMatrix(M):
    M = pd.DataFrame(M)
    Y = 1 - M
    Z = linkage(squareform(Y), method='average')
    ivl = leaves_list(Z)
    ivl = ivl[::-1]
    reorderM = pd.DataFrame(M.as_matrix()[:, ivl][ivl, :], index=M.columns[ivl], columns=M.columns[ivl])
    return reorderM.as_matrix()


def rc(M):
    x = M
    M = pd.DataFrame(M)
    Y = 1 - M
    yd = squareform(Y)
    Z = linkage(yd, method='average')
    ivl = leaves_list(Z)
    ivl = ivl[::-1]
    reorderM = pd.DataFrame(M.as_matrix()[:, ivl][ivl, :], index=M.columns[ivl], columns=M.columns[ivl])
    return reorderM.as_matrix()
#
# x = np.random.random((10,10))
# x = np.dot(x, x.T)
# x = x/x.max()
# x[x <= np.eye(10)] = 1
# y = rc(x)
