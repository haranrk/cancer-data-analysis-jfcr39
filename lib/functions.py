import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import os
from pathlib import Path as pth
import sys
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('Using non-interactive Agg backend')
    mpl.use('Agg')
from matplotlib import pyplot as plt


os.chdir(pth(__file__).parent.parent)
cwd = pth(os.getcwd())
print("Current working directory: %s" % cwd)


def clean_df(x: pd.DataFrame):
    y = x[x.notna().all(axis=1)]
    # y = x[x.notna().any(axis=1)]
    y = y.fillna(0)

    return y.as_matrix()


def rms(x: np.array, y: np.array):
    return np.mean(np.abs(x - y))


def heatmap(x, title, folder: str, show_flag=1, save=0):
    if str(x.__class__) == "<class 'numpy.ndarray'>":
        plt.suptitle(title)
        plt.imshow(x, cmap="magma", interpolation="nearest")

    elif str(x.__class__) == "<class 'dict'>":
        plt.suptitle(title)
        length = x.__len__()
        i = 1
        for key in x:
            plt.subplot(10 + length * 100 + i)
            plt.ylabel(key)
            plt.imshow(x[key], cmap="magma", interpolation="nearest")
            i += 1

    if save == 1:
        fig_path = cwd / pth("plots/%s-data/" % folder)
        fig_path.mkdir(exist_ok=True, parents=True)
        fig_path = fig_path / pth("%s.jpg" % title)
        plt.savefig(fig_path)
    elif show_flag == 1:
        plt.show()
#
# def heatmap_dict(x: dict, title, show_flag=1, save=0):
#     plt.suptitle(title)
#     length = x.__len__()
#     i = 1
#     for key in x:
#         plt.subplot(10 + length * 100 + i)
#         plt.ylabel(key)
#         plt.imshow(x[key], cmap="magma", interpolation="nearest")
#         i += 1
#
#     if save == 1:
#         fig_path = cwd / pth("plots/%s.jpg" % title)
#         plt.savefig(fig_path)
#     elif show_flag == 1:
#         plt.show()


def reorderConsensusMatrix(M: np.array):
    M = pd.DataFrame(M)
    Y = 1 - M
    Z = linkage(squareform(Y), method='average')
    ivl = leaves_list(Z)
    ivl = ivl[::-1]
    reorderM = pd.DataFrame(M.as_matrix()[:, ivl][ivl, :], index=M.columns[ivl], columns=M.columns[ivl])
    return reorderM.as_matrix()
