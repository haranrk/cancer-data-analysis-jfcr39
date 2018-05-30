import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def clean_df(x: pd.DataFrame):
    y = x[x.notna().all(axis=1)]
    # y = x[x.notna().any(axis=1)]
    y = y.fillna(0)

    return y


def rms(x: np.array, y: np.array):
    return np.mean(np.abs(x - y))


def heatmap(x, show_flag=1):
    plt.imshow(x, cmap="magma", interpolation="nearest")
    if show_flag == 1:
        plt.show()


def heatmap_dict(x: dict):
    length = x.__len__()
    i = 1
    for key in x:
        plt.subplot(10 + length * 100 + i)
        plt.title(key)
        heatmap(x[key], show_flag=0)
        i += 1
    plt.show()