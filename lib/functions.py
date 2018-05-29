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


def heatmap(x):
    plt.imshow(x, cmap="magma", interpolation="nearest")

