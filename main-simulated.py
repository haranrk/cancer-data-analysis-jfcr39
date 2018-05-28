from lib.functions import *
from lib.NmfClass import *
from matplotlib import pyplot as plt
from lib.ToyMatrixClass import *

# x_ori = pd.read_csv('data/input_CCLE_drug_IC50_zero-one.csv', index_col=0, header=0)
# x = clean_df(x_ori).as_matrix()

tm = ToyMatrix(method='c')

m = NmfModel(tm.X, 11, 1000, 50)
print(m.error)
# print(m.h)

m.super_wrapper()
print(m.error)

heatmap(m.consensus_matrix_h)
heatmap(m.consensus_matrix_w)