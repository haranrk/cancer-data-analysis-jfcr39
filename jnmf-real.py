from lib.ToyMatrixClass import *
from lib.functions import *
from lib.JointNmfClass import *
import os
from pathlib import Path as pth
import pandas as pd

os.chdir(pth(__file__).parent)

file_list = ["input_CCLE_binmat",
             "input_CCLE_drug_IC50_zero-one",
             "input_CCLE_linage_binmat_5data_modified",
             ]

s_file_list = ["DRUG",
               "MUT",
               "PROT",
               ]
x = {}
# for file in file_list:
#     a = pth(os.getcwd()) / pth("data/%s.csv" % file)
#     x[file] = pd.read_csv(a, index_col=0, header=0)
#
#     print("Imported %s" % file, x[file].shape)
#     x[file] = clean_df(x[file])
#     print("Cleaned %s" % file, x[file].shape)

for file in s_file_list:
    a = pth(os.getcwd()) / pth("data/small/%s.csv" % file)
    x[file] = pd.read_csv(a, index_col=0, header=0)

    print("Imported %s" % file, x[file].shape)
    x[file] = clean_df(x[file])
    print("Cleaned %s" % file, x[file].shape)

k_list = [2, 3, 5, 7, 4, 10]
k_list.sort()

m = {}
for k in k_list:
    m[k] = JointNmfClass(x, k, 500, 100)
    m[k].super_wrapper(verbose=0)
    print("K = %i Error = %f" % (k, m[k].error))
    # heatmap_dict(x, "x | k: %i" % k)
    heatmap_dict(m[k].cmh, "h | k: %i" % k)
    # heatmap_dict(m[k].z_score, "z score | k: %i" % k)
    heatmap(m[k].cmw, "w k: %i" % k)
    # heatmap_dict(m[k].cmz, "cmz k: %i" % k)
