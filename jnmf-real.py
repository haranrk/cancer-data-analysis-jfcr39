from lib.StandardJnmfClass import *
from lib.IntegrativeJnmfClass import *
import os
from pathlib import Path as pth
import pandas as pd

# Mutable Variables
k_list = [2, 3, 5, 7, 4, 10]
lamb = 1
save = 1
niter = 500
super_niter = 50

file_list = ["input_CCLE_binmat",
             "input_CCLE_drug_IC50_zero-one",
             "input_CCLE_linage_binmat_5data_modified",
             ]

s_file_list = ["DRUG",
               "MUT",
               "PROT",
               ]

# for file in file_list:
#     a = pth(os.getcwd()) / pth("data/%s.csv" % file)
#     x[file] = pd.read_csv(a, index_col=0, header=0)
#
#     print("Imported %s" % file, x[file].shape)
#     x[file] = clean_df(x[file])
#     print("Cleaned %s" % file, x[file].shape)

x = {}
for file in s_file_list:
    a = pth(os.getcwd()) / pth("data/small/%s.csv" % file)
    x[file] = pd.read_csv(a, index_col=0, header=0)

    print("Imported %s from %s" % (file, a), x[file].shape)
    x[file] = clean_df(x[file])
    print("Cleaned %s" % file, x[file].shape)

k_list.sort()
im = {}
sm = {}
for k in k_list:
    im[k] = IntegrativeNmfClass(x, k, niter=niter, super_niter=super_niter, lamb=lamb)
    sm[k] = StandardNmfClass(x, k, niter=niter, super_niter=super_niter)

    im[k].super_wrapper(verbose=0)
    sm[k].super_wrapper(verbose=0)

    print("INMF K = %i Error = %f" % (k, im[k].error))
    print("SNMF K = %i Error = %f" % (k, sm[k].error))
    # heatmap(x, "x | k: %i" % k)
    heatmap(im[k].cmh, "INMF-h-k=%i" % k, save=save)
    heatmap(sm[k].cmh, "SNMF-h-k=%i" % k, save=save)
    # heatmap(m[k].z_score, "z score | k: %i" % k)
    heatmap(im[k].cmw, "INMF-w-k=%i" % k, save=save)
    heatmap(sm[k].cmw, "SNMFw-k=%i" % k, save=save)
    # heatmap(m[k].cmz, "cmz k: %i" % k)
