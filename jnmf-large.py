from lib.StandardJnmfClass import *
from lib.IntegrativeJnmfClass import *
import os
from pathlib import Path as pth
import pandas as pd

# Mutable Variables
k_list = [2, 3, 5, 7, 4, 10]
lamb = [1, 0.1, 10]
save = 1
niter = 500
super_niter = 50
data_set = "s"

# Input
x = {}
file_list = ["input_CCLE_binmat",
             "input_CCLE_drug_IC50_zero-one",
             "input_CCLE_linage_binmat_5data_modified",
             ]
for file in file_list:
    a = pth(os.getcwd()) / pth("data/%s.csv" % file)
    x[file] = pd.read_csv(a, index_col=0, header=0)

    print("Imported %s" % file, x[file].shape)
    x[file] = clean_df(x[file])
    print("Cleaned %s" % file, x[file].shape)

# Program
k_list.sort()
sm = {}
print("Super niter: %i Niter: %i" % (super_niter, niter))
print("Standard NMF")
for k in k_list:
    sm[k] = StandardNmfClass(x, k, niter=niter, super_niter=super_niter)

    sm[k].super_wrapper(verbose=0)

    print("SNMF K = %i Error = %f" % (k, sm[k].error))
    heatmap(sm[k].cmh, "SNMF-h-k=%i" % k, save=save)
    heatmap(sm[k].cmw, "SNMF-w-k=%i" % k, save=save)

im = {}
i=1
print("Integrative NMF")
for l in lamb:
    print("Lambda%i = %f" % (i, l))
    for k in k_list:
        im[k] = IntegrativeNmfClass(x, k, niter=niter, super_niter=super_niter, lamb=l)

        im[k].super_wrapper(verbose=0)

        print("INMF K = %i Error = %f" % (k, im[k].error))
        heatmap(im[k].cmh, "l=%i-INMF-h-k=%i" % (i, k), save=save)
        heatmap(im[k].cmw, "l=%i-INMF-w-k=%i" % (i, k), save=save)
