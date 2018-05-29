from lib.functions import *
from lib.NmfClass import *
from matplotlib import pyplot as plt
import nimfa as nf
import os
from pathlib import Path as pth

os.chdir(pth(__file__).parent)
# x_ori = pd.read_csv('data/input_CCLE_drug_IC50_zero-one.csv', index_col=0, header=0)
# x = clean_df(x_ori).as_matrix()
# k_list = [30, 25, 10, 3, 40,2,5,6,40,]

x = nf.examples.medulloblastoma.read(normalize=True)
data_name = "medulloblastoma"
k_list = list(range(2, 10))

k_list.sort()

for k in k_list:
    m = NmfModel(x, k, niter=100, super_niter=2)
    m.super_wrapper(verbose=1)
    print("K = %i Error = %f" % (k, m.error))
    plt.subplot(121)
    plt.suptitle("k = %s" % k)

    heatmap(m.consensus_matrix_h)
    plt.title("H")
    plt.subplot(122)

    heatmap(m.consensus_matrix_w)
    plt.title("W")
    plt.show()
    plt.savefig("plots/%s-k=%i.jpg" % (data_name, k))
