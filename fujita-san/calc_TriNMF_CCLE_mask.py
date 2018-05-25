'''
Created on 2016/10/18

@author: Naoya Fujita
'''

import sys
import os
from pathlib import Path as pth
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from matplotlib.pyplot import savefig, imshow, set_cmap

os.chdir(pth(__file__).parent)

from classJointNMF_mask import *
from classConsensusMatrix import *

K = 30
if len(sys.argv) > 1:
    K = int(sys.argv[1])
    print("K =  %d" % K)

"""
Read input data files: X1, X2, and X3
"""
X1ori = pd.read_csv('../input/input_CCLE_drug_IC50_zero-one.csv', header=0, index_col=0, na_values='NaN')
# df1.index # list of 661 cell lines
# df1.columns # list of IC50 and 124 features
# df1.values
# df1.describe() # summary statistics
# df1.T # transposition
X1 = X1ori[X1ori.notnull().any(axis=1)]  # 504 rows x 24 columns
# X1 = X1ori.dropna()    # 305 rows x 24 columns
# X1 = X1ori[X1ori.notnull().all(axis=1)]    # 305 rows x 24 columns
maskX1 = 1 - X1.isnull()
X1 = X1.fillna(0)

X2ori = pd.read_csv('../input/input_CCLE_binmat.csv', header=0, index_col=0, na_values='NaN')
X2 = X2ori[X1ori.notnull().any(axis=1)]  # 504 rows x 1639 columns
# X2 = X2ori[X1ori.notnull().all(axis=1)]    # 305 rows x 1639 columns
maskX2 = 1 - X2.isnull()
X2 = X2.fillna(0)

X3ori = pd.read_csv('../input/input_CCLE_linage_binmat_5data_modified.csv', header=0, index_col=0, na_values='NaN')
X3 = X3ori[X1ori.notnull().any(axis=1)]  # 504 rows x 24 columns
# X3 = X3ori[X1ori.notnull().all(axis=1)]    # 305 rows x 24 columns
maskX3 = 1 - X3.isnull()
X3 = X3.fillna(0)

"""
Remove samples that contain N/A values
"""
# XALLori = pd.concat([X1ori, X2ori, X3ori], axis=1)
# X1 = X1ori[XALLori.notnull().all(axis=1)]    # 269 rows x 24 columns
# X2 = X2ori[XALLori.notnull().all(axis=1)]    # 269 rows x 1639 columns
# X3 = X3ori[XALLori.notnull().all(axis=1)]    # 269 rows x 24 columns

"""
Set parameters
"""
nloop = 10
maxiter = 100

"""
Run Joint NMF
Calculate consensus matrix
"""
cmatrix = ConsensusMatrix(X1, X2, X3)
for i in range(50):
    jnmf = JointNMF_mask(X1, X2, X3, maskX1, maskX2, maskX3, K, maxiter)
    jnmf.check_nonnegativity()
    jnmf.check_samplesize()
    jnmf.initialize_W_H()
    # jnmf.update_euclidean_multiplicative()
    jnmf.wrapper_calc_euclidean_multiplicative_update()
    jnmf.print_distance_of_HW_to_X(i)
    jnmf.set_PanH()
    connW = cmatrix.calcConnectivityW(jnmf.W)
    connH1 = cmatrix.calcConnectivityH(jnmf.H1)
    connH2 = cmatrix.calcConnectivityH(jnmf.H2)
    connH3 = cmatrix.calcConnectivityH(jnmf.H3)
    connPanH = cmatrix.calcConnectivityH(jnmf.PanH)
    cmatrix.addConnectivityMatrixtoConsensusMatrix(connW, connH1, connH2, connH3, connPanH)

cmatrix.finalizeConsensusMatrix()

"""
Reorder consensus matrix.

:param C: Consensus matrix.
:type C: `numpy.ndarray`
"""
cmW = cmatrix.reorderConsensusMatrix(cmatrix.cmW)
cmH1 = cmatrix.reorderConsensusMatrix(cmatrix.cmH1)
cmH2 = cmatrix.reorderConsensusMatrix(cmatrix.cmH2)
cmH3 = cmatrix.reorderConsensusMatrix(cmatrix.cmH3)
cmPanH = cmatrix.reorderConsensusMatrix(cmatrix.cmPanH)

"""
Plot reordered consensus matrix.

:param C: Reordered consensus matrix.
:type C: numpy.ndarray`
:param rank: Factorization rank.
:type rank: `int`
"""
# imshow(cmW, cmap='Blues', interpolation="nearest")
# plt.colorbar()
# savefig('../output/jnmf_CCLE_ConsensusMatrixW_k%d.png' % K)

# imshow(cmPanH, cmap='Blues', interpolation="nearest")
# plt.colorbar()
# savefig('../output/jnmf_CCLE_ConsensusMatrixPanH_k%d.png' % K)


"""
Save output files: W, H1, H2, and H3
"""
W = jnmf.W
H1 = jnmf.H1
H2 = jnmf.H2
H3 = jnmf.H3
PanH = jnmf.PanH
W.to_csv('../output/jnmf_dataW_CCLE_k%d.csv' % K)
H1.to_csv('../output/jnmf_dataH1_CCLE_k%d.csv' % K)
H2.to_csv('../output/jnmf_dataH2_CCLE_k%d.csv' % K)
H3.to_csv('../output/jnmf_dataH3_CCLE_k%d.csv' % K)
PanH.to_csv('../output/jnmf_dataPanH_CCLE_k%d.csv' % K)

"""
Save output files: cmW
"""
cmW.to_csv('../output/jnmf_CCLE_ConsensusMatrixW_k%d.csv' % K)
cmPanH.to_csv('../output/jnmf_CCLE_ConsensusMatrixPanH_k%d.csv' % K)
