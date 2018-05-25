'''
Created on 2016/10/21

@author: taiho
'''
import numpy as np
import pandas as pd
import numpy.matlib
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


class ConsensusMatrix(object):
    '''
    Connectivity matrix (single JNMF run)
    Consensus matrix (multiple JNMF run)
    Reordring consensus matrix
    Cophenetic correlation
    
    '''


    def __init__(self, X1, X2, X3):
        '''
        Constructor
        '''
        self.cmW = pd.DataFrame(np.zeros((X1.shape[0], X1.shape[0])), index = X1.index, columns = X1.index)
        self.cmH1 = pd.DataFrame(np.zeros((X1.shape[1], X1.shape[1])), index = X1.columns, columns = X1.columns)
        self.cmH2 = pd.DataFrame(np.zeros((X2.shape[1], X2.shape[1])), index = X2.columns, columns = X2.columns)
        self.cmH3 = pd.DataFrame(np.zeros((X3.shape[1], X3.shape[1])), index = X3.columns, columns = X3.columns)
        sizePanH = X1.shape[1] + X2.shape[1] + X3.shape[1]
        columnsPanH = ["X1_" + x for x in X1.columns] + ["X2_" + x for x in X2.columns] + ["X3_" + x for x in X3.columns]
        self.cmPanH = pd.DataFrame(np.zeros((sizePanH, sizePanH)), index = columnsPanH, columns = columnsPanH)
        self.cmNum = 0

    def calcConnectivityW(self, W):
        maxW = W.as_matrix().max(axis=1)
        maxW[maxW == 0] = 1
        argmaxW = W.as_matrix().argmax(axis=1)
        maxMatW = (np.tile(maxW, (W.shape[1], 1))).transpose()
        binaryW = W == maxMatW
        connMatW = np.dot(binaryW, binaryW.transpose())
        return connMatW
    
    def calcConnectivityH(self, H):
        maxH = H.as_matrix().max(axis=0)
        maxH[maxH == 0] = 1
        argmaxH = H.as_matrix().argmax(axis=0)
        maxMatH = np.tile(maxH, (H.shape[0], 1))
        binaryH = H == maxMatH
        connMatH = np.dot(binaryH.transpose(), binaryH)
        return connMatH
        
    def addConnectivityMatrixtoConsensusMatrix(self, connW, connH1, connH2, connH3, connPanH):
        self.cmW += connW
        self.cmH1 += connH1
        self.cmH2 += connH2
        self.cmH3 += connH3
        self.cmPanH += connPanH
        self.cmNum += 1
        
    def finalizeConsensusMatrix(self):
        self.cmW /= self.cmNum
        self.cmH1 /= self.cmNum
        self.cmH2 /= self.cmNum
        self.cmH3 /= self.cmNum
        self.cmPanH /= self.cmNum
        np.fill_diagonal(self.cmW.as_matrix(), 1)
        np.fill_diagonal(self.cmH1.as_matrix(), 1)
        np.fill_diagonal(self.cmH2.as_matrix(), 1)
        np.fill_diagonal(self.cmH3.as_matrix(), 1)
        np.fill_diagonal(self.cmPanH.as_matrix(), 1)

    def reorderConsensusMatrix(self, M):
        Y = 1 - M
        Z = linkage(squareform(Y), method='average')
        ivl = leaves_list(Z)
        ivl = ivl[::-1]
        reorderM = pd.DataFrame(M.as_matrix()[:, ivl][ivl, :], index = M.columns[ivl], columns = M.columns[ivl])
        return reorderM
        