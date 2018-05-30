from lib.ToyMatrixClass import *
from lib.functions import *
from lib.JointNmfClass import *

tm = ToyMatrix('m')
k_list = [2, 3, 4, 6, 7, 8]

k_list.sort()

for k in k_list:
    m = JointNmfClass(tm.x, k, 1000, 1)
    m.super_wrapper(verbose=1)
    print("K = %i Error = %f" % (k, m.error))