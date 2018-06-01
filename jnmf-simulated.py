from lib.ToyMatrixClass import *
from lib.functions import *
from lib.JointNmfClass import *
from lib.IntegrativeNmfClass import *

tm = ToyMatrix('m')
k_list = [3, 5, 7, 10]

k_list.sort()
m={}
heatmap_dict(tm.x, "x")
for k in k_list:
    m[k] = IntegrativeNmfClass(tm.x, k, 250, 50, 1)
    # m[k] = JointNmfClass(tm.x, k, 250, 50)
    m[k].super_wrapper(verbose=0)
    print("K = %i Error = %f" % (k, m[k].error))

    heatmap_dict(m[k].cmh, "h | k: %i" % k)
    # heatmap_dict(m[k].z_score, "z score | k: %i" % k)
    heatmap(m[k].cmw, "w k: %i" % k)
    heatmap_dict(m[k].cmz, "cmz k: %i" % k)