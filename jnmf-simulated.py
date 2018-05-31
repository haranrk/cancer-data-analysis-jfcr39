from lib.ToyMatrixClass import *
from lib.functions import *
from lib.JointNmfClass import *

tm = ToyMatrix('m')
k_list = [3, 5, ]

k_list.sort()
m={}
for k in k_list:
    m[k] = JointNmfClass(tm.x, k, 250, 50)
    m[k].super_wrapper(verbose=1)
    print("K = %i Error = %f" % (k, m[k].error))
    # heatmap_dict(x, "x | k: %i" % k)
    heatmap_dict(m[k].cmh, "h | k: %i" % k)
    # heatmap_dict(m[k].z_score, "z score | k: %i" % k)
    heatmap(m[k].cmw, "w k: %i" % k)
    heatmap_dict(m[k].cmz, "cmz k: %i" % k)