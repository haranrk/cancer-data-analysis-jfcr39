from lib.ToyMatrixClass import *
from lib.StandardJnmfClass import *
from lib.IntegrativeJnmfClass import *

# Mutable Variables
k_list = [2, 3, 5, 7, 4, 10]
lamb = 1
save = 1
niter = 500
super_niter = 5

tm = ToyMatrix('m')


k_list.sort()
im = {}
sm = {}
for k in k_list:
    im[k] = IntegrativeNmfClass(tm.x, k, niter=niter, super_niter=super_niter, lamb=lamb)
    sm[k] = StandardNmfClass(tm.x, k, niter=niter, super_niter=super_niter)

    im[k].super_wrapper(verbose=0)
    sm[k].super_wrapper(verbose=0)

    print("INMF K = %i Error = %f" % (k, im[k].error))
    print("SNMF K = %i Error = %f" % (k, sm[k].error))
    heatmap(im[k].cmh, "INMF-h-k=%i" % k, save=save)
    heatmap(sm[k].cmh, "SNMF-h-k=%i" % k, save=save)
    heatmap(im[k].cmw, "INMF-w-k=%i" % k, save=save)
    heatmap(sm[k].cmw, "SNMFw-k=%i" % k, save=save)
