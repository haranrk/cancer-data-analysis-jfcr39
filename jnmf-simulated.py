from lib.ToyMatrixClass import *
from lib.StandardJnmfClass import *
from lib.IntegrativeJnmfClass import *

# Mutable Variables
k_list = [2, 3, 5, 7, 4, 10]
lamb = [1, 0.1, 10]
save = 1
niter = 500
super_niter = 50

# Input
tm = ToyMatrix('m')
x=tm.x

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
    i+=1
    for k in k_list:
        im[k] = IntegrativeNmfClass(x, k, niter=niter, super_niter=super_niter, lamb=l)

        im[k].super_wrapper(verbose=0)

        print("INMF K = %i Error = %f" % (k, im[k].error))
        heatmap(im[k].cmh, "l=%i-INMF-h-k=%i" % (i, k), save=save)
        heatmap(im[k].cmw, "l=%i-INMF-w-k=%i" % (i, k), save=save)