from lib.ToyMatrixClass import *
from lib.StandardJnmfClass import *
from lib.IntegrativeJnmfClass import *
import lib.datasets as data
# Mutable Variables
k_list = [3]
lamb = [1, 0.1, 10]
save = 0
niter = 100
super_niter = 50
thresh = 1.5
data_set = [
            "simulated",
            "large",
            "small"
            ]

# Input
x={}
tm = ToyMatrix('m')
x[data_set[0]] = tm.x
x[data_set[2]] = data.small()

data_set = [
            # "simulated",
            # "large",
            "small"
            ]

# Program
print("Super niter: %i Niter: %i" % (super_niter, niter))
k_list.sort()
for d in data_set:
    print("\tDataset: %s" % d)

    sm = {}

    print("\t\tStandard NMF")
    for k in k_list:
        sm[k] = StandardNmfClass(x[d], k, niter=niter, super_niter=super_niter, thresh=thresh)

        sm[k].super_wrapper(verbose=0)

        print("\t\t\tSNMF K = %i Error = %f" % (k, sm[k].error))
        heatmap(sm[k].cmh, "SNMF-h-k=%i" % k, folder=d, save=save)
        heatmap(sm[k].cmw, "SNMF-w-k=%i" % k, folder=d, save=save)

    # im = {}
    # i=1
    # print("\tIntegrative NMF")
    # for l in lamb:
    #     print("\t\tLambda(L)%i = %f" % (i, l))
    #     i+=1
    #     for k in k_list:
    #         im[k] = IntegrativeNmfClass(x[d], k, niter=niter, super_niter=super_niter, lamb=l)
    #
    #         im[k].super_wrapper(verbose=0)
    #
    #         print("\t\t\tINMF K = %i Error = %f" % (k, im[k].error))
    #         heatmap(im[k].cmh, "L%i-INMF-h-k=%i" % (i, k), folder=d, save=save)
    #         heatmap(im[k].cmw, "L%i-INMF-w-k=%i" % (i, k), folder=d, save=save)