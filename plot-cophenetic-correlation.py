from lib.IntegrativeJnmfClass import IntegrativeNmfClass
from lib.StandardJnmfClass import StandardNmfClass 
import lib.datasets as data
import seaborn as sns
from matplotlib import pyplot as plt

x = data.small()
super_iter = 50
niter = 4960
k_list = list(range(2, 7))
lamb = 1
thresh = 0
a = []
for k in k_list:
    m = StandardNmfClass(x, k, niter, super_iter, thresh=thresh)
    try:
        m.super_wrapper(verbose=0)
    finally:
        pass
    
    a.append(m.coph_corr_w)
    print("Error:%f" % m.error)
    plt.figure()
    plt.suptitle("Rank: %i" %k)
    plt.subplot(221)
    plt.title("DRUG")
    sns.heatmap(m.max_class_cm["DRUG"])
    plt.subplot(222)
    plt.title("MUT")
    sns.heatmap(m.max_class_cm["MUT"])
    plt.subplot(223)
    plt.title("PROT")
    sns.heatmap(m.max_class_cm["PROT"])
    plt.savefig("plots/standard_cmh_%i" % k)

# plt.figure()
# plt.plot(k_list, a)
# plt.savefig("plots/coph_corr")
