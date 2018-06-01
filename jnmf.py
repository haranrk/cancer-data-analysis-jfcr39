from lib.ToyMatrixClass import *
from lib.StandardJnmfClass import *
from lib.IntegrativeJnmfClass import *

# Mutable Variables
k_list = [2, 3, 5, 7, 4, 10]
lamb = [1, 0.1, 10]
save = 1
niter = 250
super_niter = 50
data_set = [
            "simulated",
            "large",
            "small"
            ]

# Input
x={}
tm = ToyMatrix('m')
x[data_set[0]]=tm.x

x[data_set[1]] = {}
file_list = ["input_CCLE_binmat",
             "input_CCLE_drug_IC50_zero-one",
             "input_CCLE_linage_binmat_5data_modified",
             ]
for file in file_list:
    a = pth(os.getcwd()) / pth("data/%s.csv" % file)
    x[data_set[1]][file] = pd.read_csv(a, index_col=0, header=0)

    print("Imported %s" % file, x[data_set[1]][file].shape)
    x[data_set[1]][file] = clean_df(x[data_set[1]][file])
    print("Cleaned %s" % file, x[data_set[1]][file].shape)

x[data_set[2]] = {}
s_file_list = ["DRUG",
               "MUT",
               "PROT",
               ]
for file in s_file_list:
    a = pth(os.getcwd()) / pth("data/small/%s.csv" % file)
    x[data_set[2]][file] = pd.read_csv(a, index_col=0, header=0)

    print("Imported %s from %s" % (file, a), x[data_set[2]][file].shape)
    x[data_set[2]][file] = clean_df(x[data_set[2]][file])
    print("Cleaned %s" % file, x[data_set[2]][file].shape)

data_set = [
            "simulated",
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
        sm[k] = StandardNmfClass(x[d], k, niter=niter, super_niter=super_niter)

        sm[k].super_wrapper(verbose=0)

        print("\t\t\tSNMF K = %i Error = %f" % (k, sm[k].error))
        heatmap(sm[k].cmh, "SNMF-h-k=%i" % k, folder=d, save=save)
        heatmap(sm[k].cmw, "SNMF-w-k=%i" % k, folder=d, save=save)

    im = {}
    i=1
    print("\tIntegrative NMF")
    for l in lamb:
        print("\t\tLambda(L)%i = %f" % (i, l))
        i+=1
        for k in k_list:
            im[k] = IntegrativeNmfClass(x[d], k, niter=niter, super_niter=super_niter, lamb=l)

            im[k].super_wrapper(verbose=0)

            print("\t\t\tINMF K = %i Error = %f" % (k, im[k].error))
            heatmap(im[k].cmh, "L%i-INMF-h-k=%i" % (i, k), folder=d, save=save)
            heatmap(im[k].cmw, "L%i-INMF-w-k=%i" % (i, k), folder=d, save=save)