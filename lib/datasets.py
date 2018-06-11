from lib.functions import *


def small():
    x = {}
    s_file_list = ["DRUG",
                   "MUT",
                   "PROT",
                   ]
    for file in s_file_list:
        a = pth(os.getcwd()) / pth("data/small/%s.csv" % file)
        x[file] = pd.read_csv(a, index_col=0, header=0)

        print("Imported %s with shape %s" % (file, str(x[file].shape)))

    return x


# TODO - Implement it fully
def large():
    y = {}
    file_list = ["Drug",
                 "Gene",
                 "Tumor",
                 ]
    for file in file_list:
        a = pth(os.getcwd()) / pth("data/%s.csv" % file)

        y[file] = pd.read_csv(a, index_col=0, header=0)

        print("Imported %s" % file, y[file].shape)
        y[file] = clean_df(y[file])
        print("Cleaned %s" % file, y[file].shape)

    min_key = file_list[0] if y[file_list[0]].shape[0] < y[file_list[1]].shape[0] else file_list[1] if \
        y[file_list[1]].shape[0] < y[file_list[2]].shape[0] else file_list[2]
    for file in file_list:
        y[file] = y[file][y[file].index == y[min_key].index]
    return y
