from pathlib import Path as pth
import os
import sys


if sys.argv[1] in globals():
    print("Give an argument")
else:
    print("Creating script file")
    loc = sys.argv[1].split('.')[0]
    a = pth(os.getcwd())
    a = a / pth("run-%s.sh" % loc)
    a.touch()
    x = open("run-%s.sh" % loc, "w")
    x.write("echo $SHELL\n")
    x.write("#!/bin/sh\n")
    x.write("echo $SHELL\n")
    x.write("#$ -S /bin/sh # set shell in UGE\n")
    x.write("export LANG=en_US.UTF-8\n")
    x.write("module load python/3.6\n")
    x.write("python3 %s.py\n" % loc)
    x.close()
    print("Created script file run-%s.py" % loc)
