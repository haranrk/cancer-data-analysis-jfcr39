echo $SHELL
#!/bin/sh
echo $SHELL
#$ -S /bin/sh # set shell in UGE
export LANG=en_US.UTF-8
module load python/3.6
python3 jnmf-real.py
