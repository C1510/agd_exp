#!/bin/bash

#export py=/users/hert5217/anaconda3/envs/MLstuff/bin/python3
# export mpi=/usr/local/shared/openmpi/4.0.0/bin/mpiexec
export py=/users/hert5217/anaconda3/envs/torch2/bin/python3
# FCN batch

#export q=gpu24gb

#addqueue -q gpu24gb -n 1x4 -m 7 -s $py main.py config/train_shakespeare_char.py
#addqueue -q gpu24gb -n 1x4 -m 7 -s /users/hert5217/anaconda3/envs/torch2/bin/torchrun --standalone --nproc_per_node=1 main.py config/train_gpt2.py
addqueue -q gpu24gb -n 1x4 -m 7 -s /users/hert5217/anaconda3/envs/torch2/bin/python3 main.py config/train_gpt2.py
#addqueue -q gpu24gb -n 1x4 -m 7 -s /users/hert5217/anaconda3/envs/torch2/bin/python3 main.py config/eval_gpt2.py
