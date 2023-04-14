#!/bin/bash

#export py=/users/hert5217/anaconda3/envs/MLstuff/bin/python3
# export mpi=/usr/local/shared/openmpi/4.0.0/bin/mpiexec
export py=/users/hert5217/anaconda3/envs/torch2/bin/python3
# FCN batch

#export q=gpu24gb

addqueue -q gpu24gb -n 1x4 -m 12 -s $py train.py config/train_shakespeare_char.py
