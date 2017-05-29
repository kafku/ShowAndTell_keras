#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N train_ShowAndTell
#$ -e raiden_error.log
#$ -o raiden_print.log
#$ -jc nvcr-cuda8_g8.168h

source ~/.bashrc
python train.py
