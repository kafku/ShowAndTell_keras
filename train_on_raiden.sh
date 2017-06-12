#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N train_ShowAndTell
#$ -e raiden_error.log
#$ -o raiden_print.log
#$ -adds l_hard h_vmem 50G
#$ -adds l_hard m_mem_free 50G
#$ -jc nvcr-cuda8_g8.168h

source ~/.bashrc
python train.py
