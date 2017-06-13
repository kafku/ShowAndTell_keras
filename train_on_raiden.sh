#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N train_ShowAndTell
#$ -e raiden_error.log
#$ -o raiden_print.log
#$ -jc nvcr-cuda8_g8.168h
# -adds l_hard h_vmem 50G
# -adds l_hard m_mem_free 80G


export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL

source ~/.bashrc
python train.py
