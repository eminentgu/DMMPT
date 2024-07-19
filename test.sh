#!/bin/bash
#SBATCH --job-name=multi_prompt
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --output=%j.out
#SBATCH --error=%j.err
export PATH="/home/xiangg2021/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/home/xiangg2021/cuda-12.2/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="/home/xiangg2021/cuda-12.2/lib64:$LIBRARY_PATH"
export CPATH="/home/xiangg2021/cuda-12.2/targets/x86_64-linux/include:$CPATH"
export TORCH_CUDA_ARCH_LIST=8.0
mprof run python test.py