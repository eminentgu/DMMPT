#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --qos a800
#SBATCH --partition a800
#SBATCH --output=%j.out
#SBATCH --error=%j.err
export PATH="/home/xiangg2021/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/home/xiangg2021/cuda-12.2/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="/home/xiangg2021/cuda-12.2/lib64:$LIBRARY_PATH"
export CPATH="/home/xiangg2021/cuda-12.2/targets/x86_64-linux/include:$CPATH"
export TORCH_CUDA_ARCH_LIST=8.0
./scripts/inference.sh