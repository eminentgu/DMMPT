#!/bin/bash
#SBATCH --job-name=guardian
#SBATCH -n 1
#SBATCH --qos default
#SBATCH --output=%j.out
#SBATCH --error=%j.err
python watchdog.py