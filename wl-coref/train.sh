#!/bin/bash

#SBATCH --nodelist=n0100
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --mail-type=all
#SBATCH --mail-user=goya_jackie@live.nl
#SBATCH --error=batchlogs/slurm-%j.err
#SBATCH --output=batchlogs/slurm-%j.out

NAME=$1
SEED=$2


python run.py train xlm-roberta $NAME --lr=5e-4 --bertlr=3e-5 --seed=$SEED --epochs 20
