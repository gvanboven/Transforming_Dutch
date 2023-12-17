#!/bin/bash

#SBATCH --nodelist=n0100
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --mail-type=all
#SBATCH --mail-user=goya_jackie@live.nl
#SBATCH --error=batchlogs/slurm-%j.err
#SBATCH --output=batchlogs/slurm-%j.out

MODELNAME=$1
NAME=$2

python run.py eval xlm-roberta $MODELNAME --data-split test --weights $NAME
