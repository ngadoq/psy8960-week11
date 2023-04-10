#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=32gb
#SBATCH -t 00:20:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=do000100@umn.edu
#SBATCH -p amdsmall
cd ~/week11-cluster
module load R/4.2.2-openblas
Rscript week11-cluster.R

