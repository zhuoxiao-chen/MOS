#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=setup_pcdet
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH -o out_setup.txt
#SBATCH -e err_setup.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8


#export Boost_INCLUDE_DIR:PATH="/scratch/itee/uqzche24/environment/ST3D/include"
#export CMAKE_CXX_FLAGS:STRING="/scratch/itee/uqzche24/environment/ST3D/include"

module load gnu7/7.3.0
#module load boost/1.75.0

module load cuda/11.3.0

conda activate /scratch/itee/uqzche24/environment/ST3D
module load git-2.19.1-gcc-7.3.0-swjt5hp



/scratch/itee/uqzche24/environment/ST3D/bin/python setup.py develop
#/scratch/itee/uqzche24/environment/ST3D/bin/python setup.py bdist_wheel