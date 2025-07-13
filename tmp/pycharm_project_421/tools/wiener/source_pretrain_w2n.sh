#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=source_pretrain_w2n
#SBATCH --time=200:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH -o /scratch/itee/uqzche24/TTA_3D_DET/tools/wiener/out/source_pretrain_w2n.txt
#SBATCH -e /scratch/itee/uqzche24/TTA_3D_DET/tools/wiener/err/source_pretrain_w2n.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6


#export Boost_INCLUDE_DIR:PATH="/scratch/itee/uqzche24/environment/ST3D/include"
#export CMAKE_CXX_FLAGS:STRING="/scratch/itee/uqzche24/environment/ST3D/include"

module load gnu7/7.3.0
#module load boost/1.75.0

module load cuda/11.3.0

conda activate /scratch/itee/uqzche24/environment/ST3D
module load git-2.19.1-gcc-7.3.0-swjt5hp

#/scratch/itee/uqzche24/environment/ST3D/bin/python train.py --cfg_file cfgs/tta_w2n_models/secondiou/source_pretrain.yaml --batch_size 6
#/scratch/itee/uqzche24/environment/ST3D/bin/python setup.py bdist_wheel
/scratch/itee/uqzche24/environment/ST3D/bin/python test.py  --cfg_file cfgs/tta_w2n_models/secondiou/source_pretrain.yaml --batch_size 16 --eval_all