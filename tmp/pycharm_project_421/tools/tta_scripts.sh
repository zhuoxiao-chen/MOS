#!/bin/sh
cfg_path='cfgs/tta_w2k_models/secondiou/source_pretrain_ped.yaml'
gpu_id=1
pretrained_model_path='../output/tta_w2k_models/secondiou/source_pretrain/default/ckpt/checkpoint_epoch_20.pth'

#python train.py --cfg_file $cfg_path --batch_size 8 --gpu_id $gpu_id --pretrained_model $pretrained_model_path && \
#python test_tta.py  --cfg_file $cfg_path --batch_size 8 --gpu_id $gpu_id --eval_all --set DATA_CONFIG_TAR.FOV_POINTS_ONLY True

# --set DATA_CONFIG_TAR.FOV_POINTS_ONLY True


# train source
#python train.py --cfg_file $cfg_path --batch_size 8 && \
#python test.py  --cfg_file $cfg_path --batch_size 16 --eval_all

# kitti-c --pretrained_model ../output/tta_kitti-c_models/secondiou/source_pretrain_tta/default/ckpt/checkpoint_epoch_55.pth
# w2k & w2kitti-c --pretrained_model
# ../output/tta_w2k_models/secondiou/source_pretrain/default/ckpt/checkpoint_epoch_20.pth
# n2k --pretrained_model ../output/tta_n2k_models/secondiou/source_pretrain_wiener/default/ckpt/checkpoint_epoch_43.pth
# pvrcnn w2k --pretrained_model
             #../output/tta_w2k_models/pvrcnn/source_pretrain/default/ckpt/checkpoint_epoch_5.pth



# ped ../output/tta_w2k_models/secondiou/source_pretrain_ped/default/ckpt/checkpoint_epoch_29.pth

#python train.py --cfg_file $cfg_path --batch_size 8 --gpu_id $gpu_id && \
python test.py  --cfg_file $cfg_path --batch_size 8 --eval_all --set DATA_CONFIG_TAR.FOV_POINTS_ONLY True