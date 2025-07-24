# Getting Started
The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs), 
and the model configs are located within [tools/cfgs](../tools/cfgs) for different datasets. 



## Training & Testing


### Test and evaluate the pretrained models
* Test with a pretrained model: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```

* Notice that if you want to test on the setting with KITTI as **target domain**, 
  please add `--set DATA_CONFIG_TAR.FOV_POINTS_ONLY True` to enable front view
  point cloud only: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all --set DATA_CONFIG_TAR.FOV_POINTS_ONLY True
```

* To test with multiple GPUs:
```shell script
sh scripts/dist_test.sh ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}

# or

sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_GPUS} \ 
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```


### Train a model
You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters. 
  

* Train with multiple GPUs or multiple machines
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

# or 

sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

* Train with a single GPU:
```shell script
python train.py --cfg_file ${CONFIG_FILE}
```

### Train the Pre-trained 
Take Source Only model with SECOND-IoU on Waymo -> KITTI  as an example:
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file cfgs/da-waymo-kitti_models/secondiou/secondiou_old_anchor.yaml \
    --batch_size ${BATCH_SIZE}
```
Notice that you need to select the **best model** as your Pre-train model, 
because the performance of adapted model is really unstable when target domain is KITTI.


### Self-training Process
You need to set the `--pretrained_model ${PRETRAINED_MODEL}` when finish the
following self-training process.
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file cfgs/da-waymo-kitti_models/secondiou_st3d/secondiou_st3d_car.yaml \
    --batch_size ${BATCH_SIZE} --pretrained_model ${PRETRAINED_MODEL}
```
Notice that you also need to focus the performance of the **best model**.
