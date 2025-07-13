from pcdet.utils.tta_utils import *
import copy
import time
import torch

import os
import glob
import tqdm
import numpy as np
import torch.distributed as dist
from pcdet.config import cfg
from pcdet.models import load_data_to_gpu, build_network
from pcdet.utils import common_utils, commu_utils, memory_ensemble_utils
import pickle as pkl
import re
from pcdet.models.model_utils.dsnorm import set_ds_target
from torch.nn.utils import clip_grad_norm_
from pcdet.utils.train_utils import save_checkpoint, checkpoint_state
import wandb
from scipy.optimize import linear_sum_assignment

from pcdet.utils.tta_utils import update_ema_variables, TTA_augmentation, transform_augmented_boxes

from pcdet.ops.iou3d_nms import iou3d_nms_utils
import random

PSEUDO_LABELS = {}
NEW_PSEUDO_LABELS = {}



def check_already_exsit_pseudo_label(ps_label_dir, start_epoch):
    """
    if we continue training, use this to directly
    load pseudo labels from exsiting result pkl

    if exsit, load latest result pkl to PSEUDO LABEL
    otherwise, return false and

    Args:
        ps_label_dir: dir to save pseudo label results pkls.
        start_epoch: start epoc
    Returns:

    """
    # support init ps_label given by cfg
    if start_epoch == 0 and cfg.SELF_TRAIN.get('INIT_PS', None):
        if os.path.exists(cfg.SELF_TRAIN.INIT_PS):
            init_ps_label = pkl.load(open(cfg.SELF_TRAIN.INIT_PS, 'rb'))
            PSEUDO_LABELS.update(init_ps_label)

            if cfg.LOCAL_RANK == 0:
                ps_path = os.path.join(ps_label_dir, "ps_label_e0.pkl")
                with open(ps_path, 'wb') as f:
                    pkl.dump(PSEUDO_LABELS, f)

            return cfg.SELF_TRAIN.INIT_PS

    ps_label_list = glob.glob(os.path.join(ps_label_dir, 'ps_label_e*.pkl'))
    if len(ps_label_list) == 0:
        return

    ps_label_list.sort(key=os.path.getmtime, reverse=True)
    for cur_pkl in ps_label_list:
        num_epoch = re.findall('ps_label_e(.*).pkl', cur_pkl)
        assert len(num_epoch) == 1

        # load pseudo label and return
        if int(num_epoch[0]) <= start_epoch:
            latest_ps_label = pkl.load(open(cur_pkl, 'rb'))
            PSEUDO_LABELS.update(latest_ps_label)
            return cur_pkl

    return None



def bagging(model, val_loader, rank, leave_pbar, ps_label_dir, cur_epoch, model_func=None, lr_scheduler=None,
                       accumulated_iter=None, optim_cfg=None, tbar=None, total_it_each_epoch=None,
                       dataloader_iter=None, tb_log=None,ema_model=None, optimizer=None, ckpt_save_interval_iter=64, ckpt_save_dir=None,
                        logger=None,model_copy=None):
    """
    Generate pseudo label with given model.

    Args:
        model: model to predict result for pseudo label
        val_loader: data_loader to predict pseudo label
        rank: process rank
        leave_pbar: tqdm bar controller
        ps_label_dir: dir to save pseudo label
        cur_epoch
    """

    """ For SINGLE layer LS """
    # for nm, m in model.named_modules():
    #     print(nm)
    #     if 'roi_head.iou_layers.7' in nm:
    #         print(m)
    #         m.requires_grad_(True)
    #     else:
    #         m.requires_grad_(False)

    # 1. load 3 models
    # 2. for each test batch, generae the aggregated psseudolabels (results), randomly select one to update.
    


    # meter
    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    val_loader.dataset.student_aug, val_loader.dataset.teacher_aug = False, False
    val_dataloader_iter = iter(val_loader)
    total_it_each_epoch = len(val_loader)


    model_2 = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=val_loader.dataset)
    model_2.load_params_from_file(filename='../output/tta_w2k_models/secondiou/source_pretrain/default/ckpt/checkpoint_epoch_20.pth', logger=logger, report_logger=False,to_cpu=dist)
    model_3 = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=val_loader.dataset)
    model_3.load_params_from_file(filename='../output/tta_w2k_models/secondiou/source_pretrain/default/ckpt/checkpoint_epoch_20.pth', logger=logger, report_logger=False,to_cpu=dist)


    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                         desc='online_test_time_adaptation', dynamic_ncols=True)

    model_2.cuda(); model_3.cuda()
    model.eval(); model_2.eval(); model_3.eval()


    models = [model, model_2, model_3]

    b_size = val_loader.batch_size
    aggregated_model = None
    model_bank = []
    latest_model = None
    adaptation_time = []; infer_time = []; total_time = []
    c_hist = None

    det_annos = []

    for cur_it in range(total_it_each_epoch):

        total_time_start = time.time()

        try:
            target_batch = next(val_dataloader_iter)
        except StopIteration:
            target_dataloader_iter = iter(val_loader)
            target_batch = next(target_dataloader_iter)

        target_batch_copy = copy.deepcopy(target_batch)

        # generate gt_boxes for target_batch and update model weights
        with torch.no_grad():

            target_batch_2=  copy.deepcopy(target_batch)
            load_data_to_gpu(target_batch_2)
            target_batch_3=  copy.deepcopy(target_batch)
            load_data_to_gpu(target_batch_3)
            load_data_to_gpu(target_batch)

            model.eval(); model_2.eval(); model_3.eval()
            
            pred_dicts, ret_dict = model(target_batch)

            pred_dicts_2, ret_dict_2 = model_2(target_batch_2)


            pred_dicts_3, ret_dict_3 = model_3(target_batch_3)

            all_predicted_boxes = []
            all_predicted_conf = []
            all_predicted_label = []
            for i in range(len(pred_dicts)):
                
                all_predicted_boxes = torch.concat([pred_dicts[i]['pred_boxes'], pred_dicts_2[i]['pred_boxes'], pred_dicts_3[i]['pred_boxes']])
                all_predicted_conf = torch.concat([pred_dicts[i]['pred_scores'], pred_dicts_2[i]['pred_scores'], pred_dicts_3[i]['pred_scores']])
                all_predicted_label = torch.concat([pred_dicts[i]['pred_labels'], pred_dicts_2[i]['pred_labels'], pred_dicts_3[i]['pred_labels']])
                
                all_pred_cls_scores = torch.concat([pred_dicts[i]['pred_cls_scores'], pred_dicts_2[i]['pred_cls_scores'], pred_dicts_3[i]['pred_cls_scores']])
                all_pred_iou_scores = torch.concat([pred_dicts[i]['pred_iou_scores'], pred_dicts_2[i]['pred_iou_scores'], pred_dicts_3[i]['pred_iou_scores']])
                
                ids = iou3d_nms_utils.nms_gpu(all_predicted_boxes, all_predicted_conf, 0.7)[0]
                pred_dicts[i]['pred_boxes'] = all_predicted_boxes[ids]
                pred_dicts[i]['pred_scores'] = all_predicted_conf[ids]
                pred_dicts[i]['pred_labels'] = all_predicted_label[ids]
                pred_dicts[i]['pred_cls_scores'] = all_pred_cls_scores[ids]
                pred_dicts[i]['pred_iou_scores'] = all_pred_iou_scores[ids]

                # all_predicted_boxes.append(pred_dicts[i]['pred_boxes'])
                # all_predicted_conf.append(pred_dicts[i]['pred_scores'])
                # all_predicted_label.append(pred_dicts[i]['pred_labels'])


            
            # if cur_it == 0 and : # for computing g
            #     init_features = pred_dicts[0]['shared_features']

        """ Start TTA """
        samples_seen = int(cur_it)*int(b_size)

        """ Change AUG TTA """
        # if samples_seen == 256:
        #     cfg.DATA_CONFIG_TAR.TTA_DATA_AUGMENTOR.AUG_CONFIG_LIST[0]['SCALE_UNIFORM_NOISE'] = [0.9, 1.1]

        adaptation_time_start = time.time()

        """ Generate initial Psuedo labels (teacher augmentation)"""
        if cfg.TTA.METHOD in ['iwa', 'avg_agg', 'bagging']:
            _, _ = save_pseudo_label_batch(
                target_batch, pred_dicts=pred_dicts,
                need_update=(cfg.SELF_TRAIN.get('MEMORY_ENSEMBLE', None) and
                             cfg.SELF_TRAIN.MEMORY_ENSEMBLE.ENABLED and
                             cur_it > 0)
            )


        # Set BN EMA decay
        if ema_model is not None and ema_model.model_cfg.get('BN_EMA_DECAY', False):
            max_bn_ema = ema_model.model_cfg.BN_EMA
            min_bn_ema = ema_model.model_cfg.MIN_BN_EMA
            multiplier = (np.cos(cur_it / total_it_each_epoch * np.pi) + 1) * 0.5
            cur_bn_ema = min_bn_ema + multiplier * (max_bn_ema - min_bn_ema)
            model.module.set_momemtum_value_for_bn(momemtum=(1 - cur_bn_ema))

        # Start to train this single frame
        lr_scheduler.step(cur_it)
        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']
        # Log learning rate change
        wandb.log({'meta_data/learning_rate': cur_lr}, step=int(cur_it))


        if cfg.TTA.METHOD in ['naive', 'tent', 'naive_ros', 'mean_teacher', 'iwa', 'avg_agg', 'cotta', 'bagging']:

            if cfg.TTA.METHOD in ['iwa', 'avg_agg', 'cotta', 'bagging']:
                # Replaces the real GT with PS boxes
                max_box_num_batch = np.max([NEW_PSEUDO_LABELS[frame_id.item()]['gt_boxes'].shape[0] for frame_id in target_batch['frame_id']])
                new_batch_ps_boxes = torch.zeros(b_size, max_box_num_batch, 8).cuda()
                for b_id in range(b_size):
                    ps_gt_boxes = torch.tensor(NEW_PSEUDO_LABELS[target_batch['frame_id'][b_id].item()]['gt_boxes']).cuda()[:,:8].float()
                    gap = max_box_num_batch - ps_gt_boxes.shape[0]
                    ps_gt_boxes = torch.concat((ps_gt_boxes, torch.zeros(gap, 8).cuda())) # concat ps_gt_boxes and empty 0 boxes to max_box_num_batch
                    # new_batch_ps_boxes[b_id] = ps_gt_boxes.reshape(ps_gt_boxes.shape[0], ps_gt_boxes.shape[1])
                    new_batch_ps_boxes[b_id] = ps_gt_boxes
                target_batch['gt_boxes'] = new_batch_ps_boxes

            model.train()



            """ For SINGLE layer LS """
            # for nm, m in model.named_modules():
            #     # print(nm)
            #     if 'roi_head.iou_layers.7' in nm:
            #         # print(m)
            #         m.requires_grad_(True)
            #     else:
            #         m.requires_grad_(False)



            optimizer.zero_grad()

            dataset = val_loader.dataset
            if cfg.TTA.METHOD in ['mean_teacher'] or (cfg.TTA.METHOD == 'iwa' and cfg.DATA_CONFIG_TAR.get('CORRUPT', None)):
                target_batch = TTA_augmentation(dataset, target_batch, strength='strong')
            elif cfg.TTA.METHOD in ['naive_ros',  'avg_agg', 'iwa', 'bagging']:
                target_batch = TTA_augmentation(dataset, target_batch)
            elif cfg.TTA.METHOD in ['tent']:
                pass
                # https://github.com/DequanWang/tent/blob/e9e926a668d85244c66a6d5c006efbd2b82e83e8/tent.py#L96
                # disable grad, to (re-)enable only what tent updates
                model.requires_grad_(False)
                # # configure norm for tent updates: enable grad + force batch statisics
                for m in model.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.requires_grad_(True)
                        # force use of batch stats in train and eval modes
                        # m.track_running_stats = False
                        # m.running_mean = None
                        # m.running_var = None


            # Compute the loss using PS boxes
            model_id = random.choice([0, 1, 2])
            loss, st_tb_dict, st_disp_dict = model_func(models[model_id].train(), target_batch)

            loss = cfg.SELF_TRAIN.TAR.get('LOSS_WEIGHT', 1.0) * loss
            loss.backward()

            # load_data_to_gpu(target_batch)
            # st_loss, st_tb_dict, st_disp_dict = model(target_batch)

            clip_grad_norm_(models[model_id].parameters(), optim_cfg.GRAD_NORM_CLIP)
            optimizer.step()
            accumulated_iter += 1

            model.eval(); model_2.eval(); model_3.eval()
            for key, val in st_tb_dict.items():
                wandb.log({'train/' + key: val}, step=int(cur_it))
        """ Model bank size exceeds, need to remove the lowest weighted one """
        if cfg.TTA.get('IWA_SETTING', None) and cfg.TTA.IWA_SETTING.get('AGGREGATE_BANK', None) and cfg.TTA.IWA_SETTING.AGGREGATE_BANK and len(model_bank) > cfg.TTA.IWA_SETTING.AGGREGATE_NUM:

            remove_model_index = torch.argmin(c_hist[:cfg.TTA.IWA_SETTING.AGGREGATE_NUM])
            del model_bank[remove_model_index]
            c_hist = torch.zeros(cfg.TTA.IWA_SETTING.AGGREGATE_NUM).cuda()
            # if True in list(c[-1] > c):
            #
            #     remove_model_index = torch.argmin(c)
            #     del model_bank[remove_model_index]
            # else:
            #     del model_bank[-1]

        """generate online predictions"""
        if cfg.TTA.get('INSTANCE_EVAL', None) and cfg.TTA.INSTANCE_EVAL:
            load_data_to_gpu(target_batch_copy)
            with torch.no_grad():
                pred_dicts, ret_dict = model(target_batch_copy)
            disp_dict = {}
            statistics_info(cfg, ret_dict, metric, disp_dict)
            annos = val_loader.dataset.generate_prediction_dicts(
                target_batch_copy, pred_dicts, val_loader.dataset.class_names
            )
            det_annos += annos

        """ Record Time """
        iter_end_time = time.time()
        adaptation_time.append((iter_end_time - adaptation_time_start)/b_size)
        total_time.append((iter_end_time - total_time_start)/b_size)
        infer_time.append(total_time[-1] - adaptation_time[-1])
        wandb.log({'time/' + 'adap': adaptation_time[-1]}, step=int(cur_it))
        wandb.log({'time/' + 'total': total_time[-1]}, step=int(cur_it))
        wandb.log({'time/' + 'infer': infer_time[-1]}, step=int(cur_it))

        # save trained model
        # early ckpt
        if (samples_seen in cfg.TTA.SAVE_CKPT or samples_seen % cfg.TTA.SAVE_CKPT_INTERVAL==0) and rank == 0:

            """ Update Model Bank (if num exceeds, we remove the lowest weight model)"""




            ckpt_name = ckpt_save_dir / ('checkpoint_iter_%d' % samples_seen)
            state = checkpoint_state(model, optimizer, cur_it,
                                     accumulated_iter)
            save_checkpoint(state, filename=ckpt_name)
        elif cfg.TTA.get('INSTANCE_EVAL', None) and cfg.TTA.INSTANCE_EVAL:
            pass
        elif samples_seen > cfg.TTA.SAVE_CKPT[-1] + 1:
            wandb.log({'average_time/' + 'adap': np.mean(adaptation_time)})
            wandb.log({'average_time/' + 'infer': np.mean(infer_time)})
            wandb.log({'average_time/' + 'total': np.mean(total_time)})
            print('average_time_adap:', np.mean(adaptation_time))
            print('average_time_infer:', np.mean(infer_time))
            print('average_time_total:', np.mean(total_time))
            exit()
        # if cur_it <= 32:
        #     if cur_it % 2 == 0 and rank == 0:
        #         ckpt_name = ckpt_save_dir / ('checkpoint_iter_%d' % cur_it)
        #         state = checkpoint_state(model, optimizer, cur_it, accumulated_iter)
        #         save_checkpoint(state, filename=ckpt_name)
        # else:
        #     exit()
        #     if cur_it % ckpt_save_interval_iter == 0 and rank == 0:
        #         ckpt_name = ckpt_save_dir / ('checkpoint_iter_%d' % cur_it)
        #         state = checkpoint_state(model, optimizer, cur_it, accumulated_iter)
        #         save_checkpoint(state, filename=ckpt_name)


        if rank == 0:
            pbar.update()
            pbar.refresh()

    if rank == 0:
        pbar.close()

    if cfg.TTA.get('INSTANCE_EVAL', None) and cfg.TTA.INSTANCE_EVAL:
        result_str, result_dict = val_loader.dataset.evaluation(
            det_annos, val_loader.dataset.class_names
        )

        logger.info(result_str)
        ret_dict.update(result_dict)
        exit()

    gather_and_dump_pseudo_label_result(rank, ps_label_dir, cur_epoch)


def aggregate_model(model_path_list, model_weights, dataset, logger, dist, main_model):
    # agg_model = build_network(model_cfg=cfg.MODEL,num_class=len(cfg.CLASS_NAMES),dataset=dataset)
    # agg_model.cuda()
    # Clear all parameters
    # for agg_param in agg_model.parameters():
    #     agg_param.data.mul_(0)
    # # Clear BN
    # for agg_bf in agg_model.named_buffers():
    #     name, value = agg_bf
    #     if 'running_mean' in name or 'running_var' in name:
    #         value.data.mul_(0)

    model_named_buffers = main_model.module.named_buffers() if hasattr(main_model,'module') else main_model.named_buffers()
    agg_model = None

    # Highest weight only 
    # highest_model_id = torch.argmax(model_weights)
    # print(highest_model_id)
    # model_path = model_path_list[highest_model_id]
    # agg_model = build_network(model_cfg=cfg.MODEL,num_class=len(cfg.CLASS_NAMES),dataset=dataset)
    # agg_model.load_params_from_file(filename=model_path, logger=logger,report_logger=False, to_cpu=dist)
    # agg_model.cuda()
    # agg_model.eval()


    # for agg_bf, bf in zip(agg_model.named_buffers(), model_named_buffers):
    #     agg_name, agg_value = agg_bf
    #     name, value = bf
    #     assert agg_name == name, 'name not equal:{} , {}'.format(agg_name,
    #                                                             name)
    #     if 'running_mean' in name or 'running_var' in name:
    #         agg_value.data = value.data

    # return agg_model
     # Highest weight only end

    """ For SINGLE layer LS """
    # param_list = []
    # for model_path in model_path_list:
    #     past_model = build_network(model_cfg=cfg.MODEL,num_class=len(cfg.CLASS_NAMES),dataset=dataset)
    #     past_model.load_params_from_file(filename=model_path, logger=logger,report_logger=False, to_cpu=dist)
    #     past_model.cuda()
    #     past_model.eval()
    #     for name, param in past_model.named_parameters():
    #         # if 'dense_head' in name or 'backbone_2d' in name:
    #         if 'roi_head.iou_layers.7' in name and len(param.data.shape) ==3:
    #             # print('param.data.shape')
    #             # print(param.data.shape)
    #             param_list.append(param.data.squeeze())
    # param_matrix = torch.stack(param_list)
        
    # # Calculate the mean parameters (target)
    # mean_params = param_matrix.mean(dim=0)  # Shape: (256,)
    # # print('mean_params.shape')
    # # print(mean_params.shape)
    # # Use the pseudo-inverse to handle cases where the matrix is not invertible
    # optimal_params, _ = torch.lstsq(mean_params.unsqueeze(1), param_matrix.T)
    # optimal_params = optimal_params.squeeze()


    # # optimal_params = torch.pinverse(param_matrix) @ param_matrix.mean(dim=0)
    # optimal_params = optimal_params.view(1, 256, 1)

    # for name, param in past_model.named_parameters():
    #     # if 'dense_head' in name or 'backbone_2d' in name:
    #     if 'roi_head.iou_layers.7' in name and len(param.data.shape) ==3:
    #         param.data = optimal_params
    # return past_model
    """ END For SINGLE layer LS """


    weight_i = 0
    for model_path in model_path_list:
        past_model = build_network(model_cfg=cfg.MODEL,num_class=len(cfg.CLASS_NAMES),dataset=dataset)
        past_model.load_params_from_file(filename=model_path, logger=logger,report_logger=False, to_cpu=dist)
        past_model.cuda()
        past_model.eval()

        if agg_model == None: # aggregate first model
            for name, param in past_model.named_parameters():
                # if 'dense_head' in name or 'backbone_2d' in name:
                param.data.mul_(model_weights[weight_i].data)



            # for bf in past_model.named_buffers():
            #     name, value = bf
            #     if 'running_mean' in name or 'running_var' in name:
            #         value.data.mul_(model_weights[weight_i].data)

            agg_model = past_model
        else: # aggregate subsequent models
            for (agg_name, agg_param), (name, param) in zip(agg_model.named_parameters(), past_model.named_parameters()):
                # if 'dense_head' in name or 'backbone_2d' in name:
                agg_param.data.add_(model_weights[weight_i].data, param.data)
            # Aggregate BN
            # for agg_bf, bf in zip(agg_model.named_buffers(), model_named_buffers):
            #     agg_name, agg_value = agg_bf
            #     name, value = bf
            #     assert agg_name == name, 'name not equal:{} , {}'.format(agg_name,name)
            #     if 'running_mean' in name or 'running_var' in name:
            #         agg_value.data.add_(model_weights[weight_i].data, value.data)

        weight_i = weight_i + 1
        del past_model; torch.cuda.empty_cache()

    for agg_bf, bf in zip(agg_model.named_buffers(), model_named_buffers):
        agg_name, agg_value = agg_bf
        name, value = bf
        assert agg_name == name, 'name not equal:{} , {}'.format(agg_name,
                                                                name)
        if 'running_mean' in name or 'running_var' in name:
            agg_value.data = value.data

    return agg_model

def gather_and_dump_pseudo_label_result(rank, ps_label_dir, cur_epoch):
    commu_utils.synchronize()

    if dist.is_initialized():
        part_pseudo_labels_list = commu_utils.all_gather(NEW_PSEUDO_LABELS)

        new_pseudo_label_dict = {}
        for pseudo_labels in part_pseudo_labels_list:
            new_pseudo_label_dict.update(pseudo_labels)

        NEW_PSEUDO_LABELS.update(new_pseudo_label_dict)

    # dump new pseudo label to given dir
    if rank == 0:
        ps_path = os.path.join(ps_label_dir, "ps_label_e{}.pkl".format(cur_epoch))
        with open(ps_path, 'wb') as f:
            pkl.dump(NEW_PSEUDO_LABELS, f)

    commu_utils.synchronize()
    PSEUDO_LABELS.clear()
    PSEUDO_LABELS.update(NEW_PSEUDO_LABELS)
    NEW_PSEUDO_LABELS.clear()


def save_pseudo_label_batch(input_dict,
                            pred_dicts=None,
                            need_update=False):
    """
    Save pseudo label for give batch.
    If model is given, use model to inference pred_dicts,
    otherwise, directly use given pred_dicts.

    Args:
        input_dict: batch data read from dataloader
        pred_dicts: Dict if not given model.
            predict results to be generated pseudo label and saved
        need_update: Bool.
            If set to true, use consistency matching to update pseudo label
    """
    pos_ps_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))
    ign_ps_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))

    batch_size = len(pred_dicts)
    for b_idx in range(batch_size):
        pred_cls_scores = pred_iou_scores = None
        if 'pred_boxes' in pred_dicts[b_idx]:
            # Exist predicted boxes passing self-training score threshold
            pred_boxes = pred_dicts[b_idx]['pred_boxes'].detach().cpu().numpy()
            pred_labels = pred_dicts[b_idx]['pred_labels'].detach().cpu().numpy()
            pred_scores = pred_dicts[b_idx]['pred_scores'].detach().cpu().numpy()
            if 'pred_cls_scores' in pred_dicts[b_idx]:
                pred_cls_scores = pred_dicts[b_idx]['pred_cls_scores'].detach().cpu().numpy()
            if 'pred_iou_scores' in pred_dicts[b_idx]:
                pred_iou_scores = pred_dicts[b_idx]['pred_iou_scores'].detach().cpu().numpy()

            # remove boxes under negative threshold
            if cfg.SELF_TRAIN.get('NEG_THRESH', None):
                labels_remove_scores = np.array(cfg.SELF_TRAIN.NEG_THRESH)[pred_labels - 1]
                remain_mask = pred_scores >= labels_remove_scores
                pred_labels = pred_labels[remain_mask]
                pred_scores = pred_scores[remain_mask]
                pred_boxes = pred_boxes[remain_mask]
                if 'pred_cls_scores' in pred_dicts[b_idx]:
                    pred_cls_scores = pred_cls_scores[remain_mask]
                if 'pred_iou_scores' in pred_dicts[b_idx]:
                    pred_iou_scores = pred_iou_scores[remain_mask]

            labels_ignore_scores = np.array(cfg.SELF_TRAIN.SCORE_THRESH)[pred_labels - 1]
            ignore_mask = pred_scores < labels_ignore_scores
            pred_labels[ignore_mask] = -pred_labels[ignore_mask]

            if cfg.SELF_TRAIN.get('FIX_POS_NUM', None):
                expected_pos_num = pred_labels.shape[0] if pred_labels.shape[0] < cfg.SELF_TRAIN.FIX_POS_NUM else cfg.SELF_TRAIN.FIX_POS_NUM
                pred_labels[expected_pos_num:][pred_labels[expected_pos_num:] > 0] = - \
                pred_labels[expected_pos_num:][pred_labels[expected_pos_num:] > 0]

            gt_box = np.concatenate((pred_boxes,
                                     pred_labels.reshape(-1, 1),
                                     pred_scores.reshape(-1, 1)), axis=1)

        else:
            # no predicted boxes passes self-training score threshold
            gt_box = np.zeros((0, 9), dtype=np.float32)

        gt_infos = {
            'gt_boxes': gt_box,
            'cls_scores': pred_cls_scores,
            'iou_scores': pred_iou_scores,
            'memory_counter': np.zeros(gt_box.shape[0])
        }

        # record pseudo label to pseudo label dict
        if need_update:
            ensemble_func = getattr(memory_ensemble_utils, cfg.SELF_TRAIN.MEMORY_ENSEMBLE.NAME)
            gt_infos = memory_ensemble_utils.memory_ensemble(
                PSEUDO_LABELS[input_dict['frame_id'][b_idx]], gt_infos,
                cfg.SELF_TRAIN.MEMORY_ENSEMBLE, ensemble_func
            )

        # counter the number of ignore boxes for each class
        for i in range(ign_ps_nmeter.n):
            num_total_boxes = (np.abs(gt_infos['gt_boxes'][:, 7]) == (i+1)).sum()
            ign_ps_nmeter.update((gt_infos['gt_boxes'][:, 7] == -(i+1)).sum(), index=i)
            pos_ps_nmeter.update(num_total_boxes - ign_ps_nmeter.meters[i].val, index=i)

        NEW_PSEUDO_LABELS[input_dict['frame_id'][b_idx]] = gt_infos

    return pos_ps_nmeter, ign_ps_nmeter


def load_ps_label(frame_id):
    """
    :param frame_id: file name of pseudo label
    :return gt_box: loaded gt boxes (N, 9) [x, y, z, w, l, h, ry, label, scores]
    """
    if frame_id in PSEUDO_LABELS:
        gt_box = PSEUDO_LABELS[frame_id]['gt_boxes']
    else:
        raise ValueError('Cannot find pseudo label for frame: %s' % frame_id)

    return gt_box

def hungarian_match_diff(bbox_pred_1, bbox_pred_2):

    num_bboxes = bbox_pred_1.size(0)
    # 1. assign -1 by default
    # assigned_pred_2_inds = bbox_pred_1.new_full((num_bboxes,), -1, dtype=torch.long)

    # 2. compute the costs
    # normalized_pred_2_bboxes = normalize_bbox(bbox_pred_2)
    reg_cost = torch.cdist(bbox_pred_1[:, :7],  bbox_pred_2, p=1)
    cost = reg_cost

    # 3. do Hungarian matching on CPU using linear_sum_assignment
    cost = cost.detach().cpu()
    matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
    # matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred_1.device)
    # matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred_1.device)
    # assigned_pred_2_inds[matched_row_inds] = matched_col_inds + 1
    return cost[matched_row_inds].min(dim=-1)[0].sum()

def vis_pred_pc(data_dict, pred_dict, c):
    from ..ops.roiaware_pool3d import roiaware_pool3d_utils
    from pcdet.utils.box_utils import remove_points_in_boxes3d, enlarge_box3d, \
        boxes3d_kitti_lidar_to_fakelidar, boxes_to_corners_3d

    points = data_dict['points'][data_dict['points'][:,0]==0][:,1:].cpu().numpy()
    # points = points[points[:,-1]==0]
    rgb = torch.tensor([179, 179, 179]).expand(points.shape[0],3).numpy()
    points = np.concatenate([points, rgb], axis=1)
    boxes = pred_dict['pred_boxes'].cpu().numpy()[:,:7]

    gt_boxes = data_dict['gt_boxes'][0].cpu().numpy()
    gt_boxes=None
    fg_point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points[:,:3], boxes)
    bg_points = points[fg_point_masks.sum(axis=0) == 0]
    fg_points = points[fg_point_masks.sum(axis=0) != 0]

    """------- Draw BG points -------"""
    bg_rgb = np.zeros((bg_points.shape[0], 3))
    bg_rgb[:, ] = [179, 179, 179]  # grey point
    bg_points_rgb = np.array(
        [[p[0], p[1], p[2], c[0], c[1], c[2]] for p, c in
         zip(bg_points, bg_rgb)])

    """------- Draw FG points -------"""
    fg_rgb = np.zeros((fg_points.shape[0], 3))
    fg_rgb[:, ] = [179, 179, 179]  # grey point
    # light blue 139,193,205
    # blue 98 161	179
    # light green 177	202	78
    # 	122,157,61
    fg_points_rgb = np.array(
        [[p[0], p[1], p[2], c[0], c[1], c[2]] for p, c in
         zip(fg_points, fg_rgb)])

    points_rgb = np.concatenate([bg_points_rgb, fg_points_rgb])


    boxes_to_draw = []
    """------- Draw known boxes -------"""
    if boxes is not None:
        for i in range(len(boxes_to_corners_3d(boxes).tolist())):
            box = boxes_to_corners_3d(boxes).tolist()[i]
            label = "{}".format(boxes[i])

            boxes_ref_label = {
                "corners": list(box),
                # optionally customize each label
                "label": None,
                "color": [0, 255, 0], # green 0, 255, 0
            }
            boxes_to_draw.append(boxes_ref_label)

    """------- Draw known boxes -------"""
    if gt_boxes is not None:
        for i in range(len(boxes_to_corners_3d(gt_boxes).tolist())):
            gt_box = boxes_to_corners_3d(gt_boxes).tolist()[i]
            label = "{}".format(gt_boxes[i])

            boxes_ref_label = {
                "corners": list(gt_box),
                # optionally customize each label
                "label": None, #str(i)
                "color": [255,0,0], # green 0, 255, 0
            }
            boxes_to_draw.append(boxes_ref_label)

    boxes_to_draw = np.array(boxes_to_draw)

    wandb.log(
        {"3d point cloud"+"/"+str(c):
            wandb.Object3D({
                "type": "lidar/beta",
                "points": points_rgb,
                "boxes": boxes_to_draw
            })
        })

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

