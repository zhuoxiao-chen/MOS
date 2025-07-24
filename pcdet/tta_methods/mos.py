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

PSEUDO_LABELS = {}
NEW_PSEUDO_LABELS = {}

def mos(model, val_loader, rank, leave_pbar, ps_label_dir, cur_epoch, model_func=None, lr_scheduler=None,
                       accumulated_iter=None, optim_cfg=None, tbar=None, total_it_each_epoch=None,
                       dataloader_iter=None, tb_log=None,ema_model=None, optimizer=None, ckpt_save_interval_iter=64, ckpt_save_dir=None,
                        logger=None,model_copy=None):
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

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                         desc='online_test_time_adaptation', dynamic_ncols=True)

    model.eval()

    b_size = val_loader.batch_size
    super_model = None
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
            load_data_to_gpu(target_batch)
            model.eval()
            pred_dicts, ret_dict = model(target_batch)

        """ Start Adapt """
        samples_seen = int(cur_it)*int(b_size)

        adaptation_time_start = time.time()

        """ Generate Initial Pseudo Labels"""
        _, _ = save_pseudo_label_batch(
            target_batch, pred_dicts=pred_dicts,
            need_update=(cfg.SELF_TRAIN.get('MEMORY_ENSEMBLE', None) and
                         cfg.SELF_TRAIN.MEMORY_ENSEMBLE.ENABLED and
                         cur_it > 0)
        )

        # Start to train this single frame
        lr_scheduler.step(cur_it)
        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']
        # Log learning rate change
        wandb.log({'meta_data/learning_rate': cur_lr}, step=int(cur_it))

        if cfg.TTA.METHOD in ['mos'] and cfg.TTA.MOS_SETTING.AGGREGATE_START_CKPT <= samples_seen:

            feat_map_list = []
            pred_box_list_whole_batch = []
            pred_box_list_per_frame = []

            model_path_list = glob.glob(os.path.join(ckpt_save_dir, '*checkpoint_iter_*.pth'))
            model_path_list.sort(key=os.path.getmtime)
            num_aggregate = len(model_path_list) if len(model_path_list) < cfg.TTA.MOS_SETTING.AGGREGATE_NUM else cfg.TTA.MOS_SETTING.AGGREGATE_NUM
            model_path_list = model_path_list[-num_aggregate:]

            if num_aggregate >= 3:

                """ Initialize / Load Checkpoints from Model Bank """
                use_model_bank = True if cfg.TTA.MOS_SETTING.get('AGGREGATE_BANK', None) and cfg.TTA.MOS_SETTING.AGGREGATE_BANK else False
                if use_model_bank and samples_seen >= cfg.TTA.MOS_SETTING.BANK_START_CKPT:
                    if len(model_path_list) < cfg.TTA.MOS_SETTING.AGGREGATE_NUM or len(model_bank)==0:
                        # initialize model bank
                        model_bank = model_path_list

                    if latest_model is None:
                        latest_model = model_path_list[-1]
                    elif latest_model != model_path_list[-1]:
                        # found new model, add it to model_bank
                        latest_model = model_path_list[-1]
                        model_bank.append(latest_model)
                    model_path_list = model_bank

                record_predicts = []
                box_num = []
                """ Inference Models from Bank """
                for past_model_path in model_path_list:
                    past_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=val_loader.dataset)
                    past_model.load_params_from_file(filename=past_model_path, logger=logger, report_logger=False,to_cpu=dist)
                    past_model.cuda()
                    past_model.eval()
                    pred_dicts, ret_dict = past_model(target_batch)

                    box_num.append(pred_dicts[0]['pred_boxes'].shape[0])

                    record_predicts.append(pred_dicts)
                    del past_model; torch.cuda.empty_cache()

                    """ Save intermediate features of each model """
                    # feat_map_list.append(pred_dicts[0]['spatial_features_2d'].view(188 * 188,-1))
                    # For acceleration, we use pooled features instead which include richer info
                    feat_map_list.append(pred_dicts[0]['shared_features'])

                    """ Save box predictsons of each model """
                    for pred_dict in pred_dicts:
                        pred_box_list_per_frame.append(pred_dict['pred_boxes'])
                    pred_box_list_whole_batch.append(pred_box_list_per_frame)
                    pred_box_list_per_frame = []

                if cfg.TTA.METHOD == 'mos':
                    """ Compute the Generalized Gram Matrix: G """
                    total_box_cost = 0
                    G = torch.zeros(len(feat_map_list), len(feat_map_list)).cuda()
                    for i in range(len(feat_map_list)):
                        for j in range(len(feat_map_list)):
                            batch_results = torch.zeros(b_size)
                            for b in range(b_size):
                                if pred_box_list_whole_batch[j][b].shape[0] == 0 or pred_box_list_whole_batch[i][b].shape[0] == 0:
                                    # no box at all, we dont want it, make it to be similar to every model
                                    batch_results[b] = 0.01
                                else:
                                    batch_results[b] = hungarian_match_diff(pred_box_list_whole_batch[i][b], pred_box_list_whole_batch[j][b])

                            # add 0.035 to avoid 0
                            feat_sim = (2*feat_map_list[j].shape[1] - torch.linalg.matrix_rank(torch.concat([feat_map_list[i].squeeze(),feat_map_list[j].squeeze()]))) /  (2*feat_map_list[j].shape[1])
                            # add small noise here to ensure G is invertible
                            feat_sim = feat_sim + feat_sim * torch.rand(1).cuda()*0.05
                            box_sim = torch.sigmoid(1 / batch_results.sum())
                            total_box_cost = total_box_cost + batch_results.sum()
                            G[i][j] = feat_sim*box_sim

                    G_inverse = torch.linalg.inv(G)

                    ones = torch.ones(len(feat_map_list)).cuda()

                    """ Compute Final Synergy Weights: c (in paper it's denoted as w)"""
                    c = torch.matmul(G_inverse, ones)
                    c_min, c_max = c.min(), c.max(); new_min, new_max = 0.05, 0.95
                    # normalize weights
                    c = (c - c_min) / (c_max - c_min) * (new_max - new_min) + new_min
                    c /= c.sum()

                    if len(c) >=  cfg.TTA.MOS_SETTING.AGGREGATE_NUM:
                        if c_hist is None:
                            c_hist = torch.zeros(cfg.TTA.MOS_SETTING.AGGREGATE_NUM).cuda()
                        else:
                            c_hist = c_hist + c[:cfg.TTA.MOS_SETTING.AGGREGATE_NUM]

                # Now given weights c, aggregate models
                super_model = aggregate_model(model_path_list, c, val_loader.dataset, logger, dist, model)

                # save memory
                if cfg.TTA.METHOD == 'mos':
                    del G; del G_inverse; del ones
                torch.cuda.empty_cache()


        if cfg.TTA.METHOD in ['mos']:
            """ Generate PS boxes for current target_batch using super model"""
            if super_model is not None and cfg.TTA.METHOD in ['mos']:

                super_model.eval()
                with torch.no_grad():
                    load_data_to_gpu(target_batch)
                    pred_dicts, ret_dict = super_model(target_batch)
                    _, _ = save_pseudo_label_batch(
                        target_batch, pred_dicts=pred_dicts,
                        need_update=(cfg.SELF_TRAIN.get('MEMORY_ENSEMBLE',None) and
                                     cfg.SELF_TRAIN.MEMORY_ENSEMBLE.ENABLED and
                                     cur_it > 0)
                    )

            # Replaces the real GT with PS boxes
            if cfg.TTA.METHOD in ['mos']:
                max_box_num_batch = np.max([NEW_PSEUDO_LABELS[frame_id.item()]['gt_boxes'].shape[0] for frame_id in target_batch['frame_id']])
                new_batch_ps_boxes = torch.zeros(b_size, max_box_num_batch, 8).cuda()
                for b_id in range(b_size):
                    ps_gt_boxes = torch.tensor(NEW_PSEUDO_LABELS[target_batch['frame_id'][b_id].item()]['gt_boxes']).cuda()[:,:8].float()
                    gap = max_box_num_batch - ps_gt_boxes.shape[0]
                    ps_gt_boxes = torch.concat((ps_gt_boxes, torch.zeros(gap, 8).cuda())) # concat ps_gt_boxes and empty 0 boxes to max_box_num_batch
                    new_batch_ps_boxes[b_id] = ps_gt_boxes
                target_batch['gt_boxes'] = new_batch_ps_boxes

            """ Train current model with PS boxes """
            model.train()
            optimizer.zero_grad()

            dataset = val_loader.dataset

            # augmentation following ST3D or MLCNet
            if cfg.TTA.METHOD in ['mean_teacher'] or (cfg.TTA.METHOD == 'mos' and cfg.DATA_CONFIG_TAR.get('CORRUPT', None)):
                target_batch = TTA_augmentation(dataset, target_batch, strength='strong')
            elif cfg.TTA.METHOD in ['naive_ros',  'avg_agg', 'mos']:
                target_batch = TTA_augmentation(dataset, target_batch)

            loss, st_tb_dict, st_disp_dict = model_func(model, target_batch)
            loss = cfg.SELF_TRAIN.TAR.get('LOSS_WEIGHT', 1.0) * loss
            loss.backward()
            clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            optimizer.step()
            accumulated_iter += 1

            model.eval()
            for key, val in st_tb_dict.items():
                wandb.log({'train/' + key: val}, step=int(cur_it))

        """ Model Bank Update """
        if cfg.TTA.get('MOS_SETTING', None) and cfg.TTA.MOS_SETTING.get('AGGREGATE_BANK', None) and cfg.TTA.MOS_SETTING.AGGREGATE_BANK and len(model_bank) > cfg.TTA.MOS_SETTING.AGGREGATE_NUM:

            remove_model_index = torch.argmin(c_hist[:cfg.TTA.MOS_SETTING.AGGREGATE_NUM])
            del model_bank[remove_model_index]
            c_hist = torch.zeros(cfg.TTA.MOS_SETTING.AGGREGATE_NUM).cuda()

        if ema_model is not None:
            update_ema_variables(model, ema_model, model_cfg=ema_model.model_cfg, cur_epoch=cur_epoch, total_epochs=1,
                                cur_it=cur_it, total_it=total_it_each_epoch)

        """Generate online prediction"""
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

        # Save adapted checkpoints during the test-time
        if (samples_seen in cfg.TTA.SAVE_CKPT or samples_seen % cfg.TTA.SAVE_CKPT_INTERVAL==0) and rank == 0:
            ckpt_name = ckpt_save_dir / ('checkpoint_iter_%d' % samples_seen)
            state = checkpoint_state(model, optimizer, cur_it, accumulated_iter)
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
            return

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

    return


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

