import pickle
import time
import copy
import tqdm
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from collections import OrderedDict
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.models.model_utils.dsnorm import set_ds_target
from pcdet.datasets.augmentor import augmentor_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file



def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

@torch.no_grad()
def update_teacher_model(model_student, model_teacher, keep_rate=0.9996):
    student_model_dict = model_student.state_dict()
    new_teacher_dict = OrderedDict()
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] *
                (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))

    return new_teacher_dict

def extract_roi_feature(model_cfg, roi, feature_map, batch_size):
    data_cfg_file = model_cfg.DATA_CONFIG_TAR['_BASE_CONFIG_']
    with open(data_cfg_file, "r") as f:
        try:
            dataset_cfg = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            dataset_cfg = yaml.safe_load(f)
    rois = roi.detach()
    spatial_features_2d = feature_map.detach()
    height, width = spatial_features_2d.size(2), spatial_features_2d.size(3)

    min_x = dataset_cfg['POINT_CLOUD_RANGE'][0]
    min_y = dataset_cfg['POINT_CLOUD_RANGE'][1]
    voxel_size_x = dataset_cfg['DATA_PROCESSOR'][-1]['VOXEL_SIZE'][0]
    voxel_size_y = dataset_cfg['DATA_PROCESSOR'][-1]['VOXEL_SIZE'][1]
    down_sample_ratio = model_cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.DOWNSAMPLE_RATIO

    pooled_features_list = []
    torch.backends.cudnn.enabled = False
    for b_id in range(batch_size):
            # Map global boxes coordinates to feature map coordinates
        x1 = (rois[b_id, :, 0] - rois[b_id, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
        x2 = (rois[b_id, :, 0] + rois[b_id, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
        y1 = (rois[b_id, :, 1] - rois[b_id, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)
        y2 = (rois[b_id, :, 1] + rois[b_id, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)

        angle, _ = common_utils.check_numpy_to_torch(rois[b_id, :, 6])

        cosa = torch.cos(angle)
        sina = torch.sin(angle)

        theta = torch.stack((
                (x2 - x1) / (width - 1) * cosa, (x2 - x1) / (width - 1) * (-sina), (x1 + x2 - width + 1) / (width - 1),
                (y2 - y1) / (height - 1) * sina, (y2 - y1) / (height - 1) * cosa, (y1 + y2 - height + 1) / (height - 1)
            ), dim=1).view(-1, 2, 3).float()

        grid_size = model_cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.GRID_SIZE
        if torch.__version__ >= '1.3':
            affine_grid = partial(F.affine_grid, align_corners=True)
            grid_sample = partial(F.grid_sample, align_corners=True)
        else:
            affine_grid = F.affine_grid
            grid_sample = F.grid_sample
        grid = affine_grid(
                theta,
                torch.Size((rois.size(1), spatial_features_2d.size(1), grid_size, grid_size))
            )

        pooled_features = grid_sample(
                spatial_features_2d[b_id].unsqueeze(0).expand(rois.size(1), spatial_features_2d.size(1), height, width),
                grid
            )

        pooled_features_list.append(pooled_features)

        torch.backends.cudnn.enabled = True
        pooled_features = torch.cat(pooled_features_list, dim=0)

    return pooled_features

def produce_ps_label(batch_dict, pred_dicts):
    batch_size = len(pred_dicts)
    gt_boxes = []
    for b_idx in range(batch_size):
        pred_cls_scores = pred_iou_scores = None
        if 'pred_boxes' in pred_dicts[b_idx]:
            pred_boxes = pred_dicts[b_idx]['pred_boxes'].detach().cpu().numpy()
            pred_labels = pred_dicts[b_idx]['pred_labels'].detach().cpu().numpy()
            pred_scores = pred_dicts[b_idx]['pred_scores'].detach().cpu().numpy()
            if 'pred_cls_scores' in pred_dicts[b_idx]:
                pred_cls_scores = pred_dicts[b_idx]['pred_cls_scores'].detach().cpu().numpy()
            if 'pred_iou_scores' in pred_dicts[b_idx]:
                pred_iou_scores = pred_dicts[b_idx]['pred_iou_scores'].detach().cpu().numpy()
        
        gt_box = np.concatenate((pred_boxes,
                                pred_labels.reshape(-1,1)), axis=1)
        # mask = gt_box[:, 7] >= score_threshold
        # gt_box = gt_box[mask].view(-1, 8)[:, :, 7]
        gt_box = torch.from_numpy(gt_box).float().unsqueeze(0)
        gt_boxes.append(gt_box)

    batch_dict['gt_boxes'] = torch.cat(gt_boxes,dim=0).cuda()
    return batch_dict


def eval_one_epoch(cfg, model, s_model, optimizer, dataloader, test_loader, epoch_id, logger, 
                    model_func=None, dist_test=False, save_to_file=False, result_dir=None, args=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()
    s_model.train()

    if cfg.get('SELF_TRAIN', None) and cfg.SELF_TRAIN.get('DSNORM', None):
        model.apply(set_ds_target)

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    # eval_mem_epoch(cfg, model, test_loader, epoch_id, logger, dist_test=dist_test, result_dir=result_dir, save_to_file=save_to_file, args=args)

    for i, (batch_dict, test_batch) in enumerate(zip(dataloader,test_loader)):
        optimizer.zero_grad()
        load_data_to_gpu(batch_dict)
        load_data_to_gpu(test_batch)
        with torch.no_grad():
            # roi_feature: (B*MAX,C,H,W) roi: (B, MAX, 7)
            pred_dicts, ret_dict, roi_feature, roi = model(batch_dict)
        
        test_batch = produce_ps_label(batch_dict=test_batch, pred_dicts=pred_dicts)
        loss_model, s_tb_dict, s_disp_dict, s_spatial_feature_2d = model_func(s_model, test_batch)
        roi_feature_stu = extract_roi_feature(cfg, roi, s_spatial_feature_2d, batch_size=1)
        
        # NMS_POST_MAXSIZE = 128
        s_query = s_model.query_head(roi_feature_stu.view(128,-1).squeeze(0))
        t_query = s_model.query_head(roi_feature.view(128,-1).squeeze(0))
            

            # s_spatial_feature_2d = max_pool(s_spatial_feature_2d).contiguous.view(1,-1)
        s_value = s_model.value_head(roi_feature_stu.view(128,-1).squeeze(0))
        t_value = s_model.value_head(roi_feature_stu.view(128,-1).squeeze(0))
        s_model.mem_bank = s_model.memory_update(s_model.mem_bank,
                                                t_query.contiguous().unsqueeze(-1).unsqueeze(-1), 
                                                t_value.contiguous().unsqueeze(-1).unsqueeze(-1),
                                                )
        mem_s_query  = s_model.memory_read(s_model.mem_bank,
                                            s_query.contiguous().unsqueeze(-1).unsqueeze(-1), 
                                            s_value.contiguous().unsqueeze(-1).unsqueeze(-1),
                                                )
        loss_mem = s_model.get_mem_loss(s_query, roi_feature_stu, mem_s_query.squeeze(-1).squeeze(-1), s_value, roi_feature, t_value, s_model.mem_bank)
            # loss_mem = s_model.mem_loss(t_spatial_feature_2d, s_spatial_feature_2d)
            
        disp_dict = {}
            # loss = loss_mem + loss_model
        weight_model = loss_model.detach()
        weight_mem = loss_mem.detach()
        print("loss_mem", loss_mem)
        # loss = weight_mem*loss_model/(weight_model+weight_mem) + weight_model*loss_mem/(weight_model+weight_mem)
        loss = 0.5 * loss_mem + 0.5*loss_model
        print("loss_model", loss_model)
        print("loss", loss)
        loss.backward()
        optimizer.step()
        if loss_mem != 0:
            flag = 1
        # new_teacher_dict = update_teacher_model(s_model, model, keep_rate=0.998)
        # model.load_state_dict(new_teacher_dict)  
        
        # new_teacher_dict = update_teacher_model(s_model, model, keep_rate=0.999)
        # model.load_state_dict(new_teacher_dict)
        # eval_mem_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=dist_test, result_dir=result_dir, save_to_file=save_to_file, args=args)
    new_teacher_dict = update_teacher_model(s_model, model, keep_rate=0.999)
    model.load_state_dict(new_teacher_dict)        
    eval_source_only_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=dist_test, result_dir=result_dir, save_to_file=save_to_file, args=args)

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of Iteration %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    # add avg predicted number of objects to tensorboard log
    ret_dict['eval_avg_pred_bboxes'] = total_pred_objects / max(1, len(det_annos))

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict

def eval_source_only_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, args=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** Iteration %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.get('SELF_TRAIN', None) and cfg.SELF_TRAIN.get('DSNORM', None):
        model.apply(set_ds_target)

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of Iteration %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    # add avg predicted number of objects to tensorboard log
    ret_dict['eval_avg_pred_bboxes'] = total_pred_objects / max(1, len(det_annos))

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict

if __name__ == '__main__':
    pass
