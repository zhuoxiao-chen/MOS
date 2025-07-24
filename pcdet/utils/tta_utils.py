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
from .train_utils import save_checkpoint, checkpoint_state
import wandb
from scipy.optimize import linear_sum_assignment

def update_ema_variables(model, ema_model, model_cfg=None, cur_epoch=None,
                         total_epochs=None, cur_it=None, total_it=None):
    assert model_cfg is not None

    multiplier = 1.0

    alpha = model_cfg['EMA_MODEL_ALPHA']
    alpha = 1 - multiplier * (1 - alpha)

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    if model_cfg.get('COPY_BN_STATS_TO_TEACHER', False):
        if model_cfg.get('BN_WARM_UP', False) and cur_epoch == 0:
            multiplier = (np.cos(cur_it / total_it * np.pi) + 1) * 0.5
            bn_ema = model_cfg.BN_EMA - multiplier * (model_cfg.BN_EMA - 0.9)
            if hasattr(model, 'module'):
                model.module.set_momemtum_value_for_bn(momemtum=(1 - bn_ema))
            else:
                model.set_momemtum_value_for_bn(momemtum=(1 - bn_ema))

        model_named_buffers = model.module.named_buffers() if hasattr(model,'module') else model.named_buffers()

        for emabf, bf in zip(ema_model.named_buffers(), model_named_buffers):
            emaname, emavalue = emabf
            name, value = bf
            assert emaname == name, 'name not equal:{} , {}'.format(emaname,
                                                                    name)
            if 'running_mean' in name or 'running_var' in name:
                emavalue.data = value.data


def TTA_augmentation(dataset, target_batch, strength='mid'):
    target_batch['gt_boxes_mask'] = np.array([True for n in target_batch['gt_boxes'][0]], dtype=np.bool_)

    b_size = target_batch['batch_size']

    copied_target_batch = copy.deepcopy(target_batch)

    points_after_aug_list = []
    points_after_process_list = []
    voxel_coords_list = []
    voxels_list = []
    voxel_num_points_list = []

    for b_idx in range(b_size):
        # target_batch['gt_boxes_mask'] = np.array([True for n in target_batch['gt_boxes'][b_idx]], dtype=np.bool_)
        target_batch['gt_boxes_mask'] = np.array([True for n in copied_target_batch['gt_boxes'][b_idx]], dtype=np.bool_)

        target_batch['gt_boxes'] = copied_target_batch['gt_boxes'][b_idx].cpu().numpy()
        cls_inx = target_batch['gt_boxes'][:, -1]
        target_batch['gt_boxes'] = target_batch['gt_boxes'][:, :7]

        points_this_batch = copied_target_batch['points'][copied_target_batch['points'][:,0] == b_idx].cpu().numpy()
        points_batch_idx = copied_target_batch['points'][copied_target_batch['points'][:,0] == b_idx][:, 0].cpu().numpy()
        target_batch['points'] = points_this_batch[:, 1:]


        augmentor = dataset.tta_data_augmentor
        if strength=='strong':
            augmentor = dataset.strong_tta_data_augmentor
        elif strength=='weak':
            augmentor = dataset.weak_tta_data_augmentor
        target_batch = augmentor.forward(
            data_dict={
                **target_batch,
            }
        )


        #   rebuild points and gt_boxes to batches and cuda
        points_batch_idx = np.ones(target_batch['points'].shape[0])*b_idx
        points_after_aug_list.append(
            torch.tensor(np.concatenate((points_batch_idx.reshape(-1, 1), target_batch['points']),axis=1)).cuda()
        )

        # copied_target_batch['points'][copied_target_batch['points'][:,0] == b_idx]=torch.tensor(np.concatenate((points_batch_idx.reshape(-1, 1), target_batch['points']),axis=1)).cuda()
        # copied_target_batch['gt_boxes'][b_idx] = target_batch['gt_boxes'].reshape(b_idx,target_batch['gt_boxes'].shape[0],target_batch['gt_boxes'].shape[1])
        # if cfg.DATA_CONFIG_TAR.DATASET == 'NuScenesDataset':
        #     copied_target_batch['gt_boxes'][b_idx] = torch.tensor(np.concatenate(
        #             (target_batch['gt_boxes'],
        #              copied_target_batch['gt_boxes'][b_idx][:, 7:9].cpu(),
        #              cls_inx.reshape(-1, 1)),
        #         axis=1)).cuda()
        # else:
        copied_target_batch['gt_boxes'][b_idx] = torch.tensor(np.concatenate((target_batch['gt_boxes'], cls_inx.reshape(-1, 1)), axis=1)).cuda()

        """ Update voxel as well """
        # target_batch.pop('voxels'); target_batch.pop('voxel_coords'); target_batch.pop('voxel_num_points'); target_batch.pop('voxel_features')
        test_batch = copy.deepcopy(target_batch)
        if 'voxels' in test_batch.keys():
            test_batch.pop('voxels')
            test_batch.pop('voxel_coords')
            test_batch.pop('voxel_num_points')
        if 'voxel_features'  in test_batch.keys():
            test_batch.pop('voxel_features')

        test_batch = dataset.point_feature_encoder.forward(test_batch)
        test_batch = dataset.data_processor.forward(
            data_dict=test_batch
        )
        points_after_process_list.append(
            torch.tensor(
                np.concatenate(
                ((np.ones(target_batch['points'].shape[0])*b_idx).reshape(-1, 1),
                 target_batch['points']),axis=1)).cuda()
        )

        voxel_batch_idx = (np.ones(test_batch['voxel_num_points'].shape) * b_idx).reshape(-1, 1)
        voxel_coords_list.append(torch.tensor(np.concatenate([voxel_batch_idx, test_batch['voxel_coords']], axis=1)).cuda())
        voxel_num_points_list.append(torch.tensor(test_batch['voxel_num_points']).cuda())
        voxels_list.append(torch.tensor(test_batch['voxels']).cuda())

    copied_target_batch['voxel_coords']  = torch.concat(voxel_coords_list)
    copied_target_batch['voxels'] = torch.concat(voxels_list)
    copied_target_batch['voxel_num_points'] = torch.concat(voxel_num_points_list)
    points = torch.concat(points_after_process_list)

    copied_target_batch['points'] = points.float()

    if 'world_scaling' in target_batch.keys():
        copied_target_batch['world_scaling'] = target_batch['world_scaling']

    return copied_target_batch



def transform_augmented_boxes(pred_dict, teacher_batch_dict, student_batch_dict):
    flip_flag, rotate_flag, scale_flag = False, False, False
    s_flip_flag, s_rotate_flag, s_scale_flag = False, False, False
    if 'world_flip_enabled' in teacher_batch_dict:
        flip_flag = True
        teacher_world_flip_enabled = teacher_batch_dict['world_flip_enabled']
    if 'world_flip_enabled' in student_batch_dict:
        s_flip_flag = True
        student_world_flip_enabled = student_batch_dict['world_flip_enabled']
    if 'world_rotation' in teacher_batch_dict:
        rotate_flag = True
        teacher_world_rotation = teacher_batch_dict['world_rotation']
    if 'world_rotation' in student_batch_dict:
        s_rotate_flag = True
        student_world_rotation = student_batch_dict['world_rotation']
    if 'world_scaling' in teacher_batch_dict:
        scale_flag = True
        teacher_world_scaling = teacher_batch_dict['world_scaling']
    if 'world_scaling' in student_batch_dict:
        s_scale_flag = True
        student_world_scaling = student_batch_dict['world_scaling']



    batch_size = len(pred_dict)

    for index in range(batch_size):
        boxes = pred_dict[index]['pred_boxes']

        # flip
        if flip_flag and teacher_world_flip_enabled[index] == 1:
            boxes[:, 1] = -boxes[:, 1]
            boxes[:, 6] = -boxes[:, 6]
        if s_flip_flag and student_world_flip_enabled[index] == 1:
            boxes[:, 1] = -boxes[:, 1]
            boxes[:, 6] = -boxes[:, 6]

        # rotation
        rotation_angle = None
        if rotate_flag:
            rotation_angle = -teacher_world_rotation[index].item()
        if s_rotate_flag:
            rotation_angle = student_world_rotation[index].item()
        if rotate_flag and s_rotate_flag:
            rotation_angle = (student_world_rotation - teacher_world_rotation)[index].item()
        if rotation_angle is not None:
            boxes[:, 0:3] = rotate_points_along_z(boxes[np.newaxis, :, 0:3], np.array([rotation_angle]))[0]
            boxes[:, 6] += rotation_angle

        # scale
        scale_factor = None
        if scale_flag:
            scale_factor = (1 / teacher_world_scaling)[index]
        if s_scale_flag:
            scale_factor = student_world_scaling[index]
        if scale_flag and s_scale_flag:
            scale_factor = (student_world_scaling / teacher_world_scaling)[index]
        if scale_flag is not None:
            boxes[:, 3:6] *= scale_factor

        pred_dict[index]['pred_boxes'] = boxes

    return pred_dict

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    if isinstance(x, np.float64) or isinstance(x, np.float32):
        return torch.tensor([x]).float(), True
    return x, False

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float().cuda()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot



def save_pseudo_label_batch(input_dict, pseudo_labels, pred_dicts=None):

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

        pseudo_labels[input_dict['frame_id'][b_idx]] = gt_infos

    return pseudo_labels