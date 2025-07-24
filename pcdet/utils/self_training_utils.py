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

from .tta_utils import update_ema_variables, TTA_augmentation, transform_augmented_boxes

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



def test_time_adaptation_one_epoch(model, val_loader, rank, leave_pbar, ps_label_dir, cur_epoch, model_func=None, lr_scheduler=None,
                       accumulated_iter=None, optim_cfg=None, tbar=None, total_it_each_epoch=None,
                       dataloader_iter=None, tb_log=None,ema_model=None, optimizer=None, ckpt_save_interval_iter=64, ckpt_save_dir=None,
                        logger=None):
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
    # meter
    val_loader.dataset.student_aug, val_loader.dataset.teacher_aug = False, False
    val_dataloader_iter = iter(val_loader)
    total_it_each_epoch = len(val_loader)

    if ema_model is not None:
        teacher_val_loader = copy.deepcopy(val_loader)
        teacher_val_loader.dataset.student_aug, teacher_val_loader.dataset.teacher_aug = False, True
        teacher_val_dataloader_iter = iter(teacher_val_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                         desc='online_test_time_adaptation', dynamic_ncols=True)

    if cfg.SELF_TRAIN.get('DSNORM', None):
        model.apply(set_ds_target)

    # we use ema_model to generate ps labels
    if ema_model is not None:
        ema_model.eval()
    else:
        model.eval()

    b_size = val_loader.batch_size
    aggregated_model = None
    model_bank = []
    latest_model = None
    adaptation_time = []; infer_time = []; total_time = []

    for cur_it in range(total_it_each_epoch):

        total_time_start = time.time()

        try:
            if ema_model is not None:
                target_batch = next(val_dataloader_iter)
                teacher_target_batch = next(teacher_val_dataloader_iter)
            else:
                target_batch = next(val_dataloader_iter)
        except StopIteration:
            target_dataloader_iter = iter(val_loader)
            target_batch = next(target_dataloader_iter)

        # generate gt_boxes for target_batch and update model weights
        with torch.no_grad():
            load_data_to_gpu(target_batch)
            if ema_model is not None:
                load_data_to_gpu(teacher_target_batch)
                teacher_pred_dicts, ret_dict = ema_model(teacher_target_batch)
                teacher_pred_dicts = transform_augmented_boxes(teacher_pred_dicts, teacher_target_batch, target_batch)

            model.eval()
            pred_dicts, ret_dict = model(target_batch)

            # if cur_it == 0 and : # for computing g
            #     init_features = pred_dicts[0]['shared_features']

        """ Start TTA """
        samples_seen = int(cur_it)*int(b_size)

        adaptation_time_start = time.time()

        """ Generate initial Psuedo labels (teacher augmentation)"""
        if cfg.TTA.METHOD in ['mos', 'avg_agg']:
            _, _ = save_pseudo_label_batch(
                target_batch, pred_dicts=pred_dicts,
                need_update=(cfg.SELF_TRAIN.get('MEMORY_ENSEMBLE', None) and
                             cfg.SELF_TRAIN.MEMORY_ENSEMBLE.ENABLED and
                             cur_it > 0)
            )
        if cfg.TTA.METHOD in ['cotta']:
            _, _ = save_pseudo_label_batch(
                teacher_target_batch, pred_dicts=teacher_pred_dicts,
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

        """ Baseline Method Mem-CLR """
        if cfg.TTA.METHOD in ['memclr']:
            model.train()
            optimizer.zero_grad()
            loss, st_tb_dict, st_disp_dict = model_func(model, target_batch)
            student_iou_feat = st_tb_dict['shared_features']
            teacher_iou_feat = teacher_pred_dicts[0]['shared_features']

            s_query = model.query_head(student_iou_feat.squeeze())
            t_query = model.query_head(teacher_iou_feat.squeeze())

            s_value = model.value_head(student_iou_feat.squeeze())
            t_value = model.value_head(teacher_iou_feat.squeeze())

            model.mem_bank = model.memory_update(model.mem_bank,t_query.contiguous().unsqueeze(-1).unsqueeze(-1),
                                                 t_value.contiguous().unsqueeze(-1).unsqueeze(-1),)

            mem_s_query = model.memory_read(model.mem_bank,s_query.contiguous().unsqueeze(-1).unsqueeze(-1),
                                            s_value.contiguous().unsqueeze(-1).unsqueeze(-1),)

            loss_mem = model.get_mem_loss(s_query, student_iou_feat.unsqueeze(-1), mem_s_query.squeeze(-1).squeeze(-1),
                                          s_value, teacher_iou_feat,t_value, model.mem_bank)

            loss = cfg.SELF_TRAIN.TAR.get('LOSS_WEIGHT', 1.0) * loss_mem
            loss.backward()

            # load_data_to_gpu(target_batch)
            # st_loss, st_tb_dict, st_disp_dict = model(target_batch)

            clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            optimizer.step()
            accumulated_iter += 1

            model.eval()
            for key, val in st_tb_dict.items():
                wandb.log({'train/' + key: val}, step=int(cur_it))

        if cfg.TTA.METHOD in ['mos', 'avg_agg'] and cfg.TTA.MOS_SETTING.AGGREGATE_START_CKPT <= samples_seen:


            # past_model_list = []
            feat_map_list = []
            pred_box_list_whole_batch = []
            pred_box_list_per_frame = []

            model_path_list = glob.glob(os.path.join(ckpt_save_dir, '*checkpoint_iter_*.pth'))
            model_path_list.sort(key=os.path.getmtime)
            num_aggregate = len(model_path_list) if len(model_path_list) < cfg.TTA.IWA_SETTING.AGGREGATE_NUM else cfg.TTA.IWA_SETTING.AGGREGATE_NUM
            model_path_list = model_path_list[-num_aggregate:]

            if num_aggregate > 3:

                """ Model Bank """
                use_model_bank = True if cfg.TTA.IWA_SETTING.get('AGGREGATE_BANK', None) and cfg.TTA.IWA_SETTING.AGGREGATE_BANK else False
                if use_model_bank and samples_seen >= cfg.TTA.IWA_SETTING.BANK_START_CKPT:
                    if len(model_path_list) < cfg.TTA.IWA_SETTING.AGGREGATE_NUM or len(model_bank)==0:
                        # initialize model bank
                        model_bank = model_path_list

                    if latest_model is None:
                        latest_model = model_path_list[-1]
                    elif latest_model != model_path_list[-1]:
                        # found new model, add it to model_bank
                        latest_model = model_path_list[-1]
                        model_bank.append(latest_model)
                    model_path_list = model_bank


                # Inference these models
                for past_model_path in model_path_list:
                    past_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=val_loader.dataset)
                    past_model.load_params_from_file(filename=past_model_path, logger=logger, report_logger=False,to_cpu=dist)
                    past_model.cuda()
                    past_model.eval()
                    pred_dicts, ret_dict = past_model(target_batch)
                    del past_model; torch.cuda.empty_cache()

                    if cfg.TTA.METHOD == 'iwa':
                        # shared_features = torch.flatten(pred_dicts[0]['shared_features'])
                        feat_map_list.append(pred_dicts[0]['shared_features'])

                        """ HM """
                        for pred_dict in pred_dicts:
                            pred_box_list_per_frame.append(pred_dict['pred_boxes'])
                        pred_box_list_whole_batch.append(pred_box_list_per_frame)
                        pred_box_list_per_frame = []

                if cfg.TTA.METHOD == 'iwa':

                    """ Concat RANK """
                    # feat_matrix=torch.zeros(len(feat_map_list), len(feat_map_list)).cuda()
                    # for i in range(len(feat_map_list)):
                    #     for j in range(len(feat_map_list)):
                    #         feat_matrix[i][j] = torch.norm(torch.concat([feat_map_list[i].squeeze(), feat_map_list[j].squeeze()]), p='nuc')
                    # G = feat_matrix.max() / feat_matrix
                    # G_inverse = torch.linalg.inv(G)

                    """ HM match loss """
                    # HM_model_matrix=torch.zeros(len(feat_map_list), len(feat_map_list)).cuda()
                    # for i in range(len(feat_map_list)):
                    #     for j in range(len(feat_map_list)):
                    #         batch_results = torch.zeros(b_size)
                    #         for b in range(b_size):
                    #             batch_results[b] = hungarian_match_diff(pred_box_list_whole_batch[i][b], pred_box_list_whole_batch[j][b])
                    #         HM_model_matrix[i][j] = batch_results.sum()
                    # G = HM_model_matrix
                    # G_inverse = torch.linalg.inv(G)

                    """ Compute G (Concat RANK and HM)"""
                    G = torch.zeros(len(feat_map_list), len(feat_map_list)).cuda()
                    for i in range(len(feat_map_list)):
                        for j in range(len(feat_map_list)):
                            batch_results = torch.zeros(b_size)
                            for b in range(b_size):
                                if pred_box_list_whole_batch[j][b].shape[0] == 0 or pred_box_list_whole_batch[i][b].shape[0] == 0:
                                    batch_results[b] = 0
                                else:
                                    batch_results[b] = hungarian_match_diff(pred_box_list_whole_batch[i][b], pred_box_list_whole_batch[j][b])
                            G[i][j] = batch_results.sum() - torch.norm(torch.concat([feat_map_list[i].squeeze(), feat_map_list[j].squeeze()]), p='nuc')
                    G_inverse = torch.linalg.inv(G)

                    """ Compute estimated importance: g"""
                    # importance = []
                    # feat_map_mean = torch.stack(feat_map_list).squeeze().mean(dim=0)
                    # for feat_map in feat_map_list:
                    #     importance.append( # less similar to init (rank larger) / more similar to current mean (rank smaller)
                    #         torch.norm(torch.concat([feat_map.squeeze(), init_features.squeeze()]), p='nuc') /
                    #         torch.norm(torch.concat([feat_map.squeeze(), feat_map_mean]), p='nuc')
                    #     )
                    # g=torch.stack(importance).cuda()

                    g=torch.ones(len(feat_map_list)).cuda()


                    """ Compute final weights: c, and apply MIN-MAX Normalization """
                    c = torch.matmul(G_inverse, g)
                    c_min, c_max = c.min(), c.max(); new_min, new_max = 0.05, 0.95
                    c = (c - c_min) / (c_max - c_min) * (new_max - new_min) + new_min
                    c /= c.sum()



                if cfg.TTA.METHOD == 'avg_agg':
                    c = torch.ones(len(model_path_list))
                    c = torch.nn.functional.normalize(c, dim=0, p=1)

                # Now given weights c, aggregate models
                aggregated_model = aggregate_model(model_path_list, c, val_loader.dataset, logger, dist, model)

                if cfg.TTA.METHOD == 'iwa':
                    del G; del G_inverse; del g
                    # del feat_matrix; del c
                torch.cuda.empty_cache()


        if cfg.TTA.METHOD in ['naive', 'tent', 'naive_ros', 'mean_teacher', 'iwa', 'avg_agg', 'cotta']:

            if aggregated_model is not None and cfg.TTA.METHOD in ['iwa', 'avg_agg']:
                # generate gt_boxes for target_batch using aggregated models
                aggregated_model.eval()
                with torch.no_grad():
                    load_data_to_gpu(target_batch)
                    pred_dicts, ret_dict = aggregated_model(target_batch)
                    _, _ = save_pseudo_label_batch(
                        target_batch, pred_dicts=pred_dicts,
                        need_update=(cfg.SELF_TRAIN.get('MEMORY_ENSEMBLE',None) and
                                     cfg.SELF_TRAIN.MEMORY_ENSEMBLE.ENABLED and
                                     cur_it > 0)
                    )

            if cfg.TTA.METHOD in ['iwa', 'avg_agg', 'cotta']:
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
            optimizer.zero_grad()

            dataset = val_loader.dataset
            if cfg.TTA.METHOD in ['mean_teacher'] or (cfg.TTA.METHOD == 'iwa' and cfg.DATA_CONFIG_TAR.get('CORRUPT', None)):
                target_batch = TTA_augmentation(dataset, target_batch, strength='strong')
            elif cfg.TTA.METHOD in ['naive_ros',  'avg_agg', 'iwa']:
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
            loss, st_tb_dict, st_disp_dict = model_func(model, target_batch)

            loss = cfg.SELF_TRAIN.TAR.get('LOSS_WEIGHT', 1.0) * loss
            loss.backward()

            # load_data_to_gpu(target_batch)
            # st_loss, st_tb_dict, st_disp_dict = model(target_batch)

            clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            optimizer.step()
            accumulated_iter += 1

            model.eval()
            for key, val in st_tb_dict.items():
                wandb.log({'train/' + key: val}, step=int(cur_it))
        """ Model bank size exceeds, need to remove the lowest weighted one """
        if cfg.TTA.get('IWA_SETTING', None) and cfg.TTA.IWA_SETTING.get('AGGREGATE_BANK', None) and cfg.TTA.IWA_SETTING.AGGREGATE_BANK and len(model_bank) > cfg.TTA.IWA_SETTING.AGGREGATE_NUM:
            if True in list(c[-1] > c):

                remove_model_index = torch.argmin(c)
                del model_bank[remove_model_index]
            else:
                del model_bank[-1]

        if ema_model is not None:
            update_ema_variables(model, ema_model, model_cfg=ema_model.model_cfg, cur_epoch=cur_epoch, total_epochs=1,
                                cur_it=cur_it, total_it=total_it_each_epoch)


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
    cost =  reg_cost

    # 3. do Hungarian matching on CPU using linear_sum_assignment
    cost = cost.detach().cpu()
    matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
    # matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred_1.device)
    # matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred_1.device)
    # assigned_pred_2_inds[matched_row_inds] = matched_col_inds + 1
    return cost[matched_row_inds].min(dim=-1)[0].sum()
