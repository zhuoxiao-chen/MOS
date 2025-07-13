from pcdet.utils.tta_utils import *
import wandb
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu, boxes_iou_bev
from pcdet.utils.tta_utils import update_ema_variables, TTA_augmentation, transform_augmented_boxes
PSEUDO_LABELS = {}
NEW_PSEUDO_LABELS = {}


def dpo(model, val_loader, rank, leave_pbar, ps_label_dir, cur_epoch, model_func=None, lr_scheduler=None,
        accumulated_iter=None, optim_cfg=None, tbar=None, total_it_each_epoch=None,
        dataloader_iter=None, tb_log=None, ema_model=None, optimizer=None, ckpt_save_interval_iter=64, ckpt_save_dir=None,
        logger=None, model_copy=None):

    params = []
    names = []

    iou_head_params=[]
    iou_head_names = []
    for nm, m in model.named_modules():
        print(nm)
        if 'conv_mask' in nm:
            for np_, p in m.named_parameters():
                if np_ in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np_}")
        if 'roi_head.iou_layers' in nm:
            for np_, p in m.named_parameters():
                if np_ in ['weight', 'bias']:  # weight is scale, bias is shift
                    iou_head_params.append(p)
                    iou_head_names.append(f"{nm}.{np_}")
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np_, p in m.named_parameters():
                if np_ in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np_}")
        
        



    # check trainable parameters: 
    pytorch_total_params = sum(p.numel() for p in iou_head_params if p.requires_grad)
    print(f"Number of parameters: {pytorch_total_params}")

    # All Model

    base_optimizer = torch.optim.SGD


    optimizer = SAM(params, base_optimizer, lr=optim_cfg.LR, momentum=0.9, rho=cfg.TTA.RHO)
    if cfg.TTA.get('UPDATE_BATCHNORM_ONLY', None) and cfg.TTA.UPDATE_BATCHNORM_ONLY:
        optimizer = SAM(params, base_optimizer, lr=optim_cfg.LR, momentum=0.9,rho=cfg.TTA.RHO)
    else:
        optimizer = SAM(model.parameters(), base_optimizer, lr=optim_cfg.LR, momentum=0.9, rho=cfg.TTA.RHO)
    # model.parameters()

    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())

    if cfg.DATA_CONFIG_TAR.get('TTA_STUDENT_DATA_AUGMENTOR', None):
        val_loader.dataset.student_aug, val_loader.dataset.teacher_aug = True, False


    val_dataloader_iter = iter(val_loader)
    total_it_each_epoch = len(val_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                         desc='online_test_time_adaptation', dynamic_ncols=True)

    model.eval()
    b_size = val_loader.batch_size

    cost_per_box = []
    ema_cost = 0
    det_annos=[]
    for cur_it in range(total_it_each_epoch):
        try:
            target_batch = next(val_dataloader_iter)
        except StopIteration:
            target_dataloader_iter = iter(val_loader)
            target_batch = next(target_dataloader_iter)

        # generate gt_boxes for target_batch and update model weights
        with torch.no_grad():
            load_data_to_gpu(target_batch)
            model.eval()
            pred_dicts, ret_dict = model(target_batch)
            if cfg.TTA.get('INSTANCE_EVAL', None) and cfg.TTA.INSTANCE_EVAL:
                annos = val_loader.dataset.generate_prediction_dicts(
                    target_batch, pred_dicts, val_loader.dataset.class_names
                )
                det_annos += annos
            _, _ = save_pseudo_label_batch(
                target_batch, pred_dicts=pred_dicts,need_update=False)

        # Replaces the real GT with PS boxes
        max_box_num_batch = np.max(
            [NEW_PSEUDO_LABELS[frame_id.item()]['gt_boxes'].shape[0] for
             frame_id in target_batch['frame_id']])
        new_batch_ps_boxes = torch.zeros(b_size, max_box_num_batch, 8).cuda()

        if cfg.TTA.HUNG_MATCH:
            batch_pred_boxes_ps = []

        for b_id in range(b_size):
            ps_gt_boxes = torch.tensor(NEW_PSEUDO_LABELS[target_batch['frame_id'][b_id].item()]['gt_boxes']).cuda()[:, :8].float()

            if cfg.TTA.HUNG_MATCH:
                batch_pred_boxes_ps.append(ps_gt_boxes)

            gap = max_box_num_batch - ps_gt_boxes.shape[0]
            ps_gt_boxes = torch.concat((ps_gt_boxes, torch.zeros(gap,8).cuda()))  # concat ps_gt_boxes and empty 0 boxes to max_box_num_batch
            new_batch_ps_boxes[b_id] = ps_gt_boxes
        target_batch['gt_boxes'] = new_batch_ps_boxes
        dataset = val_loader.dataset

        if cfg.DATA_CONFIG_TAR.get('TTA_DATA_AUGMENTOR', None):
            target_batch = TTA_augmentation(dataset, target_batch)

        """ Start TTA """
        samples_seen = int(cur_it)*int(b_size)

        lr_scheduler.step(cur_it)
        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']
        # Log learning rate change
        # wandb.log({'meta_data/learning_rate': cur_lr}, step=int(cur_it))

        # Code is based on SAR
        # https://github.com/mr-eggplant/SAR/blob/20f6e24b17525f34503510afccedc0629b67b7c4/sar.py#L71

        model.train()
        optimizer.zero_grad()
        # forward
        loss, st_tb_dict, st_disp_dict = model_func(model, target_batch)
        # batch_pred_boxes_1 = st_disp_dict['batch_pred_boxes']
        loss.backward()
        optimizer.first_step(zero_grad=True)
        # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)

        if cfg.TTA.HUNG_MATCH:
            model.eval()
            pred_dicts_disturb, _ = model(target_batch)
            batch_pred_boxes_disturbed = [pred_dict['pred_boxes'] for pred_dict in pred_dicts_disturb]

            total_cost_batch_mean = 0
            with torch.no_grad():
                for b_id in range(b_size):
                    pred_boxes_disturbed = batch_pred_boxes_disturbed[b_id]
                    pred_boxes_ps = batch_pred_boxes_ps[b_id][:,:7]
                    # Hungarian cost here
                    if pred_boxes_disturbed.shape[0] == 0 or pred_boxes_ps.shape[0] == 0:
                        continue
                    iou_cost = -boxes_iou3d_gpu(pred_boxes_ps, pred_boxes_disturbed)
                    reg_cost = 2*torch.cdist(pred_boxes_ps,  pred_boxes_disturbed, p=1)
                    total_cost = (iou_cost + reg_cost).detach().cpu()
                    matched_row_inds, matched_col_inds = linear_sum_assignment(total_cost)

                    min_cost_each_row = total_cost[matched_row_inds].min(dim=-1)[0]
                    total_cost_batch_mean += int(min_cost_each_row.mean().item())
                    cost_per_box.extend(min_cost_each_row.detach().cpu().tolist())

                    accept_th = np.quantile(cost_per_box, cfg.TTA.HUNG_MATCH_RATE_POS)
                    reject_th = np.quantile(cost_per_box, 1-cfg.TTA.HUNG_MATCH_RATE_NEG)
                    i_accept = (min_cost_each_row < accept_th).nonzero(as_tuple=True)[0]
                    i_reject = (min_cost_each_row > reject_th).nonzero(as_tuple=True)[0]

                    # Update the gt boxes
                    accepted_boxes = target_batch['gt_boxes'][b_id][i_accept]
                    accepted_boxes[:, -1] = 1.0
                    target_batch['gt_boxes'][b_id][i_accept] = accepted_boxes

                    rejected_boxes = target_batch['gt_boxes'][b_id][i_reject]
                    target_batch['gt_boxes'][b_id][i_reject] = rejected_boxes.zero_()
            avg_batch_cost = total_cost_batch_mean / b_size
            wandb.log({"batch_avg_cost": avg_batch_cost}, step=samples_seen)
            if ema_cost == 0:
                ema_cost = total_cost_batch_mean / b_size
            else:
                ema_cost = (1 - 0.5) * ema_cost + 0.5 * (total_cost_batch_mean / b_size)
            wandb.log({"batch_ema_cost": ema_cost}, step=samples_seen)
            model.train()

        loss_second, st_tb_dict, st_disp_dict = model_func(model, target_batch)

        """ 
        remove grads, 
        such that in optimizer.second_step(), the input keeps the perturbation,
        and optimize model on perturbed inputs
        """
        for p_name, p in model.named_parameters():
            if 'conv_mask' in p_name:
                p.requires_grad_(False)
                p.grad = None

        loss_second.backward() # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.second_step(zero_grad=True)
        accumulated_iter += 1

        """ Enable Grads in NMC. """
        for p_name, p in model.named_parameters():
            if 'conv_mask' in p_name:
                p.requires_grad_(True)
        model.eval()
        # for key, val in st_tb_dict.items():
        #     wandb.log({'train/' + key: val}, step=int(cur_it))

        if cfg.TTA.get('INSTANCE_EVAL', None) and cfg.TTA.INSTANCE_EVAL:
            pass
        elif samples_seen > cfg.TTA.SAVE_CKPT[-1] + 1:
            exit()
        elif (samples_seen in cfg.TTA.SAVE_CKPT or samples_seen % cfg.TTA.SAVE_CKPT_INTERVAL==0) and rank == 0:
            ckpt_name = ckpt_save_dir / ('checkpoint_iter_%d' % samples_seen)
            state = checkpoint_state(model, optimizer, cur_it, accumulated_iter)
            save_checkpoint(state, filename=ckpt_name)
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

    exit()


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

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