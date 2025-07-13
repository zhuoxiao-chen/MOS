from pcdet.utils.tta_utils import *
import wandb
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np

def sar(model, val_loader, rank, leave_pbar, ps_label_dir, cur_epoch, model_func=None, lr_scheduler=None,
                       accumulated_iter=None, optim_cfg=None, tbar=None, total_it_each_epoch=None,
                       dataloader_iter=None, tb_log=None,ema_model=None, optimizer=None, ckpt_save_interval_iter=64, ckpt_save_dir=None,
                        logger=None, model_copy=None):


    ema = None  # to record the moving average of model output entropy, as model recovery criteria

    """Copy the model and optimizer states for resetting after adaptation."""

    params = []
    names = []
    for nm, m in model.named_modules():
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np_, p in m.named_parameters():
                if np_ in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np_}")

    base_optimizer = torch.optim.SGD
    optimizer = SAM(params, base_optimizer, lr=optim_cfg.LR, momentum=0.9)

    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())

    val_dataloader_iter = iter(val_loader)
    total_it_each_epoch = len(val_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                         desc='online_test_time_adaptation', dynamic_ncols=True)

    model.eval()
    b_size = val_loader.batch_size
    adaptation_time = []; infer_time = []; total_time = []; det_annos=[]

    for cur_it in range(total_it_each_epoch):

        total_time_start = time.time()

        try:
            target_batch = next(val_dataloader_iter)
        except StopIteration:
            target_dataloader_iter = iter(val_loader)
            target_batch = next(target_dataloader_iter)

        with torch.no_grad():
            load_data_to_gpu(target_batch)

            model.eval()
            pred_dicts, ret_dict = model(target_batch)

            """generate online predictions"""
            if cfg.TTA.get('INSTANCE_EVAL', None) and cfg.TTA.INSTANCE_EVAL:
                annos = val_loader.dataset.generate_prediction_dicts(
                    target_batch, pred_dicts, val_loader.dataset.class_names
                )
                det_annos += annos

        """ Start TTA """
        samples_seen = int(cur_it)*int(b_size)

        adaptation_time_start = time.time()

        # Start to train this single frame
        lr_scheduler.step(cur_it)
        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']
        # Log learning rate change
        wandb.log({'meta_data/learning_rate': cur_lr}, step=int(cur_it))

        model.train()
        margin = 0.3 # bg are around 0.34 while objects are around 0.25

        optimizer.zero_grad()
        # forward
        loss, st_tb_dict, st_disp_dict = model_func(model, target_batch)

        loss = 0

        entropys = st_tb_dict['entropy']
        filter_ids_1 = torch.where(entropys < margin)
        entropys = entropys[filter_ids_1]
        loss = entropys.mean(0)
        loss.backward()

        optimizer.first_step(zero_grad=True)
        # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)

        loss, st_tb_dict, st_disp_dict = model_func(model, target_batch)
        entropys2 = st_tb_dict['entropy']
        entropys2 = entropys2[filter_ids_1]  # second time forward

        loss_second_value = entropys2.clone().detach().mean(0)
        filter_ids_2 = torch.where(entropys2 < margin)
        # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
        loss_second = entropys2[filter_ids_2].mean(0)

        # record moving average loss values for model recovery
        if not np.isnan(loss_second.item()):
            if ema is None:
                ema = loss_second.item()
            else:
                with torch.no_grad():
                    ema = 0.9 * ema + (1 - 0.9) * loss_second.item()

        loss_second.backward()
        optimizer.second_step(zero_grad=True)
        accumulated_iter += 1

        # perform model recovery
        if ema is not None and ema < 0.25:
            """Restore the model and optimizer states from copies."""
            model.load_state_dict(model_state, strict=True)
            optimizer.load_state_dict(optimizer_state)
            ema = None

        model.eval()
        for key, val in st_tb_dict.items():
            wandb.log({'train/' + key: val}, step=int(cur_it))


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