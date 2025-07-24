from pcdet.utils.tta_utils import *
import wandb

def cotta(model, val_loader, rank, leave_pbar, ps_label_dir, cur_epoch, model_func=None, lr_scheduler=None,
          accumulated_iter=None, optim_cfg=None, tbar=None, total_it_each_epoch=None, dataloader_iter=None,
          tb_log=None,ema_model=None, optimizer=None, ckpt_save_interval_iter=64, ckpt_save_dir=None, logger=None, model_copy=None):

    """Copy the model for resetting after adaptation."""
    model_state = copy.deepcopy(model.state_dict())

    pseudo_labels = {}

    val_loader.dataset.student_aug, val_loader.dataset.teacher_aug = True, False
    val_dataloader_iter = iter(val_loader)
    total_it_each_epoch = len(val_loader)

    assert ema_model is not None
    teacher_val_loader = copy.deepcopy(val_loader)
    teacher_val_loader.dataset.student_aug, teacher_val_loader.dataset.teacher_aug = False, True
    teacher_val_dataloader_iter = iter(teacher_val_loader)

    if cfg.TTA.get('STOP_TEACHER', None) and cfg.TTA.STOP_TEACHER:
        # for ablation, default is to
        ema_model = model

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
    adaptation_time = []; infer_time = []; total_time = [];det_annos=[]

    for cur_it in range(total_it_each_epoch):

        total_time_start = time.time()

        try:
            target_batch = next(val_dataloader_iter)
            teacher_target_batch = next(teacher_val_dataloader_iter)
            assert False not in (teacher_target_batch['frame_id'] == target_batch['frame_id'])
        except StopIteration:
            target_dataloader_iter = iter(val_loader)
            target_batch = next(target_dataloader_iter)

        # generate gt_boxes for target_batch and update model weights
        with torch.no_grad():

            load_data_to_gpu(target_batch)
            load_data_to_gpu(teacher_target_batch)

            teacher_pred_dicts, ret_dict = ema_model(teacher_target_batch)
            teacher_pred_dicts = transform_augmented_boxes(teacher_pred_dicts, teacher_target_batch, target_batch)

            model.eval(); model(target_batch)[0][-2]['pred_boxes'][:,3].mean()

        """ Start TTA """
        samples_seen = int(cur_it)*int(b_size)
        adaptation_time_start = time.time()

        """ Generate Psuedo labels"""
        pseudo_labels = save_pseudo_label_batch(teacher_target_batch, pseudo_labels, pred_dicts=teacher_pred_dicts)

        # Start to train this single frame
        lr_scheduler.step(cur_it)
        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']
        # Log learning rate change
        wandb.log({'meta_data/learning_rate': cur_lr}, step=int(cur_it))

        """Replaces the real GT with PS boxes"""
        max_box_num_batch = np.max([pseudo_labels[frame_id.item()]['gt_boxes'].shape[0] for frame_id in target_batch['frame_id']])
        new_batch_ps_boxes = torch.zeros(b_size, max_box_num_batch, 8).cuda()
        for b_id in range(b_size):
            ps_gt_boxes = torch.tensor(pseudo_labels[target_batch['frame_id'][b_id].item()]['gt_boxes']).cuda()[:,:8].float()
            gap = max_box_num_batch - ps_gt_boxes.shape[0]
            ps_gt_boxes = torch.concat((ps_gt_boxes, torch.zeros(gap, 8).cuda())) # concat ps_gt_boxes and empty 0 boxes to max_box_num_batch
            # new_batch_ps_boxes[b_id] = ps_gt_boxes.reshape(ps_gt_boxes.shape[0], ps_gt_boxes.shape[1])
            new_batch_ps_boxes[b_id] = ps_gt_boxes
        target_batch['gt_boxes'] = new_batch_ps_boxes

        model.train()
        optimizer.zero_grad()

        # Compute the loss using PS boxes
        loss, st_tb_dict, st_disp_dict = model_func(model, target_batch)

        loss = cfg.SELF_TRAIN.TAR.get('LOSS_WEIGHT', 1.0) * loss
        loss.backward()

        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()
        accumulated_iter += 1

        model.eval()
        with torch.no_grad():
            load_data_to_gpu(target_batch)
            pred_dicts, ret_dict = model(target_batch)

            """generate online predictions"""
            if cfg.TTA.get('INSTANCE_EVAL', None) and cfg.TTA.INSTANCE_EVAL:
                annos = val_loader.dataset.generate_prediction_dicts(
                    target_batch, pred_dicts, val_loader.dataset.class_names
                )
                det_annos += annos

        for key, val in st_tb_dict.items():
            wandb.log({'train/' + key: val}, step=int(cur_it))

        update_ema_variables(model, ema_model, model_cfg=ema_model.model_cfg, cur_epoch=cur_epoch, total_epochs=1,
                            cur_it=cur_it, total_it=total_it_each_epoch)



        """ Stochastic restore """
        if cfg.TTA.get('SR', None) and cfg.TTA.SR:
            rst = 0.1
            for nm, m in model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape) < rst).float().cuda()
                        with torch.no_grad():
                            p.data = model_state[f"{nm}.{npp}"] * mask + p * (1. - mask)


        """ Record Time """
        iter_end_time = time.time()
        adaptation_time.append((iter_end_time - adaptation_time_start)/b_size)
        total_time.append((iter_end_time - total_time_start)/b_size)
        infer_time.append(total_time[-1] - adaptation_time[-1])
        wandb.log({'time/' + 'adap': adaptation_time[-1]}, step=int(cur_it))
        wandb.log({'time/' + 'total': total_time[-1]}, step=int(cur_it))
        wandb.log({'time/' + 'infer': infer_time[-1]}, step=int(cur_it))

        # save trained model
        if (samples_seen in cfg.TTA.SAVE_CKPT or samples_seen % cfg.TTA.SAVE_CKPT_INTERVAL==0) and rank == 0:
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

