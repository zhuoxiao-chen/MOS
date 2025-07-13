from pcdet.utils.tta_utils import *
import wandb

def tent(model, val_loader, rank, leave_pbar, ps_label_dir, cur_epoch, model_func=None, lr_scheduler=None,
                       accumulated_iter=None, optim_cfg=None, tbar=None, total_it_each_epoch=None,
                       dataloader_iter=None, tb_log=None,ema_model=None, optimizer=None, ckpt_save_interval_iter=64, ckpt_save_dir=None,
                        logger=None,model_copy=None):

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

    val_dataloader_iter = iter(val_loader)
    total_it_each_epoch = len(val_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                         desc='online_test_time_adaptation', dynamic_ncols=True)

    model.eval()
    b_size = val_loader.batch_size
    adaptation_time = []; infer_time = []; total_time = []
    det_annos=[]

    for cur_it in range(total_it_each_epoch):

        total_time_start = time.time()

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
        optimizer.zero_grad()

        # Compute the loss
        loss, st_tb_dict, st_disp_dict = model_func(model, target_batch)
        loss = cfg.SELF_TRAIN.TAR.get('LOSS_WEIGHT', 1.0) * loss
        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()
        accumulated_iter += 1

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

