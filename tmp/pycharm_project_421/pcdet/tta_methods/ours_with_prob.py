from pcdet.utils.tta_utils import *
import wandb

PSEUDO_LABELS = {}
NEW_PSEUDO_LABELS = {}


def iwa(model, val_loader, rank, leave_pbar, ps_label_dir, cur_epoch, model_func=None, lr_scheduler=None,
         accumulated_iter=None, optim_cfg=None, tbar=None, total_it_each_epoch=None, dataloader_iter=None,
         tb_log=None,ema_model=None, optimizer=None, ckpt_save_interval_iter=64, ckpt_save_dir=None,logger=None,model_copy=None):

    pseudo_labels = {}

    # val_loader.dataset.student_aug, val_loader.dataset.teacher_aug = True, False
    val_dataloader_iter = iter(val_loader)
    total_it_each_epoch = len(val_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                         desc='online_test_time_adaptation', dynamic_ncols=True)

    model.eval()

    b_size = val_loader.batch_size
    aggregated_model = None
    model_bank = []
    latest_model = None
    adaptation_time = []; infer_time = []; total_time = []

    for cur_it in range(total_it_each_epoch):

        total_time_start = time.time()

        try:
            target_batch = next(val_dataloader_iter)
        except StopIteration:
            val_dataloader_iter = iter(val_loader)
            target_batch = next(val_dataloader_iter)

        # generate gt_boxes for target_batch and update model weights
        with torch.no_grad():
            load_data_to_gpu(target_batch)
            model.eval()
            pred_dicts, ret_dict = model(target_batch)

        """ Generate initial Psuedo labels (teacher augmentation)"""
        pseudo_labels = save_pseudo_label_batch(target_batch, pseudo_labels, pred_dicts=pred_dicts)

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


        if cfg.TTA.IWA_SETTING.AGGREGATE_START_CKPT <= samples_seen:

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

                # main_model_state = copy.deepcopy(model.state_dict());torch.cuda.empty_cache()
                # del model; torch.cuda.empty_cache() # save main model
                # Inference these models

                # main_model_state = copy.deepcopy(model.state_dict())
                # with torch.no_grad():
                for past_model_path in model_path_list:
                    past_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=val_loader.dataset)
                    # past_model = copy.deepcopy(model_copy)
                    # past_model=model
                    past_model.load_params_from_file(filename=past_model_path, logger=logger, report_logger=False,to_cpu=dist)
                    past_model.cuda()
                    past_model.eval()
                    pred_dicts, ret_dict = past_model(target_batch)
                    feat_map_list.append(pred_dicts[0]['shared_features'])
                    """ HM """
                    for pred_dict in pred_dicts:
                        pred_box_list_per_frame.append(pred_dict['pred_boxes'])
                    pred_box_list_whole_batch.append(pred_box_list_per_frame)
                    pred_box_list_per_frame = []
                    del past_model; del pred_dicts; del ret_dict; torch.cuda.empty_cache()
                    # get main model back

                # state_dict, update_model_state = model._load_state_dict(main_model_state, strict=False)
                # for key in state_dict:
                #     if key not in update_model_state:
                #         logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))




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
                beta = 1 if not cfg.TTA.IWA_SETTING.get('BALANCE_BETA', None) else cfg.TTA.IWA_SETTING.BALANCE_BETA
                alpha = 1 if not cfg.TTA.IWA_SETTING.get('BALANCE_ALPHA', None) else cfg.TTA.IWA_SETTING.BALANCE_ALPHA
                G = torch.zeros(len(feat_map_list), len(feat_map_list)).cuda()
                for i in range(len(feat_map_list)):
                    for j in range(len(feat_map_list)):
                        batch_results = torch.zeros(b_size)
                        for b in range(b_size):
                            if pred_box_list_whole_batch[j][b].shape[0] == 0 or pred_box_list_whole_batch[i][b].shape[0] == 0:
                                batch_results[b] = 0
                            else:
                                batch_results[b] = hungarian_match_diff(pred_box_list_whole_batch[i][b], pred_box_list_whole_batch[j][b])
                        # G[i][j] = batch_results.sum()
                        # G[i][j] = -torch.norm(torch.concat([feat_map_list[i].squeeze(),feat_map_list[j].squeeze()]), p='nuc')
                        G[i][j] = alpha*batch_results.sum() - beta*torch.norm(torch.concat([feat_map_list[i].squeeze(), feat_map_list[j].squeeze()]), p='nuc')
                G_inverse = torch.linalg.inv(G)

                """ Compute estimated importance: g"""
                g=torch.ones(len(feat_map_list)).cuda()


                """ Compute final weights: c, and apply MIN-MAX Normalization """
                c = torch.matmul(G_inverse, g)
                c_min, c_max = c.min(), c.max(); new_min, new_max = 0.05, 0.95
                c = (c - c_min) / (c_max - c_min) * (new_max - new_min) + new_min
                c /= c.sum()

                # c = torch.ones(len(feat_map_list)).cuda() / len(feat_map_list) # avg

                # Now given weights c, aggregate models
                aggregated_model = aggregate_model(model_path_list, c, val_loader.dataset, logger, dist, model)

                if cfg.TTA.METHOD == 'iwa':
                    del G; del G_inverse; del g
                    # del feat_matrix; del c
                torch.cuda.empty_cache()

        if aggregated_model is not None:
            # generate gt_boxes for target_batch using aggregated models
            aggregated_model.eval()
            with torch.no_grad():
                load_data_to_gpu(target_batch)
                # pred_dicts, ret_dict = aggregated_model(target_batch)
                model.eval(); pred_dicts, ret_dict = model(target_batch)
                """ Generate initial Psuedo labels (teacher augmentation)"""
                pseudo_labels = save_pseudo_label_batch(target_batch, pseudo_labels, pred_dicts=pred_dicts)


        # Replaces the real GT with PS boxes
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

        dataset = val_loader.dataset
        if cfg.TTA.METHOD in ['mean_teacher'] or (cfg.TTA.METHOD == 'iwa' and cfg.DATA_CONFIG_TAR.get('CORRUPT', None)):
            target_batch = TTA_augmentation(dataset, target_batch, strength='strong')
        elif cfg.TTA.METHOD in ['naive_ros',  'avg_agg', 'iwa']:
            target_batch = TTA_augmentation(dataset, target_batch)

        # Compute the loss using PS boxes
        loss, st_tb_dict, st_disp_dict = model_func(model, target_batch)

        loss = cfg.SELF_TRAIN.TAR.get('LOSS_WEIGHT', 1.0) * loss
        loss.backward()

        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()
        accumulated_iter += 1

        # model.eval()
        for key, val in st_tb_dict.items():
            wandb.log({'train/' + key: val}, step=int(cur_it))


        """ Model bank size exceeds, need to remove the lowest weighted one """
        if cfg.TTA.get('IWA_SETTING', None) and cfg.TTA.IWA_SETTING.get('AGGREGATE_BANK', None) and cfg.TTA.IWA_SETTING.AGGREGATE_BANK and len(model_bank) > cfg.TTA.IWA_SETTING.AGGREGATE_NUM:
            if True in list(c[-1] > c):

                remove_model_index = torch.argmin(c)
                del model_bank[remove_model_index]
            else:
                del model_bank[-1]


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
            ckpt_name = ckpt_save_dir / ('checkpoint_iter_%d' % samples_seen)
            state = checkpoint_state(model, optimizer, cur_it, accumulated_iter)
            save_checkpoint(state, filename=ckpt_name)
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

    exit()


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

    # main_model_state = copy.deepcopy(main_model.state_dict())

    for model_path in model_path_list:
        past_model = build_network(model_cfg=cfg.MODEL,num_class=len(cfg.CLASS_NAMES),dataset=dataset)
        # past_model = copy.deepcopy(model_copy)
        # past_model=model
        # past_model = main_model
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
        assert agg_name == name, 'name not equal:{} , {}'.format(agg_name,name)
        if 'running_mean' in name or 'running_var' in name:
            agg_value.data = value.data

    # get main model back
    # state_dict, update_model_state = main_model._load_state_dict(
    #     main_model_state, strict=False)
    # for key in state_dict:
    #     if key not in update_model_state:
    #         logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

    return agg_model

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
