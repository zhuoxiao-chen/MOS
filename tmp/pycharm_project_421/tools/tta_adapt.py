import numpy as np

import _init_path
import os
from pathlib import Path
import argparse
import datetime
import glob

import torch 
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.distributed as dist

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models.model_utils.dsnorm import DSNorm
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
from train_utils.train_st_utils import train_model_st
import wandb
import time
import re
from eval_utils import eval_utils

import copy

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=16, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_fov_only', action='store_true', default=False, help='')
    parser.add_argument('--eval_src', action='store_true', default=False, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=30, help='number of checkpoints to be evaluated')
    parser.add_argument('--gpu_id', type=int, default=0, help='which gpu to run')


    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_iter_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_iter_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        iter_id = num_list[-1]
        if 'optim' in iter_id:
            continue
        if float(iter_id) not in evaluated_ckpt_list and int(float(iter_id)) >= args.start_epoch:
            return iter_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    total_time = 0
    first_eval = True
    while True:
        # check whether there is checkpoint which is not evaluated
        cur_iter_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_iter_id == -1 or int(float(cur_iter_id)) < args.start_epoch:
            if cfg.LOCAL_RANK == 0:
                tb_log.flush()

            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        # start evaluation
        cur_result_dir = eval_output_dir / ('iter_%s' % cur_iter_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.eval_source_only_epoch(
            cfg, model, test_loader, cur_iter_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir, save_to_file=args.save_to_file, args=args
        )

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_iter_id)
                wandb.log({key: val}, step=int(cur_iter_id))

        # record this interation which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_iter_id, file=f)
        logger.info('Iteration %s has been evaluated' % cur_iter_id)


def main():
    args, cfg = parse_config()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    ps_label_dir = output_dir / 'ps_label'
    ps_label_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    # gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    source_set, source_loader, source_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )
    if cfg.get('SELF_TRAIN', None):
        target_set, target_loader, target_sampler = build_dataloader(
            cfg.DATA_CONFIG_TAR, cfg.DATA_CONFIG_TAR.CLASS_NAMES, args.batch_size,
            dist_train, workers=args.workers, logger=logger, training=False
        )
    else:
        target_set = target_loader = target_sampler = None

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES),dataset=source_set)
    model_copy = None

    # Setup EMA model (teacher)
    if cfg.get('TTA', None) and cfg.TTA.METHOD in ['mean_teacher','memclr','cotta']:
        # ema_model = build_network(model_cfg=cfg.MODEL,num_class=len(cfg.CLASS_NAMES),dataset=source_set)

        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        if args.sync_bn:
            ema_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ema_model.cuda()
        if ema_model.model_cfg.get('COPY_BN_STATS_TO_TEACHER', False):
            ema_model.eval()
        else:
            ema_model.train()
    else:
        ema_model = None

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif cfg.get('SELF_TRAIN', None) and cfg.SELF_TRAIN.get('DSNORM', None):
        model = DSNorm.convert_dsnorm(model)

    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist, logger=logger)
        if cfg.get('TTA', None) and cfg.TTA.METHOD in ['mean_teacher', 'cotta', 'memclr']:
            ema_model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist, logger=logger)
    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    if cfg.get('SELF_TRAIN', None):
        total_iters_each_epoch = len(target_loader) if not args.merge_all_iters_to_one_epoch \
                                            else len(target_loader) // args.epochs
    else:
        total_iters_each_epoch = len(source_loader) if not args.merge_all_iters_to_one_epoch \
            else len(source_loader) // args.epochs

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=total_iters_each_epoch, total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # select proper trainer
    train_func = train_model_st if cfg.get('SELF_TRAIN', None) else train_model

    if cfg.LOCAL_RANK == 0:
        wandb.init(project='TTA_adapt_' + cfg.DATA_CONFIG._BASE_CONFIG_.split('/')[-1].split('.')[0], entity='PUT_YOUR_ACCOUNT_HERE')
        wandb.run.name = args.cfg_file.split('/')[-1]
        wandb.config.update(args)
        wandb.config.update(cfg)

    logger.info('**********************Start test-time single-pass adaptation %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    train_func(
        model,
        optimizer,
        source_loader,
        target_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        ps_label_dir=ps_label_dir,
        source_sampler=source_sampler,
        target_sampler=target_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        logger=logger,
        ema_model=ema_model,
        model_copy=model_copy
    )


if __name__ == '__main__':
    main()
