import torch
from torch import nn
from .detector3d_template import Detector3DTemplate
from ..model_utils.model_nms_utils import class_agnostic_nms
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from torch.nn import functional as F
from .memory_bank import Memory_trans_update, Memory_trans_read
from queue import Queue
from pcdet.config import cfg

class SECONDNetIoU(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        if cfg.get('TTA', None) and cfg.TTA.METHOD == 'memclr':
            self.mem_items = 1024
            self.mem_features = 256
            self.mem_counter = 0
            self.mem_bank = F.normalize(
                torch.rand((self.mem_items, self.mem_features), dtype=torch.float),
                dim=1).cuda()
            self.memory_update = Memory_trans_update(memory_size=self.mem_items,
                                                     feature_dim=256, key_dim=256,
                                                     temp_update=0.1,
                                                     temp_gather=0.1)
            self.memory_read = Memory_trans_read()

            dim_in = 256
            dim_out = 256
            feat_dim = 256
            self.query_head = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.ReLU(inplace=True),
                nn.Linear(dim_out, feat_dim)
            )
            self.value_head = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.ReLU(inplace=True),
                nn.Linear(dim_out, feat_dim)
            )


    def forward(self, batch_dict):
        batch_dict['dataset_cfg'] = self.dataset.dataset_cfg
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
            if cur_module.__class__.__name__ == "SECONDHead":
                pooled_features = batch_dict['pooled_features']
                rois = batch_dict["rois"]

        if self.training:
            weights = batch_dict.get('SEP_LOSS_WEIGHTS', None)
            loss, tb_dict, disp_dict = self.get_training_loss(weights)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing_multicriterion(batch_dict)
            # need to modify junjie's code
            # pred_dicts['pooled_features']=pooled_features
            # pred_dicts['rois'] =  rois
            return pred_dicts, recall_dicts

    def get_training_loss(self, weights=None):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss(weights)

        if cfg.get('TTA', None) and cfg.TTA.METHOD in ['tent', 'sar']: # loss_rpn here is entropy loss
            return loss_rpn, tb_dict, disp_dict

        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        iou_weight = 1.0
        if weights is not None:
            iou_weight = weights[-1]

        loss = loss_rpn + iou_weight * loss_rcnn


        return loss, tb_dict, disp_dict


class SECONDNetIoUSTU(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.mem_items = 640
        self.mem_features = 512
        self.mem_counter = 0
        self.mem_bank = F.normalize(torch.rand((self.mem_items, self.mem_features), dtype=torch.float), dim=1).cuda()
        self.memory_update = Memory_trans_update(memory_size=self.mem_items, feature_dim=512, key_dim=512, temp_update=0.1, temp_gather=0.1)
        self.memory_read = Memory_trans_read()
        
        dim_in = 512*7*7
        dim_out = 512
        feat_dim = 512
        self.query_head = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.ReLU(inplace=True),
                nn.Linear(dim_out, feat_dim)
            )
        self.value_head = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.ReLU(inplace=True),
                nn.Linear(dim_out, feat_dim)
            )


    def forward(self, batch_dict):
        batch_dict['dataset_cfg'] = self.dataset.dataset_cfg
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
            # print(cur_module.__class__.__name__)
            if cur_module.__class__.__name__ == "BaseBEVBackbone":
                spatial_features_2d = batch_dict['spatial_features_2d']

        if self.training:
            weights = batch_dict.get('SEP_LOSS_WEIGHTS', None)
            loss, tb_dict, disp_dict = self.get_training_loss(weights)
            # print("Here might be the problem")

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict, spatial_features_2d
        else:
            pred_dicts, recall_dicts = self.post_processing_multicriterion(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, weights=None):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss(weights)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        iou_weight = 1.0
        if weights is not None:
            iou_weight = weights[-1]

        loss = loss_rpn + iou_weight * loss_rcnn
        return loss, tb_dict, disp_dict
    
    
