# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Additionally modified by NAVER Corp. for ViDT
# ------------------------------------------------------------------------
"""Build a VIDT detector for object detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.misc import (nested_tensor_from_tensor_list,
                        inverse_sigmoid, NestedTensor)
from methods.swin_w_ram import swin_nano, swin_tiny, swin_small, swin_base_win7, swin_large_win7
from methods.coat_w_ram import coat_lite_tiny, coat_lite_mini, coat_lite_small
from .matcher import build_matcher
from .criterion import SetCriterion
from .postprocessor import PostProcess
from .deformable_transformer import build_deforamble_transformer
from methods.vidt.fpn_fusion import FPNFusionModule
import copy
import math


def _get_clones(module, N):
    """ Clone a moudle N times """

    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Detector(nn.Module):
    """ This is a combination of "Swin with RAM" and a "Neck-free Deformable Decoder" """

    def __init__(self, backbone, transformer, num_classes, num_queries,
                 aux_loss=False, with_box_refine=False,
                 # The three techniques were not used in ViDT paper.
                 # After submitting our paper, we saw the ViDT performance could be further enhanced with them.
                 cross_scale_fusion=None, iou_aware=False, token_label=False,
                 distil=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes 
            num_queries: number of object queries (i.e., det tokens). This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            cross_scale_fusion: None or fusion module available
            iou_aware: True if iou_aware is to be used.
              see the original paper https://arxiv.org/abs/1912.05992
            token_label: True if token_label is to be used.
              see the original paper https://arxiv.org/abs/2104.10858
            distil: whether to use knowledge distillation with token matching
        """
        # backbone = SwinTransformer module
        # transformer = DeformableTransformer module
        # num_classes = 91 (for COCO obj det)
        # num_queries = 100 (NOTE: though they note that 300 is a better speed-accuracy tradeoff)
        # aux_loss = True,
        # with_box_refine = True,
        # cross_scale_fusion = None
        # iou_aware = True
        # token_label = False
        # distil = False

        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model # 256
        # (256, 91)
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        # input dim = 256, hidden dim = 256, output_dim = 4, num layers = 3 (2 hidden + 1 output)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.backbone = backbone

        # two essential techniques used [default use]
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine

        # three additional techniques not used in the ViDT paper
        # optional use, we will revise our paper for the below techniques
        self.iou_aware = iou_aware
        self.token_label = token_label

        # distillation
        self.distil = distil

        # [PATCH] token channel reduction for the input to transformer decoder
        if cross_scale_fusion is None:
            num_backbone_outs = len(backbone.num_channels) # 4
            input_proj_list = []
            for _ in range(num_backbone_outs): # in range(4)
                in_channels = backbone.num_channels[_] # 96 ; 192 ; 384 ; 256
                input_proj_list.append(nn.Sequential(
                  
                  # input dim = 96 ; 192 ; 384 ; 256
                  # output dim = 256
                  # This is 1x1 conv -> so linear layer
                  nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                  nn.GroupNorm(32, hidden_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)

            # initialize the projection layer for [PATCH] tokens
            for proj in self.input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)
            self.fusion = None
        else:
            # the cross scale fusion module has its own reduction layers
            self.fusion = cross_scale_fusion

        # channel dim reduction for [DET] tokens
        self.tgt_proj = nn.Sequential(
                # input dim = 384
                # output dim = 256 
                # This is 1x1 conv -> so linear layer
                nn.Conv2d(self.backbone.num_channels[-2], hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )

        # channel dim reductionfor [DET] learnable pos encodings
        self.query_pos_proj = nn.Sequential(
                # input & output dim = 256
                # This is 1x1 conv -> so linear layer
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )

        # initialize detection head: box regression and classification
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # initialize projection layer for [DET] tokens and encodings
        nn.init.xavier_uniform_(self.tgt_proj[0].weight, gain=1)
        nn.init.constant_(self.tgt_proj[0].bias, 0)
        nn.init.xavier_uniform_(self.query_pos_proj[0].weight, gain=1)
        nn.init.constant_(self.query_pos_proj[0].bias, 0)

        # the prediction is made for each decoding layers + the standalone detector (Swin with RAM)
        num_pred = transformer.decoder.num_layers + 1 # 7

        # set up all required nn.Module for additional techniques
        if with_box_refine:
            # get 7 deep copies of the class_embed in a ModuleList
            self.class_embed = _get_clones(self.class_embed, num_pred)
            # get 7 deep copies of the class embed in a ModuleList
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        if self.iou_aware:
            # input dim = 256, hidden dim = 256, output dim = 1, num layers = 3 (2 hidden + 1 out)
            self.iou_embed = MLP(hidden_dim, hidden_dim, 1, 3)
            if with_box_refine:
                # get 7 deep copies of iou_embed in a ModuleList
                self.iou_embed = _get_clones(self.iou_embed, num_pred)
            else:
                self.iou_embed = nn.ModuleList([self.iou_embed for _ in range(num_pred)])


    def forward(self, samples: NestedTensor):
        """ The forward step of ViDT

        Parameters:
            The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        Returns:
            A dictionary having the key and value pairs below:
            - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
            - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
            - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
                            If iou_aware is True, "pred_ious" is also returns as one of the key in "aux_outputs"
            - "enc_tokens": If token_label is True, "enc_tokens" is returned to be used

            Note that aux_loss and box refinement is used in ViDT in default. The detailed ablation of using
            the cross_scale_fusion, iou_aware & token_lablel loss will be discussed in a later version
        """

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        x = samples.tensors # RGB input
        mask = samples.mask # padding mask

        # return multi-scale [PATCH] tokens along with final [DET] tokens and their pos encodings
        # # returns:
        #   -features: intermediate [PATCH] token outputs from the second layer onwards, multi-scale feautures,
        #                list of size **4** (last two originates from the same layer 
        #                with the very last one undergoing further Linear projections).
        #                where each element is a tensor of shapes: 
        #                [B, 96, H/8, W/8] ; [B, 192, H/16, W/16] ; [B, 384, H/32, W/32] ; [B, 256, H/64, W/64]
        #   -det_tgt: final refined [DET] tokens, it is a tensor of shape: [B, 384, 100]
        #   -det_pos: final [DET] token learnable positional encoding, it is a Parameter, a tensor of shape: [1, 256, 100]
        features, det_tgt, det_pos = self.backbone(x, mask)

        # [DET] token and encoding projection to compact representation for the input to the Neck-free transformer
        # "det_tgt.unsqueeze(-1)" -> [B, 384, 100, 1] (now it follows the (B, C, H, W) format required for Conv2d)
        # "self.tgt_proj()" -> [B, 256, 100, 1]
        # ".squeeze(-1).permute(..)" -> [B,100, 256]
        det_tgt = self.tgt_proj(det_tgt.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
        # "unsqueeze" -> [1, 256, 100, 1]
        # "self.query_pos_proj()" -> [1, 256, 100, 1]
        # "squeeze + permute" -> [1, 100, 256]
        det_pos = self.query_pos_proj(det_pos.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)

        # [PATCH] token projection
        shapes = []
        # l goes from 0 to 3, 
        # src in order is: [B, 96, H/8, W/8] ; [B, 192, H/16, W/16] ; [B, 384, H/32, W/32] ; [B, 256, H/64, W/64]
        for l, src in enumerate(features): 
            shapes.append(src.shape[-2:]) # [H/8, W/8] ; [H/32, W/32] ; [H/64, W/64]

        srcs = []
        if self.fusion is None: # cross_scale_fusion is a ViDT+ feature
            # l goes from 0 to 3, 
            # src in order is: [B, 96, H/8, W/8] ; [B, 192, H/16, W/16] ; [B, 384, H/32, W/32] ; [B, 256, H/64, W/64]
            for l, src in enumerate(features):
                # [B, 256, H/8, W/8] ; [B, 256, H/16, W/16] ; [B, 256, H/32, W/32] ; [B, 256, H/64, W/64]
                srcs.append(self.input_proj[l](src))
        else:
            # multi-scale fusion is used if fusion is not None
            srcs = self.fusion(features)

        masks = [] # filled with resized masks [B, H/8, W/8] ; [B, H/16, W/16] ; [B, H/32, W/32] ; [B, H/64, W/64]
        for l, src in enumerate(srcs):
            # resize mask
            # # [H/8, W/8] ; [H/32, W/32] ; [H/64, W/64]
            shapes.append(src.shape[-2:])
            # [B, H/8, W/8] ; [B, H/32, W/32] ; [B, H/64, W/64]
            _mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
            masks.append(_mask)
            assert mask is not None

        outputs_classes = [] # 7 * t.s. [B, 100, 91]
        outputs_coords = [] # 7 * t.s. [B, 100, 4]

        # return the output of the neck-free decoder
        # returns:
        #   hs = initial [DET] tokens from Swin backbone which is fed into the decoder + 
        #        intermediate outputs from all decoder layers, t.s. [7, B, 100, 256]
        #   init_reference = initial reference points, generated from linearly projecting 
        #                        [DET] token positional embeddings from the swin backbone, t.s. [B, 100, 2]
        #   inter_references = init_reference_out bbox refined + all reference points refined in each layer of the decoder,
        #                          t.s. [7, B, 100, 4]
        #   enc_token_class_unflat = None
        hs, init_reference, inter_references, enc_token_class_unflat = \
          self.transformer(srcs, masks, det_tgt, det_pos)

        # perform predictions via the detection head
        for lvl in range(hs.shape[0]): # in range(7)
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            # get the logits for the bbox class probabilities
            outputs_class = self.class_embed[lvl](hs[lvl]) # t.s. [B, 100, 91]
            ## bbox output + reference
            tmp = self.bbox_embed[lvl](hs[lvl]) # t.s. [B, 100, 4]
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        # stack all predictions made from each decoding layers
        outputs_class = torch.stack(outputs_classes) # t.s. [7, B, 100, 91]
        outputs_coord = torch.stack(outputs_coords) # t.s. [7, B, 100, 4]

        # final prediction is made the last decoding layer
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        # aux loss is defined by using the rest predictions
        # adds 'pred_logits' and 'pred_boxes' for each of the intermediate decoder layers
        if self.aux_loss and self.transformer.decoder.num_layers > 0:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        # out has 7 pairs of 'pred_logits' and 'pred_boxes'
        
        # iou awareness loss is defined for each decoding layer similar to auxiliary decoding loss
        if self.iou_aware:
            outputs_ious = []
            for lvl in range(hs.shape[0]): # in range(7)
                # get single predicted iou value
                outputs_ious.append(self.iou_embed[lvl](hs[lvl])) # t.s. [B, 100, 1]
            outputs_iou = torch.stack(outputs_ious) # t.s. [7, B, 100, 1]
            out['pred_ious'] = outputs_iou[-1]

            if self.aux_loss:
                for i, aux in enumerate(out['aux_outputs']):
                    aux['pred_ious'] = outputs_iou[i]

        # token label loss
        if self.token_label:
            out['enc_tokens'] = {'pred_logits': enc_token_class_unflat}

        if self.distil:
            # 'patch_token': multi-scale patch tokens from each stage
            # 'body_det_token' and 'neck_det_tgt': the input det_token for multiple detection heads
            out['distil_tokens'] = {'patch_token': srcs, 'body_det_token': det_tgt, 'neck_det_token': hs}

        # dict:
        #   -pred_logits: logits for the predicted probabilities for the object category, t.s. [B, 100, 91]
        #   -pred_boxes: bbox prediction for the object, t.s. [B, 100, 4]
        #   -pred_ious: predicted iou for each of the bboxes, t.s. [B, 100, 1]
        #   -aux_outputs: list of size 6 where each element is a dict containing the key value pairs: 
        #                   "pred_logits", "pred_boxes", and "pred_ious" as described above 
        #                   for the [DET] tokens output from the swin backbone + each of the intermediate decoder layers
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class MLP(nn.Module):
  """ Very simple multi-layer perceptron (also called FFN)"""

  def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
      super().__init__()
      self.num_layers = num_layers
      h = [hidden_dim] * (num_layers - 1)
      self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

  def forward(self, x):
      for i, layer in enumerate(self.layers):
          x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
      return x

def build(args, is_teacher=False):

    # a teacher model for distilation
    if is_teacher:
        return build_teacher(args)
    #

    if args.dataset_file == 'coco':
        num_classes = 91

    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    if args.backbone_name == 'swin_nano':
        backbone, hidden_dim = swin_nano(pretrained=args.pre_trained)
    elif args.backbone_name == 'swin_tiny':
        backbone, hidden_dim = swin_tiny(pretrained=args.pre_trained)
    elif args.backbone_name == 'swin_small':
        backbone, hidden_dim = swin_small(pretrained=args.pre_trained)
    elif args.backbone_name == 'swin_base_win7_22k':
        backbone, hidden_dim = swin_base_win7(pretrained=args.pre_trained)
    elif args.backbone_name == 'swin_large_win7_22k':
        backbone, hidden_dim = swin_large_win7(pretrained=args.pre_trained)
    elif args.backbone_name == 'coat_lite_tiny':
        backbone, hidden_dim = coat_lite_tiny(pretrained=args.pre_trained)
    elif args.backbone_name == 'coat_lite_mini':
        backbone, hidden_dim = coat_lite_mini(pretrained=args.pre_trained)
    elif args.backbone_name == 'coat_lite_small':
        backbone, hidden_dim = coat_lite_small(pretrained=args.pre_trained)
    else:
        raise ValueError(f'backbone {args.backbone_name} not supported')

    backbone.finetune_det(method=args.method, # vidt or vidt_wo_neck
                          det_token_num=args.det_token_num, # 100 NOTE: they noted that using 300 was a better speed/accuracy tradeoff
                          pos_dim=args.reduced_dim, # size of embeddings for head (default 256)
                          cross_indices=args.cross_indices # stages where [DET x PATCH] cross-attention was applied, by default [3] (which is the last stage)
                          )

    cross_scale_fusion = None
    if args.cross_scale_fusion: # NOTE: VIDT+
        cross_scale_fusion = FPNFusionModule(backbone.num_channels, fuse_dim=args.reduced_dim)

    deform_transformers = build_deforamble_transformer(args)

    model = Detector(
        backbone,
        deform_transformers,
        num_classes=num_classes,
        num_queries=args.det_token_num,
        # two essential techniques used in ViDT
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        # three additional techniques (optionally)
        cross_scale_fusion=cross_scale_fusion,
        iou_aware=args.iou_aware,
        token_label=args.token_label,
        # distil
        distil=False if args.distil_model is None else True,
    )

    matcher = build_matcher(args)
    # 'loss_ce': 2, 'loss_bbox': 5
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    # 'loss_giou' : 2
    weight_dict['loss_giou'] = args.giou_loss_coef

    ##
    if args.iou_aware:
        # 'loss_iouaware' : 2
        weight_dict['loss_iouaware'] = args.iouaware_loss_coef

    if args.token_label:
        weight_dict['loss_token_focal'] = args.token_loss_coef
        weight_dict['loss_token_dice'] = args.token_loss_coef

    if args.distil_model is not None:
        weight_dict['loss_distil'] = args.distil_loss_coef

    # aux decoding loss
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1 + 1): # in range(6)
            # replicate all the losses above with '_[i]' appended to the key
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        # replicate for encoder (but NOTE: since this model is encoder-less this is not used, kept so that logging doesn't need to change)
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.iou_aware:
        losses += ['iouaware']

    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(args.dataset_file)}

    return model, criterion, postprocessors


def build_teacher(args):

    if args.dataset_file == 'coco':
        num_classes = 91

    if args.dataset_file == "coco_panoptic":
        num_classes = 250

    if args.distil_model == 'vidt_nano':
        backbone, hidden_dim = swin_nano()
    elif args.distil_model == 'vidt_tiny':
        backbone, hidden_dim = swin_tiny()
    elif args.distil_model == 'vidt_small':
        backbone, hidden_dim = swin_small()
    elif args.distil_model == 'vidt_base':
        backbone, hidden_dim = swin_base_win7()
    else:
        raise ValueError(f'backbone {args.backbone_name} not supported')

    backbone.finetune_det(method=args.method,
                          det_token_num=args.det_token_num,
                          pos_dim=args.reduced_dim,
                          cross_indices=args.cross_indices)

    cross_scale_fusion = None
    if args.cross_scale_fusion:
        cross_scale_fusion = FPNFusionModule(backbone.num_channels, fuse_dim=args.reduced_dim, all=args.cross_all_out)

    deform_transformers = build_deforamble_transformer(args)

    model = Detector(
        backbone,
        deform_transformers,
        num_classes=num_classes,
        num_queries=args.det_token_num,
        # two essential techniques used in ViDT
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        # three additional techniques (optionally)
        cross_scale_fusion=cross_scale_fusion,
        iou_aware=args.iou_aware,
        token_label=args.token_label,
        # distil
        distil=False if args.distil_model is None else True,
    )

    return model
